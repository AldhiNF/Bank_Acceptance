import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys, types
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, fbeta_score,
    roc_auc_score, average_precision_score
)

# ===== Kompat imblearn untuk unpickle sampler =====
try:
    from imblearn.over_sampling import RandomOverSampler  # noqa: F401
except Exception:
    pass

# === Versi environment (diagnostik) ===
try:
    import sklearn, numpy, joblib
    ENV_INFO = f"Env → numpy: {numpy.__version__} | sklearn: {sklearn.__version__} | joblib: {joblib.__version__}"
except Exception:
    ENV_INFO = None

# =====================================================
# Kompat custom transformer untuk pickle (jika ada)
# =====================================================
class ColumnSlicer(BaseEstimator, TransformerMixin):
    def __init__(self, idx):
        self.idx = idx
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        idx = np.asarray(self.idx)
        return X[:, idx]

ColumnSlicer.__module__ = "main"
if "main" in sys.modules:
    setattr(sys.modules["main"], "ColumnSlicer", ColumnSlicer)
else:
    _mod = types.ModuleType("main")
    _mod.ColumnSlicer = ColumnSlicer
    sys.modules["main"] = _mod

# =====================================================
# App setup
# =====================================================
st.set_page_config(page_title="Bank Marketing Campaign", layout="wide")
st.title("📈 Bank Marketing Campaign Prediction & Cost Evaluation (Team Beta JCDS-0612)")
st.caption("App ini menjalankan pipeline model yang sudah dilatih (.sav/.pkl): single, batch CSV, dan evaluasi biaya.")

# =====================================================
# Defaults
# =====================================================
CONTACT_COST = 1.6
TP_PROFIT    = 12.5
FN_COST      = 0.0

# ===== Path model default yang dibundel bersama app =====
DEFAULT_MODEL_PATH = "final_model.sav"   # ubah kalau nama file default berbeda

# =====================================================
# Utilities
# =====================================================
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur turunan yang kamu pakai saat training (jika memang digunakan)."""
    out = df.copy()
    if "campaign" in out.columns and "campaign_cap" not in out.columns:
        out["campaign_cap"] = np.where(out["campaign"].astype(float) > 5, 5, out["campaign"]).astype(int)
    if "previous" in out.columns and "previous_cap" not in out.columns:
        out["previous_cap"] = np.where(out["previous"].astype(float) > 3, 3, out["previous"]).astype(int)
    if "pdays" in out.columns and "pdays_flag" not in out.columns:
        out["pdays_flag"] = (out["pdays"].astype(float) == 999).astype(int)
    return out

def ensure_column_order(df: pd.DataFrame, expected: pd.Index | None) -> pd.DataFrame:
    if expected is None:
        return df.copy()
    out = df.copy()
    for c in expected:
        if c not in out.columns:
            out[c] = np.nan
    return out[expected]

def _detect_expected_raw_columns(loaded_model) -> pd.Index | None:
    """Ambil daftar kolom raw dari ColumnTransformer pada step 'preprocessing'."""
    try:
        pre = None
        if hasattr(loaded_model, "named_steps") and "preprocessing" in loaded_model.named_steps:
            pre = loaded_model.named_steps["preprocessing"]
        elif hasattr(loaded_model, "steps") and len(loaded_model.steps) > 0:
            for name, step in getattr(loaded_model, "steps", []):
                if step.__class__.__name__ == "ColumnTransformer":
                    pre = step
                    break
        if pre is not None and hasattr(pre, "transformers"):
            cols = []
            for _, trans, cols_sel in pre.transformers:
                if cols_sel == "drop" or trans is None:
                    continue
                if isinstance(cols_sel, (list, tuple, pd.Index, np.ndarray)):
                    cols.extend(list(cols_sel))
            if cols:
                return pd.Index(pd.unique(cols))
    except Exception:
        pass
    return None

# === Ambil grup kolom numerik & kategorikal dari ColumnTransformer ===
def _extract_col_groups(loaded_model):
    """
    Ambil (num_cols, cat_cols) dari ColumnTransformer (step 'preprocessor'/'preprocessing').
    Kembalikan pd.Index atau None.
    """
    pre = None
    if hasattr(loaded_model, "named_steps"):
        for key in ("preprocessor", "preprocessing"):
            if key in loaded_model.named_steps:
                pre = loaded_model.named_steps[key]
                break
    if pre is None and hasattr(loaded_model, "steps"):
        for name, step in loaded_model.steps:
            if step.__class__.__name__ == "ColumnTransformer":
                pre = step
                break
    if pre is None or not hasattr(pre, "transformers"):
        return None, None

    num_cols, cat_cols = set(), set()
    for name, trans, cols_sel in pre.transformers:
        if cols_sel == "drop" or trans is None:
            continue
        if not isinstance(cols_sel, (list, tuple, pd.Index, np.ndarray)):
            continue
        cols_sel = list(cols_sel)
        # cari komponen dalam pipeline
        components = [trans]
        if hasattr(trans, "steps"):
            components = [s for _, s in trans.steps]
        comp_names = " ".join(getattr(c, "__class__", type(c)).__name__.lower() for c in components)
        tname = getattr(trans, "__class__", type(trans)).__name__.lower()

        if ("onehot" in tname or "onehot" in comp_names or
            "ordinalencoder" in tname or "ordinalencoder" in comp_names):
            cat_cols.update(cols_sel)
        elif ("standardscaler" in tname or "minmaxscaler" in tname or
              "robustscaler" in tname or "scaler" in comp_names):
            num_cols.update(cols_sel)
        else:
            # fallback: anggap kategorikal
            cat_cols.update(cols_sel)

    cat_cols = set(c for c in cat_cols if c not in num_cols)
    return (pd.Index(sorted(num_cols)) if num_cols else None,
            pd.Index(sorted(cat_cols)) if cat_cols else None)

# --- Pembersihan tipe agar scaler tidak error isnan pada object dtype ---
_TEXT_NA = {"", " ", "na", "n/a", "NA", "NaN", "nan", "None", "none", "NULL", "Null", "null"}
def _clean_object_col(s: pd.Series) -> pd.Series:
    if s.dtype != "O":
        return s
    s2 = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    s2 = s2.map(lambda v: (np.nan if isinstance(v, str) and v in _TEXT_NA else v))
    try:
        s_num = pd.to_numeric(s2, errors="ignore")
        return s_num
    except Exception:
        return s2

def _smart_cast_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = _clean_object_col(out[c])
    return out

# === Enforcer skema keras (num→float, cat→string 'unknown') ===
def enforce_schema(X: pd.DataFrame, num_cols: pd.Index | None, cat_cols: pd.Index | None) -> pd.DataFrame:
    out = X.copy()
    if num_cols is not None and len(num_cols) > 0:
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out[num_cols] = out[num_cols].astype("float64")
    if cat_cols is not None and len(cat_cols) > 0:
        for c in cat_cols:
            if c in out.columns:
                out[c] = out[c].astype("string")
        for c in cat_cols:
            if c in out.columns:
                out[c] = out[c].fillna("unknown")
    return out

def prepare_X_for_model(raw_df: pd.DataFrame, expected_cols: pd.Index | None,
                        num_cols: pd.Index | None = None,
                        cat_cols: pd.Index | None = None) -> pd.DataFrame:
    X = add_derived_columns(raw_df)
    X = ensure_column_order(X, expected_cols)
    X = _smart_cast_numeric_like(X)
    X = enforce_schema(X, num_cols, cat_cols)   # inti perbaikan
    X = _smart_cast_numeric_like(X)             # no-op utk float/string
    return X

# ===== Loader pickle -> joblib + deteksi MT19937/BitGenerator =====
def load_model(upload_file):
    import io, joblib
    pos = upload_file.tell()
    try:
        obj = pickle.load(upload_file)
        return obj
    except Exception as e1:
        upload_file.seek(pos)
        data = upload_file.read()
        upload_file.seek(pos)
        try:
            return joblib.load(io.BytesIO(data))
        except Exception as e2:
            msg1, msg2 = str(e1), str(e2)
            if ("MT19937" in msg1 or "MT19937" in msg2 or
                "BitGenerator" in msg1 or "BitGenerator" in msg2):
                raise RuntimeError(
                    "Model gagal dimuat karena ketidakcocokan versi NumPy antara environment training dan app.\n"
                    f"Detail: {msg1 or msg2}\n"
                    "Solusi: samakan versi numpy/sklearn atau simpan ulang model (random_state integer) lalu joblib.dump."
                )
            raise

def load_default_model():
    """Load model default dari file lokal DEFAULT_MODEL_PATH."""
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(
            f"Model default tidak ditemukan di '{DEFAULT_MODEL_PATH}'. "
            "Letakkan file pipeline (pickle/joblib) di path tersebut atau ubah DEFAULT_MODEL_PATH."
        )
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        return load_model(f)

def eval_cost_from_confmat(cm, contact_cost=CONTACT_COST, tp_profit=TP_PROFIT, fn_cost=FN_COST, tn_benefit=0.0):
    tn, fp, fn, tp = cm.ravel()
    total_contacts = int(tp + fp)
    total_cost = (total_contacts * contact_cost) + (fn * fn_cost) - (tp * tp_profit) - (tn * tn_benefit)
    n = int(tp + fp + fn + tn)
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Total Contacts": total_contacts,
        "Total Cost": float(total_cost),
        "Cost per Prospect": float(total_cost) / n if n else np.nan,
        "Cost per Contacted": float(total_cost) / total_contacts if total_contacts else np.nan,
    }

def evaluate_at_threshold(model, X, y_true, threshold=0.5,
                          contact_cost=CONTACT_COST, tp_profit=TP_PROFIT, fn_cost=FN_COST):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        proba = 1 / (1 + np.exp(-scores))
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cost = eval_cost_from_confmat(cm, contact_cost, tp_profit, fn_cost)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        **cost,
    }

# =====================================================
# Helper probabilitas
# =====================================================
def _proba_from_model(model, X):
    if isinstance(model, np.ndarray):
        raise TypeError("File yang diupload adalah numpy.ndarray, bukan estimator/pipeline.")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba
    if hasattr(model, "decision_function"):
        scores = np.ravel(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    if hasattr(model, "predict"):
        yhat = np.ravel(model.predict(X)).astype(int)
        return yhat.astype(float)
    raise AttributeError("Model tidak punya predict_proba/decision_function/predict.")

# =====================================================
# Sidebar – Sumber model & parameter cost
# =====================================================
st.sidebar.header("🔧 Pengaturan")

model_source = st.sidebar.radio(
    "Sumber model",
    ["Gunakan model default", "Upload model lain"],
    index=0,
    help="Default: pakai file pipeline bawaan app (DEFAULT_MODEL_PATH). Upload: coba model lain tanpa menghapus default."
)

uploaded_model_file = None
if model_source == "Upload model lain":
    uploaded_model_file = st.sidebar.file_uploader("Upload model (.sav/.pkl)", type=["sav", "pkl", "pickle"])

threshold = st.sidebar.slider("Threshold positif", 0.0, 1.0, 0.50, 0.01)

with st.sidebar.expander("Biaya/Benefit (default)"):
    contact_cost = st.number_input("Cost per call (€/kontak)", value=CONTACT_COST, min_value=0.0, step=0.1)
    tp_profit    = st.number_input("Benefit per nasabah (TP)", value=TP_PROFIT, min_value=0.0, step=0.5)
    fn_cost      = st.number_input("Opportunity cost FN", value=FN_COST, min_value=0.0, step=0.5)


loaded_model = None
expected_cols = None
NUM_COLS = None
CAT_COLS = None

try:
    # Prioritas: file upload (jika ada), kalau tidak pakai default bawaan
    if uploaded_model_file is not None:
        loaded_model = load_model(uploaded_model_file)
        active_model_name = getattr(uploaded_model_file, "name", "uploaded_model")
        st.sidebar.success("Model (upload) berhasil dimuat.")
    else:
        loaded_model = load_default_model()
        active_model_name = os.path.basename(DEFAULT_MODEL_PATH)
        st.sidebar.success(f"Model default berhasil dimuat: {active_model_name}")

    if not isinstance(loaded_model, BaseEstimator):
        raise TypeError(
            f"File berisi {type(loaded_model).__name__}, bukan estimator/pipeline scikit-learn."
        )

    st.sidebar.caption(f"Model aktif: **{active_model_name}**")
    st.sidebar.caption(f"Tipe objek model: **{type(loaded_model).__name__}**")
    if ENV_INFO:
        st.sidebar.caption(ENV_INFO)

    # Deteksi input & grup kolom
    expected_cols = _detect_expected_raw_columns(loaded_model)
    NUM_COLS, CAT_COLS = _extract_col_groups(loaded_model)

    if expected_cols is not None:
        st.sidebar.caption(f"Jumlah kolom yang diharapkan preprocessor: **{len(expected_cols)}**")
    else:
        st.sidebar.caption("Kolom mentah tidak terdeteksi (opsional).")
    if NUM_COLS is not None:
        st.sidebar.caption(f"Kolom numerik terdeteksi: {len(NUM_COLS)}")
    if CAT_COLS is not None:
        st.sidebar.caption(f"Kolom kategorikal terdeteksi: {len(CAT_COLS)}")

except Exception as e:
    if ENV_INFO:
        st.sidebar.info(ENV_INFO)
    st.sidebar.error(f"Gagal memuat model: {e}")
    loaded_model, expected_cols, NUM_COLS, CAT_COLS = None, None, None, None

# =====================================================
# Tabs (UI sama seperti sebelumnya)
# =====================================================
T1, T2, T3 = st.tabs(["🧾 Single Input", "📤 Batch CSV", "📊 Evaluasi Biaya (berlabel)"])

with T1:
    st.subheader("🧾 Prediksi Satu Baris")
    st.caption("Isi form sesuai UCI Bank Marketing. App otomatis menambah kolom turunan & membersihkan tipe.")

    with st.form("single_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("age", 18, 95, 34)
            job = st.selectbox("job", [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                "self-employed", "services", "student", "technician", "unemployed", "unknown"
            ])
            default = st.selectbox("default", ["no", "yes"], index=0)
            contact = st.selectbox("contact", ["cellular", "telephone"], index=0)
            duration = st.number_input("duration (sec)", 0, 5000, 210)
        with c2:
            marital = st.selectbox("marital", ["single", "married", "divorced", "unknown"], index=1)
            education = st.selectbox("education", [
                "basic.4y","basic.6y","basic.9y","high.school","professional.course",
                "university.degree","illiterate","unknown"
            ], index=5)
            housing = st.selectbox("housing", ["no","yes","unknown"], index=1)
            month = st.selectbox("month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], index=4)
            campaign = st.number_input("campaign", 1, 100, 2)
        with c3:
            loan = st.selectbox("loan", ["no","yes","unknown"], index=0)
            day_of_week = st.selectbox("day_of_week", ["mon","tue","wed","thu","fri"], index=0)
            pdays = st.number_input("pdays", -1, 999, 999)
            previous = st.number_input("previous", 0, 100, 0)
            poutcome = st.selectbox("poutcome", ["nonexistent","failure","success","other"], index=0)
        with c4:
            emp_var_rate = st.number_input("emp.var.rate", value=-1.8, step=0.1, format="%0.3f")
            cons_price_idx = st.number_input("cons.price.idx", value=93.444, step=0.001, format="%0.3f")
            cons_conf_idx  = st.number_input("cons.conf.idx", value=-36.0, step=0.1, format="%0.1f")
            euribor3m      = st.number_input("euribor3m", value=4.960, step=0.001, format="%0.3f")
            nr_employed    = st.number_input("nr.employed", value=5191.0, step=0.1, format="%0.1f")
        submitted = st.form_submit_button("🔮 Prediksi")

    if submitted:
        single = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan, "contact": contact,
            "month": month, "day_of_week": day_of_week, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m, "nr.employed": nr_employed
        }])
        X_single = prepare_X_for_model(single, expected_cols, NUM_COLS, CAT_COLS)

        if loaded_model is None:
            st.warning("Upload model pada sidebar terlebih dahulu.")
        else:
            try:
                proba_val = float(_proba_from_model(loaded_model, X_single)[0])
                yhat = int(proba_val >= threshold)
                st.metric("Probabilitas menerima (positif)", f"{proba_val:.2%}")
                st.write("Prediksi:", "✅ **YA**" if yhat == 1 else "❌ **TIDAK**")
                with st.expander("Detail fitur (setelah penambahan & pembersihan)"):
                    st.dataframe(X_single)
            except Exception as e:
                # retry sekali dengan enforcer lagi (edge cases)
                try:
                    X_single2 = enforce_schema(X_single, NUM_COLS, CAT_COLS)
                    proba_val = float(_proba_from_model(loaded_model, X_single2)[0])
                    yhat = int(proba_val >= threshold)
                    st.info("Prediksi berhasil setelah penyesuaian tipe kolom otomatis.")
                    st.metric("Probabilitas menerima (positif)", f"{proba_val:.2%}")
                    st.write("Prediksi:", "✅ **YA**" if yhat == 1 else "❌ **TIDAK**")
                except Exception as e2:
                    msg = str(e2)
                    if "found unknown categories" in msg or "handle_unknown" in msg:
                        st.error("Prediksi gagal karena ada kategori baru yang tidak dikenal oleh OneHotEncoder.\n"
                                 "Solusi: latih OHE dengan `handle_unknown='ignore'` atau pastikan nilai input sesuai kategori training.")
                    else:
                        st.error(f"Gagal memprediksi: {e2}\n"
                                 "Kemungkinan masih ada kolom numerik bertipe object atau teks non-numerik di kolom numerik.")

with T2:
    st.subheader("📤 Prediksi Batch (CSV)")
    st.caption("Upload CSV tanpa target. App menambah kolom turunan, membersihkan tipe, dan susun urutan kolom sesuai preprocessor.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="csv_up")
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.write("Cuplikan data asli:")
            st.dataframe(df.head())
            X = prepare_X_for_model(df, expected_cols, NUM_COLS, CAT_COLS)
            if loaded_model is None:
                st.warning("Upload model pada sidebar terlebih dahulu.")
            else:
                try:
                    proba = _proba_from_model(loaded_model, X)
                except Exception:
                    X2 = enforce_schema(X, NUM_COLS, CAT_COLS)
                    proba = _proba_from_model(loaded_model, X2)
                pred = (proba >= threshold).astype(int)
                out = df.copy()
                out["proba_accept"] = proba
                out["y"] = pred  # <-- ubah nama kolom prediksi
                st.success("Prediksi batch berhasil.")
                cols = (["y", "proba_accept"] + [c for c in out.columns if c not in ("y","proba_accept")])
                st.dataframe(out[cols].head(25))
                st.download_button(
                    "⬇️ Unduh Hasil Prediksi",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="prediksi_batch.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Gagal memproses CSV: {e}")

with T3:
    st.subheader("📊 Evaluasi Biaya (Data Berlabel)")
    st.caption("Upload CSV yang memuat kolom target 'y' (0/1).")
    lbl = st.file_uploader("Upload CSV berlabel", type=["csv"], key="csv_lbl")
    if lbl is not None:
        try:
            dfL = pd.read_csv(lbl)
            if "y" not in dfL.columns:
                st.error("CSV harus memiliki kolom target bernama 'y' (0/1).")
            else:
                y_true = dfL["y"].values.astype(int)
                X = prepare_X_for_model(dfL.drop(columns=["y"]), expected_cols, NUM_COLS, CAT_COLS)
                if loaded_model is None:
                    st.warning("Upload model pada sidebar terlebih dahulu.")
                else:
                    try:
                        proba = _proba_from_model(loaded_model, X)
                    except Exception:
                        X2 = enforce_schema(X, NUM_COLS, CAT_COLS)
                        proba = _proba_from_model(loaded_model, X2)
                    y_pred = (proba >= threshold).astype(int)
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    cost = eval_cost_from_confmat(cm, contact_cost, tp_profit, fn_cost)
                    res = {
                        "threshold": threshold,
                        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                        "recall": float(recall_score(y_true, y_pred)),
                        "f2": float(fbeta_score(y_true, y_pred, beta=2)),
                        "roc_auc": float(roc_auc_score(y_true, proba)),
                        "pr_auc": float(average_precision_score(y_true, proba)),
                        **cost,
                    }
                    st.markdown(f"### Ringkasan @ threshold = **{threshold:.2f}**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Precision (ketepatan)", f"{res['precision']:.1%}")
                        st.metric("Recall (cakupan)", f"{res['recall']:.1%}")
                        st.metric("F2-Score", f"{res['f2']:.3f}")
                    with c2:
                        st.metric("Yang dihubungi", f"{cost['Total Contacts']:,}")
                        st.metric("TP (tepat & jadi)", f"{cost['TP']:,}")
                        st.metric("FP (salah target)", f"{cost['FP']:,}")
                    with c3:
                        st.metric("FN (miss)", f"{cost['FN']:,}")
                        st.metric("TN (tepat tidak dihubungi)", f"{cost['TN']:,}")
                    st.markdown("#### Estimasi Dampak Finansial")
                    st.write(f"**Total Cost**: €{cost['Total Cost']:,.2f}")
                    st.write(f"**Rata biaya per prospek**: €{cost['Cost per Prospect']:,.2f}")
                    if np.isfinite(cost['Cost per Contacted']):
                        st.write(f"**Rata biaya per yang dihubungi**: €{cost['Cost per Contacted']:,.2f}")
                    with st.expander("Detail perhitungan (JSON)"):
                        st.json(res)
        except Exception as e:
            st.error(f"Gagal mengevaluasi: {e}")

st.markdown("""
---
**Catatan**
1) Dependensi: `pip install streamlit scikit-learn imbalanced-learn pandas numpy joblib`.
2) Pipeline ideal: `preprocessing: ColumnTransformer(cat→OneHotEncoder, num→StandardScaler) -> RandomOverSampler -> GradientBoostingClassifier`.
3) Bila OHE tidak `handle_unknown="ignore"`, kategori baru saat inferensi dapat menyebabkan error.
""")
