from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).parent

# غيّر الأسماء دي لو معنى الـ 0 و 1 في مشروعك مختلف
LABELS = {0: "Normal (0)", 1: "Suspicious / Attack (1)"}

CATEGORICAL = ["src_device", "dst_service", "protocol"]  # أعمدة نصية بتتحوّل لأرقام
NOT_SCALED = ["src_device", "dst_service"]  # دول أرقام صحيحة من غير scaling


# ---------------------------------------------------------------- تحميل الملفات
@st.cache_resource
def load_model():
    return joblib.load(BASE / "Final_model.pkl")


def find_csv():
    files = sorted(BASE.glob("banking_network_traffic*.csv")) or sorted(BASE.glob("*.csv"))
    return files[0] if files else None


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


# ---------------------------------------------------------------- التجهيز (preprocessing)
def build_preprocessor(df, features):
    """
    الـ scaler المحفوظ في model_scaler.pkl متدرب على عمود واحد بس (protocol)،
    فمينفعش نستخدمه. بنعيد بناء نفس الخطوات من الداتا الأصلية:
    1) الأعمدة النصية -> أرقام (بالترتيب الأبجدي زي LabelEncoder)
    2) باقي الأعمدة (غير src_device و dst_service) -> MinMaxScaler
    """
    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}

    def encode(frame):
        out = frame[features].copy()
        for c in CATEGORICAL:
            out[c] = out[c].map({v: i for i, v in enumerate(categories[c])})
        return out.astype(float)

    to_scale = [f for f in features if f not in NOT_SCALED]
    scaler = MinMaxScaler().fit(encode(df)[to_scale])

    def transform(frame):
        X = encode(frame)
        X[to_scale] = scaler.transform(X[to_scale])
        return X

    return categories, transform


# ---------------------------------------------------------------- الواجهة
st.set_page_config(page_title="NetGuard", layout="wide")
st.title("NetGuard")

model = load_model()
features = list(model.feature_names_in_)

csv_path = find_csv()
if csv_path is not None:
    df = read_csv(csv_path)
else:
    up = st.file_uploader("ملف الـ CSV الأصلي مش موجود جنب app.py، ارفعه هنا", type="csv")
    if up is None:
        st.stop()
    df = pd.read_csv(up)

missing = [f for f in features if f not in df.columns]
if missing:
    st.error(f"الأعمدة دي ناقصة في الـ CSV: {missing}")
    st.stop()

categories, transform = build_preprocessor(df, features)

st.subheader("Enter connection details")
values = {}
cols = st.columns(3)
for i, f in enumerate(features):
    with cols[i % 3]:
        if f in CATEGORICAL:
            values[f] = st.selectbox(f, categories[f])
        else:
            values[f] = st.number_input(f, min_value=0.0, value=float(df[f].median()))

if st.button("Predict", type="primary"):
    X = transform(pd.DataFrame([values]))
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    st.success(f"Prediction: {LABELS.get(pred, pred)}")
    st.write({LABELS.get(int(c), int(c)): f"{p:.1%}" for c, p in zip(model.classes_, proba)})

# ---------------------------------------------------------------- فحص الدقة على الداتا
with st.expander("Check: test the model on the CSV"):
    other_cols = [c for c in df.columns if c not in features]
    if not other_cols:
        st.info("مفيش عمود تاني في الـ CSV (زي الـ label) أقارن بيه.")
    else:
        target = st.selectbox("Which column is the label / target?", other_cols)
        if st.button("Run check"):
            preds = model.predict(transform(df))
            st.write(pd.crosstab(df[target], preds, rownames=["actual"], colnames=["predicted"]))
            try:
                acc = (df[target].astype(int).values == preds).mean()
                st.metric("Accuracy", f"{acc:.1%}")
            except (ValueError, TypeError):
                st.caption("الـ label مش أرقام 0/1، فاقرأ الجدول اللي فوق يدويًا.")