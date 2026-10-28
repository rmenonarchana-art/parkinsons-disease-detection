import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="NeuroCheck | Parkinson's Research Demo", page_icon="🧠", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "parkinsons_sample.csv"

st.markdown("""
<style>
.block-container{padding-top:1.5rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#10233f,#175b75,#298d91);padding:2.2rem 2.4rem;border-radius:22px;color:white;margin-bottom:1.2rem}
.hero h1{font-size:2.3rem;color:white;margin:.25rem 0 .5rem}
.hero p{font-size:1rem;max-width:820px;opacity:.92;margin:0}
.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:.75rem;font-weight:700;opacity:.8}
div[data-testid="stMetric"]{background:#f6f9fb;padding:1rem;border-radius:14px;border:1px solid #e0e8ed}
.notice{background:#fff8e6;border-left:4px solid #d49b21;padding:1rem;border-radius:8px;color:#624a12}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="eyebrow">Data Science • Machine Learning Project</div>
<h1>🧠 NeuroCheck</h1>
<p>Parkinson's Disease Detection — an educational machine-learning demonstration using voice-measurement features and a classification model.</p>
</div>
""",unsafe_allow_html=True)
st.markdown('<div class="notice"><b>Educational use only:</b> This demo is not a medical device and cannot diagnose, rule out, or monitor Parkinson’s disease. Never use its output to make healthcare decisions. Consult a qualified healthcare professional for symptoms or concerns.</div>',unsafe_allow_html=True)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

def get_xy(df):
    df=df.copy()
    target = next((c for c in df.columns if c.lower() in ["status","target","diagnosis"]),None)
    if target is None:
        raise ValueError("Target column not found. Include a 'status' column (0 or 1).")
    y=pd.to_numeric(df[target],errors="coerce")
    X=df.drop(columns=[target])
    # remove IDs/names
    X=X.drop(columns=[c for c in X.columns if c.lower() in ["name","subject","subject_id","id"]],errors="ignore")
    for c in X.columns: X[c]=pd.to_numeric(X[c],errors="coerce")
    valid=y.isin([0,1]) & y.notna()
    X=X.loc[valid].replace([np.inf,-np.inf],np.nan)
    y=y.loc[valid].astype(int)
    if len(X)<20 or y.nunique()<2:
        raise ValueError("Need at least 20 usable rows and both target classes (0 and 1).")
    return X,y

with st.sidebar:
    st.header("⚙️ Model settings")
    upload=st.file_uploader("Upload a labelled CSV dataset",type=["csv"])
    model_name=st.selectbox("Classification algorithm",["Support Vector Machine (SVM)","Random Forest","Logistic Regression"])
    st.divider()
    st.caption("Built with Python, Streamlit, Pandas and Scikit-learn.")
    st.caption("The included dataset is synthetic demo data, not clinical evidence.")

df=upload and pd.read_csv(upload) or load_data()
try:
    X,y=get_xy(df)
    models={
        "Support Vector Machine (SVM)":SVC(kernel="rbf",class_weight="balanced",random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=2000,class_weight="balanced")
    }
    pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",models[model_name])])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    pipeline.fit(X_train,y_train)
    preds=pipeline.predict(X_test)
except Exception as e:
    st.error(f"Dataset/model issue: {e}")
    st.info("Upload a labelled CSV with numeric voice features and a 'status' column containing 0 and 1.")
    st.stop()

tab_demo,tab_metrics,tab_data,tab_info=st.tabs(["🔬 Feature demo","📈 Model evaluation","🗂 Dataset","ℹ️ About project"])
with tab_demo:
    st.subheader("Explore a sample feature profile")
    st.write("Adjust numeric measurements to explore how the trained model classifies a feature profile. These inputs are for demonstration only.")
    input_values={}
    cols=st.columns(3)
    for i,c in enumerate(X.columns):
        s=X[c].dropna()
        lo=float(s.quantile(.01)) if not s.empty else 0
        hi=float(s.quantile(.99)) if not s.empty else 1
        if lo==hi: hi=lo+1
        default=float(s.median()) if not s.empty else 0
        with cols[i%3]:
            input_values[c]=st.number_input(c.replace("_"," "),value=float(np.clip(default,lo,hi)),min_value=lo,max_value=hi,key="feature_"+c,format="%.5f")
    st.caption("The included sample data is synthetic. Its output has no medical meaning.")
    if st.button("Run educational model demo",type="primary"):
        pred=int(pipeline.predict(pd.DataFrame([input_values],columns=X.columns))[0])
        st.metric("Model class output", "Class 1" if pred==1 else "Class 0")
        st.warning("This is only a classifier demonstration on synthetic data—not a diagnosis or an estimate of your personal disease risk.")
with tab_metrics:
    st.subheader("Held-out test-set metrics")
    acc=accuracy_score(y_test,preds)
    prec=precision_score(y_test,preds,zero_division=0)
    rec=recall_score(y_test,preds,zero_division=0)
    f1=f1_score(y_test,preds,zero_division=0)
    a,b,c,d=st.columns(4)
    a.metric("Accuracy",f"{acc:.3f}")
    b.metric("Precision",f"{prec:.3f}")
    c.metric("Recall",f"{rec:.3f}")
    d.metric("F1 score",f"{f1:.3f}")
    st.caption("These metrics describe only the included synthetic demonstration dataset; they are not evidence of clinical performance.")
    cm=confusion_matrix(y_test,preds,labels=[0,1])
    fig,ax=plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0,1],["Predicted 0","Predicted 1"])
    ax.set_yticks([0,1],["Actual 0","Actual 1"])
    ax.set_title("Confusion matrix")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.text(classification_report(y_test,preds,zero_division=0))
with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100),use_container_width=True)
    st.write(f"{len(df)} rows · {len(df.columns)} columns")
    quality=pd.DataFrame({"Column":df.columns,"Missing values":[int(df[c].isna().sum()) for c in df.columns],"Data type":[str(df[c].dtype) for c in df.columns]})
    st.dataframe(quality,use_container_width=True,hide_index=True)
    st.download_button("Download demo dataset",df.to_csv(index=False).encode(),file_name="parkinsons_sample.csv",mime="text/csv")
with tab_info:
    st.subheader("Project overview")
    st.markdown("""
**Problem statement:** Demonstrate how supervised machine-learning classifiers can be trained on labelled voice-measurement features for a research/education exercise.

**Workflow**
1. Load a labelled dataset and inspect missing values.
2. Separate numeric input features from the `status` target.
3. Impute missing measurements and standardize features.
4. Split data into training and held-out test sets.
5. Train SVM, Random Forest or Logistic Regression.
6. Review accuracy, precision, recall, F1 and a confusion matrix.

**Important limitations:** This project is not clinically validated. The included data is synthetic and generated only to make the app runnable. For academic analysis, replace it with a documented, ethically sourced research dataset and clearly discuss limitations, class imbalance, leakage, and external validation.
""")
    st.markdown("**Technology stack:** Python · Streamlit · Pandas · NumPy · Matplotlib · Scikit-learn")
