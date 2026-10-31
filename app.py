import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

# Basic page setup: browser tab title, tab icon, and use the full screen width.
st.set_page_config(page_title="Pump Leakage Diagnosis", page_icon="🔧", layout="wide")

LABEL = "pump_leakage"
CLASS_IDS = [0, 1, 2]
CLASS_NAMES = ["No leakage", "Weak leakage", "Severe leakage"]
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# Where each trained model file is stored. The dropdown in the sidebar
# uses the names on the left to pick the file on the right.
SAVED_MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def get_model(path):
    # Loads saved model from disk. The cache means each model file is read
    # only once, not every time the user clicks something in the app.
    return joblib.load(path)


def score_model(pipe, X, y):
    # Predicts with a saved pipeline and computes the six metrics.
    # The model gives two things: the predicted class for each row,
    # and the probability it assigns to each of the three classes.
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro",
                            labels=CLASS_IDS)
    except ValueError:
        auc = np.nan  # AUC needs every class present in the uploaded file
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return scores, pred


def fmt(v):
    # Shows a number with 4 decimals, or "n/a" if it could not be calculated.
    return "n/a" if pd.isna(v) else f"{v:.4f}"


# Page Header
st.title("⚙️ Hydraulic Pump Leakage Diagnosis")
st.caption("The five models were already trained. "
           "Upload a test CSV to see how each one performs.")

# Sidebar where file upload, model choice and a sample file download
with st.sidebar:
    st.header("Setup")
    upload = st.file_uploader("Test data (CSV)", type="csv")
    chosen = st.selectbox("Model", list(SAVED_MODELS))
    st.markdown(
        "The CSV needs the 34 sensor-summary columns (e.g. `PS1_mean`, `PS1_std`) "
        "plus the label column `pump_leakage` (0 none, 1 weak, 2 severe)."
    )
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as fh:
            st.download_button("Download sample test_data.csv", fh,
                               file_name="test_data.csv", mime="text/csv")

# Nothing to show until a file is uploaded, so show a hint and stop here.
if upload is None:
    st.info("Upload a test CSV from the sidebar to see results.")
    st.stop()

# Reads the uploaded file and shows a quick preview
df = pd.read_csv(upload)
st.subheader("Uploaded data")
st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
st.dataframe(df.head(10))

# Check the file is valid before using it
if LABEL not in df.columns:
    st.error(f"Label column '{LABEL}' is missing from the uploaded file.")
    st.stop()


# Load the model the user picked and ask it which input columns it was
# trained on, then make sure the uploaded file has all of them.
pipe = get_model(SAVED_MODELS[chosen])
needed = list(pipe.feature_names_in_)
absent = [c for c in needed if c not in df.columns]
if absent:
    st.error(f"Missing sensor columns: {absent}")
    st.stop()

# The label must be only 0, 1 or 2.
y_true = df[LABEL].astype(int)
if not set(y_true.unique()) <= set(CLASS_IDS):
    st.error("pump_leakage must only contain 0, 1 or 2.")
    st.stop()

# Run the chosen model and show the six metrics
X = df[needed]
scores, y_hat = score_model(pipe, X, y_true)

st.subheader(f"Results: {chosen}")
for box, name in zip(st.columns(6), METRIC_ORDER):
    box.metric(name, fmt(scores[name]))

st.caption("Precision, recall and F1 are worked out separately for each leakage "
           "type (none, weak, severe) and then averaged. AUC is worked out for "
           "each type against the other two and then averaged.")


# Confusion matrix (left) and classification report (right)
col_cm, col_rep = st.columns(2)
with col_cm:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_hat, labels=CLASS_IDS),
        display_labels=["None", "Weak", "Severe"],
    ).plot(ax=ax, cmap="Oranges", colorbar=False)
    st.pyplot(fig)
    plt.close(fig)
with col_rep:
    # Classification report: precision, recall and F1 for each leakage type
    # separately, plus overall averages.
    st.markdown("**Classification report**")
    rep = classification_report(y_true, y_hat, labels=CLASS_IDS,
                                target_names=CLASS_NAMES, output_dict=True,
                                zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3))

# Comparison table: score all five models on the same uploaded file
st.subheader("All five models on this file")
table = []
for name, path in SAVED_MODELS.items():
    s, _ = score_model(get_model(path), X, y_true)
    table.append({"Model": name, **s})
st.dataframe(pd.DataFrame(table).set_index("Model").round(4))

with st.expander("Row-level predictions"):
    out = df.copy()
    # Convert predicted numbers (0/1/2) into readable names.
    out["predicted"] = [CLASS_NAMES[i] for i in y_hat]
    st.dataframe(out[[LABEL, "predicted"]].head(50))
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name="predictions.csv", mime="text/csv")
