"""Training  and evaluation of five classifiers on the hydraulic pump-leakage dataset
taken from UCI.
Input : hydraulic_features.csv (built by build_hydraulic_features.py)
Output: model/*.joblib, model/metrics_summary.csv, test_data.csv
"""
import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

DATA_FILE = "hydraulic_features.csv"
LABEL = "pump_leakage"                  # 0 = no leakage, 1 = weak, 2 = severe
SEED = 7
TEST_FRACTION = 0.2                     # Training and Testing data ratio 80:20

print (f"Reading the 'hydraulic_features.csv' data file")
data = pd.read_csv(DATA_FILE)
features = data.drop(columns=[LABEL])       # inputs: sensor mean and std values, so dropping the label column
labels = data[LABEL]                        # output: leakage level, the models must predict

# Split into train and test, stratify labels to keeps the same share of 0, 1 and 2
# in both Train and Test parts
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=TEST_FRACTION, stratify=labels, random_state=SEED)
print (f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")

# The Streamlit app uses these rows (features + correct label) to test the model.
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

# The five models to be studied
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED)),
}

os.makedirs("model", exist_ok=True)          # makes the "model" folder if it is not there
records = []                                       # holds the scores of every model


# Train and test each model
print ("Training and Testing the models....")
for label, estimator in candidates.items():
    # Pipeline i.e. first scale the numbers, then run the model.
    # Scaling puts all sensors on a similar range,
    # sensor does not dominate the others.
    pipe = Pipeline([("scale", StandardScaler()), ("clf", estimator)])
    pipe.fit(X_train, y_train)      # learn from the training rows

    pred = pipe.predict(X_test)      # predicted class (0, 1 or 2)
    proba = pipe.predict_proba(X_test)

    # Calculate the scores on the test rows
    records.append({
        "ML Model Name": label,
        "Accuracy": accuracy_score(y_test, pred),
        # three classes -> one-vs-rest AUC, macro-averaged
        "AUC": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
        # macro = find the score for each class (0, 1, 2) separately,
        # then average them. Example: (0.90 + 0.60 + 0.90) / 3 = 0.80
        "Precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, pred, average="macro"),
        "F1": f1_score(y_test, pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    # Save the trained models, e.g. "Decision Tree" -> model/decision_tree.joblib
    joblib.dump(pipe, os.path.join("model", label.lower().replace(" ", "_") + ".joblib"))

# Save the scores of all the models to metric_summary.csv
summary = pd.DataFrame(records).round(4)
summary.to_csv(os.path.join("model", "metrics_summary.csv"), index=False)

# Print the scores as a Markdown table
print (f"The model score are as follows:")
cols = list(summary.columns)
print("| " + " | ".join(cols) + " |")
print("|" + "------|" * len(cols))
for _, row in summary.iterrows():
    print("| " + " | ".join(str(row[c]) for c in cols) + " |")
