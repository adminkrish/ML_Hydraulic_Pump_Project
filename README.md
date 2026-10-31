# Hydraulic Pump Leakage Diagnosis (Machine Learning - Assignment 2)

## a. Problem statement
Classify the internal leakage condition of a hydraulic pump (no leakage, weak
leakage, severe leakage) from sensor readings recorded during repeated
60-second load cycles. This is a three-class classification problem, and it
is the kind of condition-monitoring task used for predictive maintenance of
pumps. Five classifiers are trained on the same data, compared on six metrics,
and served through an interactive Streamlit web application that scores uploaded 
test data.

## b. Dataset description
- Source: UCI Machine Learning Repository, "Condition monitoring of hydraulic
  systems" (id 447). Creators: ZeMA gGmbH (Helwig, Pignanelli, Schutze).
- Raw data: 2,205 cycles from a hydraulic test rig, 17 sensors (6 pressure,
  motor power, 2 volume flow, 4 temperature, vibration, and 3 derived
  efficiency/cooling sensors), sampled at 1, 10 or 100 Hz within each cycle.
- Features built for this project: mean and standard deviation of each
  sensor per cycle = 34 numeric features using script: build_hydraulic_features.py.
- Target: pump_leakage, 0 = no leakage , 1 = weak ,  2 = severe.
- Instances: 2,205. Features: 34. Missing values: none.
- Split: 80/20 stratified, random_state = 7 (1,764 train / 441 test).
  The test split is saved as test_data.csv.
- Preprocessing: StandardScaler inside a scikit-learn Pipeline, fitted on the
  training split only. The other three condition columns (cooler, valve,
  accumulator) and the stable flag were not used as features.

## c. GitHub repository
https://github.com/adminkrish/ML_Hydraulic_Pump_Project

## d. Models used
Metrics are computed on the 441-row test split. Precision, recall and F1 are
macro-averaged over the three classes; AUC is one-vs-rest, macro-averaged.

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9909   | 0.9968 | 0.9866    | 0.9905 | 0.9885 | 0.9848 |
| Decision Tree          | 0.9955   | 0.9956 | 0.9934    | 0.9932 | 0.9932 | 0.9924 |
| KNN                    | 0.9478   | 0.9926 | 0.9262    | 0.9262 | 0.9262 | 0.9122 |
| Naive Bayes            | 0.6077   | 0.7623 | 0.5815    | 0.4308 | 0.4163 | 0.2613 |
| Support Vector Machine | 0.9751   | 0.9925 | 0.9632    | 0.9668 | 0.965  | 0.9582 |


### Observations
| ML Model Name          | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                       |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression    | * Performed very well i.e. accuracy is 99.09% and it has the highest AUC (0.9968) of all five models.<br/>* This shows the three leakage levels are almost linearly separable using the sensor features.<br/>* It is simple, fast and gives balanced results across all classes.                                                                                                                          |
| Decision Tree          | * Best overall (accuracy 99.55%, highest F1 and MCC).<br/>* A few simple sensor-value rules are enough to tell the leakage levels apart.<br/>* Its AUC (0.9956) is slightly below LR (0.9968) because a tree gives hard, near-0/1 probabilities, so it ranks cases less smoothly.<br/>* No depth limit or tuning was applied, so the tree could be sensitive to the particular train/test split.          |
| KNN                    | * Good but weaker than the top models (accuracy 94.78%).<br/>* It decides by looking at the nearest similar cycles, so it gets confused when classes overlap in the 34-dimensional feature space.<br/>* It is also slower when predicting.                                                                                                                                                                |
| Naive Bayes            | * Worst performer i.e. accuracy is 60.77% and MCC is 0.26.<br/>* It assumes every feature is independent of the others, but sensors in a hydraulic system are connected (pressure, flow and temperature affect each other).<br/>* Because of this wrong assumption, it often misses some of the leakage classes.                                                                                          |
| Support Vector Machine | * Strong accuracy (97.51%) and MCC (0.9582), with AUC 0.9925, so it ranks the classes very well.<br/>* It separates the classes with good margins, but trails LR and the tree slightly on accuracy, which may be down to hyperparameters (C, gamma, kernel) not being tuned.<br/>* It is also costlier to train than LR.                                                                                  |
| **Overall winner**     | * Decision Tree gave the best results, with Logistic Regression and SVM very close behind.<br/>* The gap between Decision Tree (0.9955) and Logistic Regression (0.9909) is tiny.<br/>* With a test set of 441 samples, that is only 2 misclassified rows, so we cannot say one is definitively better.<br/>* Naive Bayes performed poorly because its independence assumption does not suit sensor data. |

## Repository structure
- app.py: Streamlit app (CSV upload, model dropdown, metrics, confusion matrix and report)
- train_models.py: trains and evaluates the five models, saves them to model/
- hydraulic.zip: raw sensor data from the UCI dataset 
- build_hydraulic_features.py: turns the raw UCI files into hydraulic_features.csv. 
- hydraulic_features.csv: contains the 2,205-row feature table
- test_data.csv: held-out test split used in the app
- .streamlit/config.toml: Theme configuration
- model/: saved pipelines (.joblib) and metrics_summary.csv
- requirements.txt

## Running locally
    pip install -r requirements.txt
    python train_models.py
    streamlit run app.py