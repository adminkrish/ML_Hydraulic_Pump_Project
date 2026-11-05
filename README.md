# Hydraulic Pump Leakage Diagnosis (Machine Learning - Assignment 2)

## a. Problem statement
Pumps are critical equipment in any plant, and catching a leak early is much 
cheaper than dealing with a breakdown. This project asks whether a machine learning 
model can work out the leakage condition of a hydraulic pump (none, weak or severe) 
just from its sensor data. The data comes from a test rig that repeats a 60-second 
load cycle, and each cycle is summarised from its sensor readings. I compare five 
classification models on the same data using six metrics, and I’ve put them in a 
Streamlit app where you can upload a test file and see the results for any model.

## b. Dataset description
- Source: Took from UCI Machine Learning Repository, “Condition monitoring of hydraulic 
  systems” (dataset id 447), created by ZeMA gGmbH (Helwig, Pignanelli and Schütze).
- What it is: The values are the recordings from a hydraulic test rig that repeats a 
  60-second load cycle. There are 2,205 cycles, each with readings from 17 sensors: 
  6 pressure, 1 motor power, 2 flow, 4 temperature, 1 vibration and 3 calculated efficiency 
  and cooling sensors. The sensors record 1, 10 or 100 readings per second.
- My features: classifier needs one row per sample, so for each sensor I took the mean 
  and standard deviation of its readings in every cycle. That gives 34 features, built by 
  build_hydraulic_features.py.
- What I predict: Predicts wheather there is pump_leakage or not, where 0 is no leakage, 
  1 is weak leakage and 2 is severe leakage.
- Size: 2,205 rows, 34 features, no missing values.
- Train/test split: 80/20, stratified so each part has the same class mix, with random_state = 7. 
  That gives 1,764 training rows and 441 test rows, and the test rows are saved as test_data.csv.
- Preprocessing: the features are scaled with StandardScaler, fitted only on the training data, 
  inside a scikit-learn Pipeline. I did not use the other condition columns (cooler, valve, accumulator) or the stable flag as features.

## c. GitHub repository and streamlit live app
- Github: https://github.com/adminkrish/ML_Hydraulic_Pump_Project
- streamlit app: https://hydraulicpumpproject.streamlit.app/

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
| Decision Tree          | * Best overall (accuracy 99.55%, highest F1 and MCC).<br/>* A few simple sensor-value rules are enough to tell the leakage levels apart.<br/>* Its AUC (0.9956) is slightly below LR (0.9968) because a tree gives hard, near-0/1 probabilities, so it ranks cases less smoothly.<br/>* Depth limited to 8 but not tuned, so the tree could be sensitive to the particular train/test split.          |
| KNN                    | * Good but weaker than the top models (accuracy 94.78%).<br/>* It decides by looking at the nearest similar cycles, so it gets confused when classes overlap in the 34-dimensional feature space.<br/>* It is also slower when predicting.                                                                                                                                                                |
| Naive Bayes            | * Worst performer i.e. accuracy is 60.77% and MCC is 0.26.<br/>* It assumes every feature is independent of the others, but sensors in a hydraulic system are connected (pressure, flow and temperature affect each other).<br/>* Because of this wrong assumption, it often misses some of the leakage classes.                                                                                          |
| Support Vector Machine | * Strong accuracy (97.51%) and MCC (0.9582), with AUC 0.9925, so it ranks the classes very well.<br/>* It separates the classes with good margins, but trails LR and the tree slightly on accuracy, which may be down to hyperparameters (C, gamma, kernel) not being tuned.<br/>* It is also costlier to train than LR.<br/> * By default an SVM does not give class probabilities, and AUC needs them, so wrapped it in CalibratedClassifierCV.<br/> * This fits the SVM several times internally, which is one more reason it trains more slowly than Logistic Regression.                                                                                  |
| **Overall winner**     | * Decision Tree gave the best results, with Logistic Regression and SVM very close behind.<br/>* The gap between Decision Tree (0.9955) and Logistic Regression (0.9909) is tiny.<br/>* With a test set of 441 samples, that is only 2 misclassified rows, so we cannot say one is definitively better.<br/>* Naive Bayes performed poorly because its independence assumption does not suit sensor data. |

## Repository structure
- app.py: Streamlit app (CSV upload, model dropdown, metrics, confusion matrix and report)
- train_models.py: trains and evaluates the five models, saves them to model/
- hydraulic.zip: raw sensor data from the UCI dataset 
- build_hydraulic_features.py: turns the raw UCI files into hydraulic_features.csv. 
- hydraulic_features.csv: contains the 2,205-row feature table
- test_data.csv: held-out test split used in the app for testing.
- .streamlit/config.toml: Theme configuration
- model/: saved pipelines (.joblib) and metrics_summary.csv
- requirements.txt

## Running locally
    pip install -r requirements.txt
    python train_models.py
    streamlit run app.py