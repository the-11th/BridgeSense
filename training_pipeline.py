import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import time

from sklearn.model_selection import StratifiedKFold
from sklearnex import patch_sklearn, unpatch_sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import numpy as np

patch_sklearn()

df = pd.read_csv("NBI_data_imputed.csv")

X = df[
    [
        "43A - Main Span Material",
        "43B - Main Span Design",
        "45 - Number of Spans in Main Unit",
        "49 - Structure Length (ft.)",
        "Bridge Age (yr)",
        "CAT29 - Deck Area (sq. ft.)",
        "34 - Skew Angle (degrees)",
        "48 - Length of Maximum Span (ft.)",
        "51 - Bridge Roadway Width Curb to Curb (ft.)",
        "91 - Designated Inspection Frequency",
        "64 - Operating Rating (US tons)",
        "66 - Inventory Rating (US tons)",
        "30 - Year of Average Daily Traffic",
        "Computed - Average Daily Truck Traffic (Volume)",
        "Average Relative Humidity",
        "Average Temperature",
        "Maximum Temperature",
        "Minimum Temperature",
        "Mean Wind Speed",
        "64 - Operating Rating (US tons)_missing_flag",
        "66 - Inventory Rating (US tons)_missing_flag",
        "109 - Average Daily Truck Traffic (Percent ADT)_missing_flag",
        "Computed - Average Daily Truck Traffic (Volume)_missing_flag",
    ]
]
Y = (df["CAT10 - Bridge Condition"] == "Poor").astype(int)

df_cb = df.copy()

X_encoded = pd.get_dummies(X, drop_first=True)

s_model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=50)
s_model.fit(X_encoded, Y)
importances = s_model.feature_importances_
series = pd.Series(importances, index=X_encoded.columns)
series1 = series.sort_values(ascending=False)
cul_importances = series1.cumsum()
i = (cul_importances <= 0.95).sum() + 1
index1 = series1.head(i).index
X_selected = X_encoded[index1]

modelR = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=50)
selectorR = RFE(estimator=modelR, n_features_to_select=15)
selectorR.fit(X_selected, Y)
final = selectorR.get_feature_names_out()
X_final = X_selected[final].copy()

RFECV0 = RFECV(
    cv=3,  
    scoring="recall",
    step=3, 
    min_features_to_select=10,
    estimator=RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42), 
    n_jobs=-1,
)
RFECV0.fit(X_encoded, Y)
champion = RFECV0.get_feature_names_out()
letters = ["43A - Main Span Material", "43B - Main Span Design"]
champion_cb = []
for feature in champion:
    encoded = False
    for letter in letters:
        if feature.startswith(letter + "_"):
            champion_cb.append(letter)
            encoded = True
            break
    if not encoded:
        champion_cb.append(feature)

final_cb = list(set(champion_cb))
X_cb = X[final_cb].copy()

Age_x_Traffic = (
    X_final["Bridge Age (yr)"]
    * X_final["Computed - Average Daily Truck Traffic (Volume)"]
)
squared = (X_final["Bridge Age (yr)"]) ** 2
X_final["Age_x_Traffic"] = Age_x_Traffic
X_final["squared"] = squared

gb_model1 = GradientBoostingClassifier(n_estimators=5, random_state=42)
gb_model1.fit(X_final, Y)
explainer = shap.TreeExplainer(gb_model1)
shap_values = explainer.shap_values(X_final)

X_final["Age_x_OperatingRating"] = (
    X_final["Bridge Age (yr)"] * X_final["64 - Operating Rating (US tons)"]
)

X_cb["Age_x_Traffic"] = (
    X_cb["Bridge Age (yr)"] * X_cb["Computed - Average Daily Truck Traffic (Volume)"]
)
X_cb["squared"] = (X_cb["Bridge Age (yr)"]) ** 2
X_cb["Age_x_OperatingRating"] = (
    X_cb["Bridge Age (yr)"] * X_cb["64 - Operating Rating (US tons)"]
)
X_cb1 = X_cb.copy()

letters = ["43A - Main Span Material", "43B - Main Span Design"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_final, Y, test_size=0.3, random_state=42, stratify=Y
)
X_train_cb, X_test_cb, Y_train_cb, Y_test_cb = train_test_split(
    X_cb1, Y, test_size=0.3, random_state=42, stratify=Y
)

smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy="auto")
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
catboost_weight = Y_train.value_counts()[0] / Y_train.value_counts()[1]

pruner = optuna.pruners.MedianPruner()

def rf(trial):
    n_est = trial.suggest_int("n_estimators", 50, 150)  
    depth = trial.suggest_int("max_depth", 5, 12)  
    leaf = trial.suggest_int("min_samples_leaf", 1, 15)  
    feature = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    rfc = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf, max_features=feature,
        n_jobs=-1, random_state=42
    )
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train_resampled, Y_train_resampled)):
        X_train_fold, X_val_fold = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
        y_train_fold, y_val_fold = Y_train_resampled.iloc[train_idx], Y_train_resampled.iloc[val_idx]
        
        rfc.fit(X_train_fold, y_train_fold)
        pred = rfc.predict(X_val_fold)
        recall = recall_score(y_val_fold, pred)
        scores.append(recall)
        trial.report(recall, i)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(rf, n_trials=20, show_progress_bar=True)  
print("best_params: ", study.best_params)
print("best_value: ", study.best_value)

# def gb(trial):
#     learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3)  
#     n_est1 = trial.suggest_int("n_estimators", 80, 200)  
#     depth1 = trial.suggest_int("max_depth", 3, 5)  
#     subs = trial.suggest_float("subsample", 0.7, 1.0)  

#     gbc = GradientBoostingClassifier(
#         learning_rate=learning_rate,
#         n_estimators=n_est1,
#         max_depth=depth1,
#         subsample=subs,
#         random_state=42
#     )
    
#     kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     scores = []
#     for i, (train_idx, val_idx) in enumerate(kf.split(X_train_resampled, Y_train_resampled)):
#         X_train_fold, X_val_fold = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
#         y_train_fold, y_val_fold = Y_train_resampled.iloc[train_idx], Y_train_resampled.iloc[val_idx]
        
#         gbc.fit(X_train_fold, y_train_fold)
#         pred = gbc.predict(X_val_fold)
#         recall = recall_score(y_val_fold, pred)
#         scores.append(recall)
#         trial.report(recall, i)
        
#         if trial.should_prune():
#             raise optuna.TrialPruned()
    
#     return np.mean(scores)

# study1 = optuna.create_study(direction="maximize", pruner=pruner)
# study1.optimize(gb, n_trials=30, show_progress_bar=True)  
# print("best_params1: ", study1.best_params)
# print("best_value1: ", study1.best_value)

def cb(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3) 
    iterations = trial.suggest_int("iterations", 80, 200)  
    depth1 = trial.suggest_int("max_depth", 3, 5) 
    subs = trial.suggest_float("subsample", 0.7, 1.0)  
    verbose = 0
    cat_feature = letters
    scale_pos_weight = catboost_weight

    cat_model = []
    for col in X_train_cb.columns:
        if col in letters:
            cat_model.append(col)
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train_cb, Y_train)):
        X_train_fold, X_val_fold = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
        y_train_fold, y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
        
        cbc = CatBoostClassifier(
            learning_rate=learning_rate,
            iterations=iterations,
            depth=depth1,
            subsample=subs,
            verbose=0,
            cat_features=cat_model,
            scale_pos_weight=catboost_weight,
            random_state=42
        )
        cbc.fit(X_train_fold, y_train_fold)
        pred = cbc.predict(X_val_fold)
        recall = recall_score(y_val_fold, pred)
        scores.append(recall)
        trial.report(recall, i)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

study2 = optuna.create_study(direction="maximize", pruner=pruner)
study2.optimize(cb, n_trials=30, show_progress_bar=True) 
print("best_params2: ", study2.best_params)
print("best_value2: ", study2.best_value)

rfc1 = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
# gbc1 = GradientBoostingClassifier(**study1.best_params, random_state=42)
cbc1 = CatBoostClassifier(
    **study2.best_params,
    random_state=42,
    cat_features=[col for col in X_train_cb.columns if col in letters],
    verbose=0
)

rfc1.fit(X_train_resampled, Y_train_resampled)
# gbc1.fit(X_train_resampled, Y_train_resampled)
cbc1.fit(X_train_cb, Y_train)
predY_rf = rfc1.predict(X_test)
# predY_gb = gbc1.predict(X_test)
predY_cb = cbc1.predict(X_test_cb)

print("Accuracy: ", accuracy_score(Y_test, predY_rf))
# print("Accuracy: ", accuracy_score(Y_test, predY_gb))
print("Accuracy: ", accuracy_score(Y_test_cb, predY_cb))

print("Classification_report: \n", classification_report(Y_test, predY_rf))
# print("Classification_Report: \n", classification_report(Y_test, predY_gb))
print("Classification_Report: \n", classification_report(Y_test_cb, predY_cb))

# final_model = rbc1

# model_filename = "bridge_risk_model.joblib"
# joblib.dump(final_model, model_filename)

# explainer = shap.TreeExplainer(final_model)
# shap_values = explainer.shap_values(X_test)

# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
# plt.title("SHAP Feature Importance - Final Model")
# plt.tight_layout()
# plt.savefig("shap_final.png")


# print('Confusion_Matrix: \n', confusion_matrix(Y_test, predY_rf))
# print('Confusion_Matrix: \n', confusion_matrix(Y_test, predY_gb))
# temporary = df[['Bridge Age (yr)', 'Average Daily Traffic', 'Average Temperature', '49 - Structure Length (ft.)']]
# plt.figure(figsize=(20, 15))
# sns.heatmap(temporary.corr(), cmap='coolwarm')

# sns.countplot(hue='CAT10 - Bridge Condition', data=df, x='43A - Main Span Material')