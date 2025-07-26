import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shap
# import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearnex import patch_sklearn, unpatch_sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from collections import Counter

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
condition = {'Poor': 0, 'Fair': 1, 'Good': 2}
Y = (df["CAT10 - Bridge Condition"]).map(condition)
print("Class distribution:", Counter(Y))

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
    scoring="f1_macro",
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

# gb_model1 = GradientBoostingClassifier(n_estimators=5, random_state=42)
# gb_model1.fit(X_final, Y)
# explainer = shap.TreeExplainer(gb_model1)
# shap_values = explainer.shap_values(X_final)

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

X_train, X_test, Y_train, Y_test = train_test_split(
    X_final, Y, test_size=0.3, random_state=42, stratify=Y
)
X_train_cb, X_test_cb, Y_train_cb, Y_test_cb = train_test_split(
    X_cb1, Y, test_size=0.3, random_state=42, stratify=Y
)

smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy="auto")
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

class_counts = Counter(Y_train)
catboost_weights = {0: class_counts[2]/class_counts[0],  # Poor
                    1: class_counts[2]/class_counts[1],  # Fair
                    2: 1} #Good

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
        f1 = f1_score(y_val_fold, pred, average='macro')
        scores.append(f1)
        trial.report(f1, i)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(rf, n_trials=10, show_progress_bar=True)  
print("best_params: ", study.best_params)
print("best_value: ", study.best_value)

def xgb(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    max_depth = trial.suggest_int("max_depth", 3, 8)  # Reduced max depth
    min_child_weight = trial.suggest_int("min_child_weight", 1, 5)
    subsample = trial.suggest_float("subsample", 0.8, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.8, 1.0)
    gamma = trial.suggest_float("gamma", 0, 2)  # Reduced gamma range
    reg_alpha = trial.suggest_float("reg_alpha", 0, 5)
    reg_lambda = trial.suggest_float("reg_lambda", 0, 5)
    
    class_counts = Counter(Y_train)
    class_weights = {
        0: class_counts[2]/class_counts[0],
        1: class_counts[2]/class_counts[1],
        2: 1
    }
    
    xgb_model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='mlogloss',  
        early_stopping_rounds=10  
    )
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, Y_train)):
        try:
            sample_weights = np.array([class_weights[y] for y in Y_train.iloc[train_idx]])
            
            xgb_model.fit(
                X_train.iloc[train_idx],
                Y_train.iloc[train_idx],
                sample_weight=sample_weights,
                eval_set=[(X_train.iloc[val_idx], Y_train.iloc[val_idx])],
                verbose=0
            )
            
            pred = xgb_model.predict(X_train.iloc[val_idx])
            f1 = f1_score(Y_train.iloc[val_idx], pred, average='macro')
            scores.append(f1)
            
            trial.report(f1, fold)
            
            if f1 < 0.3 and fold > 0:  
                raise optuna.TrialPruned()
                
        except Exception as e:
            print(f"⚠️ Fold {fold} failed: {str(e)}")
            raise optuna.TrialPruned()
    
    return np.mean(scores)

study1 = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=2,    
        interval_steps=1      
    )
)

study1.optimize(xgb, n_trials=50, show_progress_bar=True)  

if len(study1.trials) > 0:
    print("best_params_xgb: ", study1.best_params)
    print("best_value_xgb: ", study1.best_value)
else:
    print("No completed trials.")

print("best_params_xgb: ", study1.best_params)
print("best_value_xgb: ", study1.best_value)

Y_train_cb = pd.Series(Y_train_cb).values

def cb(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.05, 0.3) 
    iterations = trial.suggest_int("iterations", 80, 200)  
    depth1 = trial.suggest_int("max_depth", 3, 5) 
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    verbose = 0
    cat_feature = letters

    subsample = 0.8
    if bootstrap_type == "Bernoulli":
        subsample = trial.suggest_float("subsample", 0.7, 1.0)  
    
    cat_model=[]
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
            bootstrap_type=bootstrap_type,
            subsample=subsample if bootstrap_type == "Bernoulli" else None,
            verbose=0,
            cat_features=cat_model,
            class_weights=catboost_weights,
            random_state=42
        )
        cbc.fit(X_train_fold, y_train_fold)
        pred = cbc.predict(X_val_fold)
        f1 = f1_score(y_val_fold, pred, average='macro')
        scores.append(f1)
        trial.report(f1, i)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

study2 = optuna.create_study(direction="maximize", pruner=pruner)
study2.optimize(cb, n_trials=30, show_progress_bar=True) 
print("best_params2: ", study2.best_params)
print("best_value2: ", study2.best_value)

rfc1 = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1, class_weight='balanced')
xgb1 = XGBClassifier(**study.best_params,objective='multi:softmax',num_class=3,random_state=42,n_jobs=-1)
cbc1 = CatBoostClassifier(**study2.best_params, 
    random_state=42, 
    cat_features=[col for col in X_train_cb.columns if col in letters],
    verbose=0
)

rfc1.fit(X_train_resampled, Y_train_resampled)
xgb1.fit(X_train_resampled,Y_train_resampled,sample_weight=[class_counts[2]/class_counts[y] for y in Y_train_resampled]
)
cbc1.fit(X_train_cb, Y_train_cb)
predY_rf = rfc1.predict(X_test)
predY_xgb = xgb1.predict(X_test)
predY_cb = cbc1.predict(X_test_cb)

print("Accuracy_rf: ", accuracy_score(Y_test, predY_rf))
print("Accuracy_xgb: ", accuracy_score(Y_test, predY_xgb))
print("Accuracy_cb: ", accuracy_score(Y_test_cb, predY_cb))

print("Classification_report_rf: \n", classification_report(Y_test, predY_rf))
print("Classification Report_xg:\n", classification_report(Y_test, predY_xgb, target_names=['Poor', 'Fair', 'Good']))
print("Classification_Report_cb: \n", classification_report(Y_test_cb, predY_cb))

def get_rf_proba(X):
    X_encoded = pd.get_dummies(X, drop_first=True)
    missing_cols = set(X_final.columns) - set(X_encoded.columns)
    for col in missing_cols:
        X_encoded[col] = 0
    X_encoded = X_encoded[X_final.columns]
    return rfc1.predict_proba(X_encoded)

def get_cb_proba(X):
    X_cb = X[final_cb].copy()
    X_cb["Age_x_Traffic"] = X_cb["Bridge Age (yr)"] * X_cb["Computed - Average Daily Truck Traffic (Volume)"]
    X_cb["squared"] = (X_cb["Bridge Age (yr)"]) ** 2
    X_cb["Age_x_OperatingRating"] = X_cb["Bridge Age (yr)"] * X_cb["64 - Operating Rating (US tons)"]
    return cbc1.predict_proba(X_cb)

def get_xgb_proba(X):
    X_encoded = pd.get_dummies(X, drop_first=True)
    missing_cols = set(X_final.columns) - set(X_encoded.columns)
    for col in missing_cols:
        X_encoded[col] = 0
    X_encoded = X_encoded[X_final.columns]
    return xgb1.predict_proba(X_encoded)

rf_proba = get_rf_proba(X_test)
cb_proba = get_cb_proba(X_test)
xgb_proba = get_xgb_proba(X_test)

weights = {
    'RandomForest': 0.4, 
    'CatBoost': 0.3,
    'XGBoost': 0.3
}
avg_proba = (weights['RandomForest'] * rf_proba + 
             weights['CatBoost'] * cb_proba +
             weights['XGBoost'] * xgb_proba)

ensemble_pred = np.argmax(avg_proba, axis=1)

print("\nThree-Model Ensemble Performance:")
print("Accuracy:", accuracy_score(Y_test, ensemble_pred))
print("F1 Macro:", f1_score(Y_test, ensemble_pred, average='macro'))
print(classification_report(Y_test, ensemble_pred, target_names=['Poor', 'Fair', 'Good']))

# final_model = xgc1

# model_filename = "bridge_risk_model.joblib"
# joblib.dump(final_model, model_filename)

# explainer = shap.TreeExplainer(final_model)
# shap_values = explainer.shap_values(X_test)

# plt.figure(figsize=(10, 6))
# shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
# plt.title("SHAP Feature Importance - Final Model")
# plt.tight_layout()
# plt.savefig("shap_final.png")


