import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

# Create directory for final ROC data
os.makedirs('final_model_roc_data', exist_ok=True)

# Load the best parameters summary
directory = os.path.dirname(os.path.realpath(__file__))
best_params_path = os.path.join(directory, 'Best_Parameters_Summary.csv')
best_params_df = pd.read_csv(best_params_path)

# Function to get most common parameter value
def get_most_common_param(model_name, param_name):
    param_rows = best_params_df[(best_params_df['Model'] == model_name) & 
                               (best_params_df['Parameter'] == param_name)]
    if param_rows.empty:
        return None
    return param_rows.sort_values('Count', ascending=False).iloc[0]['Value']

# Extract best parameters for each model
rf_params = {
    'n_estimators': 300, 'max_depth': 5, 'min_samples_leaf': 5,
}

xgb_params = {
    'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
}

knn_params = {
    'n_neighbors': 9, 'weights': 'distance', 'metric': 'manhattan'
}

logreg_params = {
    'C': 0.1, 'max_iter': 1000
}

# Load and prepare your data
# Replace with your actual data loading and preprocessing code
directory = os.path.dirname(os.path.realpath(__file__))
train_file = 'De-identified ARR Data.xlsx'

# Creating paths to each file from the directory
path = os.path.join(directory, train_file)

# Reading in the Excel file
df = pd.read_excel(path)

# Columns to be deleted
columns_to_delete = ["id", "BMI", "ETHNICITY_DESC", "NfLValue", "HighNfL Binary"]

# Other target values to be analyzed later
columns_other_2_target = [
    "Number of Future Relapses 1monthto3years",
    "Number of Future Relapses 1yrto3yrs",
    "RelapseInYearBeforeFVBinary",
    "RelapseInThe3YearsBeforeFVBinary"
]

# Deleting specified columns
df_1 = df.drop(columns=columns_to_delete, errors='ignore')
df_2 = df_1.drop(columns=columns_other_2_target, errors='ignore')

# Deleting rows with any missing values
df_cleaned = df_2.dropna()

# Remove rows containing the value 'X'
columns_with_X = [
    "PYRAMIDAL_FUNCTION", "CEREBELLAR_FUNCTION", "BRAINSTEM_FUNCTION",
    "SENSORY_FUNCTION", "BOWEL_BLADDER_FUNCTION", "VISUAL_FUNCTION", "MENTAL_FUNCTION"
]
df_cleaned_no_x = df_cleaned[~df_cleaned[columns_with_X].isin(['X']).any(axis=1)]

# Encode Sex, SMOKING_EVER, and TreatmentBeforeFV to 0 and 1
columns_MF = ["SEX"]
columns_NY = ["SMOKING_EVER", "TreatmentBeforeFV"]

for col in columns_MF:
    df_cleaned_no_x[col] = df_cleaned_no_x[col].apply(lambda x: 1 if x == 'M' else 0)

for col in columns_NY:
    df_cleaned_no_x[col] = df_cleaned_no_x[col].apply(lambda x: 1 if x == 'Y' else 0)

# Apply one-hot encoding
columns_one_hot = ["RACE_DESC", "FAMILY_MS", "DISEASE_CATEGORY_DESC_FV", "NewT2lesionYearBeforeFV", "NewGadLesionYearBeforeFV"]

for col in columns_one_hot:
    dummies = pd.get_dummies(df_cleaned_no_x[col], prefix=col)
    df_cleaned_no_x = pd.concat([df_cleaned_no_x, dummies], axis=1)
    df_cleaned_no_x.drop(col, axis=1, inplace=True)

# Columns to normalize
columns_to_normalize = ["AgeatFV", "DiseasedurationatFV", "EDSS_FV", 
                        "PYRAMIDAL_FUNCTION", "CEREBELLAR_FUNCTION", "BRAINSTEM_FUNCTION", 
                        "SENSORY_FUNCTION", "BOWEL_BLADDER_FUNCTION", "VISUAL_FUNCTION", 
                        "MENTAL_FUNCTION", 
                        "TotalnumberofrelapsesbeforeFV", "Numberofrelapsesinthe3yearsbeforeFV", 
                        "Numberofrelapsesinthe1yearbeforeFV", "timeSinceLastAttack"]

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the specified columns
df_cleaned_no_x[columns_to_normalize] = scaler.fit_transform(df_cleaned_no_x[columns_to_normalize])

# Assuming 'Future Relapse Binary' is the target column
X = df_cleaned_no_x.drop('Future Relapse Binary', axis=1).astype(np.float64)
y = df_cleaned_no_x['Future Relapse Binary'].astype(np.float64)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the classifiers with optimal parameters
rf_classifier = make_pipeline(
    SMOTE(),
    RandomForestClassifier(
        n_estimators=rf_params['n_estimators'], 
        max_depth=rf_params['max_depth'], 
        min_samples_leaf=rf_params['min_samples_leaf'],
        random_state=42
    )
)

xgb_classifier = make_pipeline(
    SMOTE(),
    XGBClassifier(
        n_estimators=xgb_params['n_estimators'], 
        max_depth=xgb_params['max_depth'], 
        learning_rate=xgb_params['learning_rate'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        eval_metric='logloss',
        random_state=42
    )
)

knn_classifier = make_pipeline(
    SMOTE(),
    KNeighborsClassifier(
        n_neighbors=knn_params['n_neighbors'], 
        weights=knn_params['weights'], 
        metric=knn_params['metric']
    )
)

logreg_classifier = make_pipeline(
    SMOTE(),
    LogisticRegression(
        C=logreg_params['C'], 
        max_iter=logreg_params['max_iter'],
        random_state=42
    )
)

# Train the models
rf_classifier.fit(X_train, y_train)
xgb_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
logreg_classifier.fit(X_train, y_train)

# Generate ROC curve data for each model
def save_roc_data(model, model_name):
    # Predict probabilities
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    
    # Save to separate CSV files
    fpr_df = pd.DataFrame({'false_positive_rate': fpr})
    tpr_df = pd.DataFrame({'true_positive_rate': tpr})
    thresholds_df = pd.DataFrame({'thresholds': thresholds})
    
    model_dir = os.path.join('final_model_roc_data', model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    fpr_df.to_csv(os.path.join(model_dir, 'fpr.csv'), index=False)
    tpr_df.to_csv(os.path.join(model_dir, 'tpr.csv'), index=False)
    thresholds_df.to_csv(os.path.join(model_dir, 'thresholds.csv'), index=False)
    
    # Also save as a single file with all ROC data
    # (Handle different array lengths by merging up to minimum length)
    min_length = min(len(fpr), len(tpr), len(thresholds))
    
    combined_df = pd.DataFrame({
        'false_positive_rate': fpr[:min_length],
        'true_positive_rate': tpr[:min_length],
        'thresholds': thresholds[:min_length]
    })
    
    combined_df.to_csv(os.path.join(model_dir, 'roc_data.csv'), index=False)
    
    return fpr, tpr, thresholds

# Save ROC data for each model
rf_fpr, rf_tpr, rf_thresholds = save_roc_data(rf_classifier, 'RandomForest')
xgb_fpr, xgb_tpr, xgb_thresholds = save_roc_data(xgb_classifier, 'XGBoost')
knn_fpr, knn_tpr, knn_thresholds = save_roc_data(knn_classifier, 'KNN')
logreg_fpr, logreg_tpr, logreg_thresholds = save_roc_data(logreg_classifier, 'LogisticRegression')

# Create voting ensemble predictions
# For voting, create a weighted average of probabilities from individual models
def predict_voting_proba(X):
    y_scores_rf = rf_classifier.predict_proba(X)[:, 1]
    y_scores_xgb = xgb_classifier.predict_proba(X)[:, 1]
    y_scores_knn = knn_classifier.predict_proba(X)[:, 1]
    y_scores_logreg = logreg_classifier.predict_proba(X)[:, 1]
    
    # Use the same weights as in the original code
    weights = np.array([2, 2, 1, 2])
    
    # Calculate weighted average
    y_scores_voting = (
        weights[0] * y_scores_rf + 
        weights[1] * y_scores_xgb + 
        weights[2] * y_scores_knn + 
        weights[3] * y_scores_logreg
    ) / np.sum(weights)
    
    return y_scores_voting

# Calculate and save ROC curve data for voting ensemble
y_scores_voting = predict_voting_proba(X_test)
fpr_voting, tpr_voting, thresholds_voting = roc_curve(y_test, y_scores_voting)

# Save voting ensemble ROC data
model_dir = os.path.join('final_model_roc_data', 'VotingEnsemble')
os.makedirs(model_dir, exist_ok=True)

fpr_df = pd.DataFrame({'false_positive_rate': fpr_voting})
tpr_df = pd.DataFrame({'true_positive_rate': tpr_voting})
thresholds_df = pd.DataFrame({'thresholds': thresholds_voting})

fpr_df.to_csv(os.path.join(model_dir, 'fpr.csv'), index=False)
tpr_df.to_csv(os.path.join(model_dir, 'tpr.csv'), index=False)
thresholds_df.to_csv(os.path.join(model_dir, 'thresholds.csv'), index=False)

min_length = min(len(fpr_voting), len(tpr_voting), len(thresholds_voting))
combined_df = pd.DataFrame({
    'false_positive_rate': fpr_voting[:min_length],
    'true_positive_rate': tpr_voting[:min_length],
    'thresholds': thresholds_voting[:min_length]
})
combined_df.to_csv(os.path.join(model_dir, 'roc_data.csv'), index=False)

print("Generated ROC curve data for all models using best parameters.")
print(f"Data saved to {os.path.join(directory, 'final_model_roc_data')}")