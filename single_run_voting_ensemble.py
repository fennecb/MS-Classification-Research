import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, auc, precision_recall_curve, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, KFold
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import make_pipeline
import shap
import matplotlib.pyplot as plt

os.makedirs('roc_plots', exist_ok=True)
os.makedirs('roc_data', exist_ok=True)  # New directory for ROC data CSV files

# Disable output warnings
pd.set_option('mode.chained_assignment', None)

# Definitions for directory, data subdirectory, and file name
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


# Define the voting function
def voting_function(predictions, weights):
    # Calculate the weighted sum of the predictions
    weighted_sum = np.dot(predictions.T, weights)

    # Apply majority vote
    return (weighted_sum > 0).astype(int)

# Define the parameter grid for the random forest
param_grid_rf = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [5, 7, 9, 11],
    'randomforestclassifier__min_samples_leaf': [3, 5, 10],
}

# Define the parameter grid for XGBoost
param_grid_xgb = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__max_depth': [5, 7, 9],
    'xgbclassifier__learning_rate': [0.05, 0.1, 0.2],
    'xgbclassifier__subsample': [0.8, 1],
    'xgbclassifier__colsample_bytree': [0.8, 1]
}

# Define the parameter grid for KNN
param_grid_knn = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan']
}

param_grid_logreg = {
    'logisticregression__C': [0.001, 0.1, 1, 10, 100],
    'logisticregression__max_iter': [500, 1000]
}

# Create a pipeline including SMOTE and each classifier
pipeline_rf = make_pipeline(SMOTE(), RandomForestClassifier())
pipeline_xgb = make_pipeline(SMOTE(), XGBClassifier(eval_metric='logloss'))
pipeline_knn = make_pipeline(SMOTE(), KNeighborsClassifier())
pipeline_logreg = make_pipeline(SMOTE(), LogisticRegression())

# Create GridSearchCV objects
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_logreg = GridSearchCV(pipeline_logreg, param_grid_logreg, cv=5, scoring='roc_auc', n_jobs=-1)

# Create empty Data Frames to store results of each fold
rf_fold_metrics = []
xgb_fold_metrics = []
knn_fold_metrics = []
logreg_fold_metrics = []
voting_fold_metrics = []

# Create empty lists to store ROC curve data
rf_roc_data = []
xgb_roc_data = []
knn_roc_data = []
logreg_roc_data = []
voting_roc_data = []

# Initialize lists to store all SHAP values and their test indices
shap_values_list = []
test_indices_list = []

# Perform 10-fold cross-validation
for k, (train_index, test_index) in enumerate(KFold(n_splits=10, shuffle=True, random_state=42).split(X, y)):
    print(f'------------------\nFold {k+1}\n------------------')
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Use GridSearchCV with SMOTE to find the best parameters
    grid_search_rf.fit(X_train, y_train)
    print("Best Random Forest Parameters:", grid_search_rf.best_params_)
    rf_classifier = grid_search_rf.best_estimator_

    grid_search_xgb.fit(X_train, y_train)
    print("Best XGBoost Parameters:", grid_search_xgb.best_params_)
    xgb_classifier = grid_search_xgb.best_estimator_

    grid_search_knn.fit(X_train, y_train)
    print("Best KNN Parameters:", grid_search_knn.best_params_)
    knn_classifier = grid_search_knn.best_estimator_

    grid_search_logreg.fit(X_train, y_train)
    print("Best Logistic Regression Parameters:", grid_search_logreg.best_params_)
    logreg_classifier = grid_search_logreg.best_estimator_
    
    # Predict probabilities on the test set
    y_scores_rf = rf_classifier.predict_proba(X_test)[:, 1]
    y_scores_xgb = xgb_classifier.predict_proba(X_test)[:, 1]
    y_scores_knn = knn_classifier.predict_proba(X_test)[:, 1]
    y_scores_logreg = logreg_classifier.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_scores_rf)
    fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_scores_xgb)
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_scores_knn)
    fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, y_scores_logreg)
    
    # Store ROC curve data
    rf_roc_data.append({'fold': k+1, 'fpr': fpr_rf, 'tpr': tpr_rf, 'thresholds': thresholds_rf})
    xgb_roc_data.append({'fold': k+1, 'fpr': fpr_xgb, 'tpr': tpr_xgb, 'thresholds': thresholds_xgb})
    knn_roc_data.append({'fold': k+1, 'fpr': fpr_knn, 'tpr': tpr_knn, 'thresholds': thresholds_knn})
    logreg_roc_data.append({'fold': k+1, 'fpr': fpr_logreg, 'tpr': tpr_logreg, 'thresholds': thresholds_logreg})

    # Display ROC AUC plot
    # fig, ax = plt.subplots(figsize=(10, 8))
    # rf_roc_plot = RocCurveDisplay.from_estimator(rf_classifier, X_test, y_test, name='Random Forest', ax=ax)
    # xgb_roc_plot = RocCurveDisplay.from_estimator(xgb_classifier, X_test, y_test, name='XGBoost', ax=ax)
    # knn_roc_plot = RocCurveDisplay.from_estimator(knn_classifier, X_test, y_test, name='KNN', ax=ax)
    # logreg_roc_plot = RocCurveDisplay.from_estimator(logreg_classifier, X_test, y_test, name='Logistic Regression', ax=ax)
    # ax.set_title(f'ROC Curves for Multiple Classifiers - Fold {k+1}')
    # ax.grid(linestyle='--')
    # plt.legend(loc='lower right')
    # plt.tight_layout()
    # plt.savefig(os.path.join('roc_plots', f'roc_comparison_fold{k+1}.png'), dpi=300, bbox_inches='tight')
    # plt.close(fig)
    
    # Find the threshold that is closest to both specificity and recall being 70%
    optimal_idx_rf = np.argmin(np.abs(tpr_rf - 0.8))
    optimal_threshold_rf = thresholds_rf[optimal_idx_rf]
    
    optimal_idx_xgb = np.argmin(np.abs(tpr_xgb - 0.8))
    optimal_threshold_xgb = thresholds_xgb[optimal_idx_xgb]

    optimal_idx_knn = np.argmin(np.abs(tpr_knn - 0.8))
    optimal_threshold_knn = thresholds_knn[optimal_idx_knn]

    optimal_idx_logreg = np.argmin(np.abs(tpr_logreg - 0.8))
    optimal_threshold_logreg = thresholds_logreg[optimal_idx_logreg]
    # optimal_threshold_rf = 0.5
    # optimal_threshold_xgb = 0.5
    # optimal_threshold_knn = 0.5
    # optimal_threshold_logreg = 0.5

    # Re-evaluate model performance using the optimal threshold
    y_pred_rf = (y_scores_rf >= optimal_threshold_rf).astype(int)
    y_pred_xgb = (y_scores_xgb >= optimal_threshold_xgb).astype(int)
    y_pred_knn = (y_scores_knn >= optimal_threshold_knn).astype(int)
    y_pred_logreg = (y_scores_logreg >= optimal_threshold_logreg).astype(int)
    
    # Calculate evaluation metrics for RF for current fold
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_roc_rf = roc_auc_score(y_test, y_scores_rf)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    specificity_rf = tn / (tn + fp)
    precision_rf_curve, recall_rf_curve, _ = precision_recall_curve(y_test, y_scores_rf)
    
    auprc_rf = auc(recall_rf_curve, precision_rf_curve)

    # Store metrics in a dictionary for the current fold
    rf_metrics = {
        'Fold': k+1,
        'Accuracy': accuracy_rf,
        'Specificity': specificity_rf,
        'Precision': precision_rf,
        'Recall': recall_rf,
        'F1 Score': f1_rf,
        'AUC-ROC': auc_roc_rf,
        'AUPRC': auprc_rf,
    }

    # Append the dictionary to the list
    rf_fold_metrics.append(rf_metrics)

    # Extract the RandomForestClassifier from the pipeline
    rf_model = rf_classifier.named_steps["randomforestclassifier"]

    # Explain the model's predictions using SHAP for the current fold
    explainer = shap.TreeExplainer(rf_model, X_train, model_output="probability")
    shap_values = explainer(X_test)

    # Append SHAP values for the current fold and store the test indices
    shap_values_list.append(shap_values.values)
    test_indices_list.append(test_index)

    # Calculate evaluation metrics for XGB for current fold
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    precision_xgb = precision_score(y_test, y_pred_xgb)
    recall_xgb = recall_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    auc_roc_xgb = roc_auc_score(y_test, y_scores_xgb)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
    specificity_xgb = tn / (tn + fp)
    precision_xgb_curve, recall_xgb_curve, _ = precision_recall_curve(y_test, y_scores_xgb)
    auprc_xgb = auc(recall_xgb_curve, precision_xgb_curve)

    # Store metrics in a dictionary for the current fold
    xgb_metrics = {
        'Fold': k+1,
        'Accuracy': accuracy_xgb,
        'Specificity': specificity_xgb,
        'Precision': precision_xgb,
        'Recall': recall_xgb,
        'F1 Score': f1_xgb,
        'AUC-ROC': auc_roc_xgb,
        'AUPRC': auprc_xgb,
    }

    # Append the dictionary to the list
    xgb_fold_metrics.append(xgb_metrics)

    # Calculate evaluation metrics for KNN for current fold
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    precision_knn = precision_score(y_test, y_pred_knn)
    recall_knn = recall_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)
    auc_roc_knn = roc_auc_score(y_test, y_scores_knn)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
    specificity_knn = tn / (tn + fp)
    precision_knn_curve, recall_knn_curve, _ = precision_recall_curve(y_test, y_scores_knn)
    auprc_knn = auc(recall_knn_curve, precision_knn_curve)
    

    # Store metrics in a dictionary for the current fold
    knn_metrics = {
        'Fold': k+1,
        'Accuracy': accuracy_knn,
        'Specificity': specificity_knn,
        'Precision': precision_knn,
        'Recall': recall_knn,
        'F1 Score': f1_knn,
        'AUC-ROC': auc_roc_knn,
        'AUPRC': auprc_knn,
    }

    # Append the dictionary to the list
    knn_fold_metrics.append(knn_metrics)

    # Calculate evaluation metrics for LogReg for current fold
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    precision_logreg = precision_score(y_test, y_pred_logreg)
    recall_logreg = recall_score(y_test, y_pred_logreg)
    f1_logreg = f1_score(y_test, y_pred_logreg)
    auc_roc_logreg = roc_auc_score(y_test, y_scores_logreg)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_logreg).ravel()
    specificity_logreg = tn / (tn + fp)
    precision_logreg_curve, recall_logreg_curve, _ = precision_recall_curve(y_test, y_pred_logreg)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_logreg)
    auprc_logreg = auc(fpr_curve, tpr_curve)
    precision_logreg_curve, recall_logreg_curve, _ = precision_recall_curve(y_test, y_scores_logreg)
    auprc_logreg = auc(recall_logreg_curve, precision_logreg_curve)

    # Store metrics in a dictionary for the current fold
    logreg_metrics = {
        'Fold': k+1,
        'Accuracy': accuracy_logreg,
        'Specificity': specificity_logreg,
        'Precision': precision_logreg,
        'Recall': recall_logreg,
        'F1 Score': f1_logreg,
        'AUC-ROC': auc_roc_logreg,
        'AUPRC': auprc_logreg,
    }

    # Append the dictionary to the list
    logreg_fold_metrics.append(logreg_metrics)

    ################# Ensemble-Voting ######################
    # Define the predictions of individual classifiers (convert all 0s to -1s so weighted voted can be used)
    predictions = np.array([
        np.where(y_pred_rf == 0, -1, y_pred_rf),   
        np.where(y_pred_xgb == 0, -1, y_pred_xgb),
        np.where(y_pred_knn == 0, -1, y_pred_knn),
        np.where(y_pred_logreg == 0, -1, y_pred_logreg),
    ])

    # Define the weights for each classifier
    weights = np.array([2, 2, 1, 2])  # Adjust weights based on classifier performance

    # Perform cross-validated predictions using the voting classifier
    y_pred_voting = voting_function(predictions, weights)

    # For voting, create a dummy probability for ROC curve calculation
    # Use weighted average of probabilities from individual models
    y_scores_voting = (2*y_scores_rf + 2*y_scores_xgb + 1*y_scores_knn + 2*y_scores_logreg) / 7
    fpr_voting, tpr_voting, thresholds_voting = roc_curve(y_test, y_scores_voting)
    
    # Store ROC curve data for voting ensemble
    voting_roc_data.append({'fold': k+1, 'fpr': fpr_voting, 'tpr': tpr_voting, 'thresholds': thresholds_voting})

    # Evaluate the performance of the voting classifier
    accuracy_voting = accuracy_score(y_test, y_pred_voting)
    precision_voting = precision_score(y_test, y_pred_voting)
    recall_voting = recall_score(y_test, y_pred_voting)
    f1_voting = f1_score(y_test, y_pred_voting)
    auc_roc_voting = roc_auc_score(y_test, y_scores_voting)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_voting).ravel()
    specificity_voting = tn / (tn + fp)
    precision_voting_curve, recall_voting_curve, _ = precision_recall_curve(y_test, y_pred_voting)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_scores_voting)
    auprc_voting = auc(fpr_curve, tpr_curve)

    # Store metrics in a dictionary for the current fold
    voting_metrics = {
        'Fold': k+1,
        'Accuracy': accuracy_voting,
        'Specificity': specificity_voting,
        'Precision': precision_voting,
        'Recall': recall_voting,
        'F1 Score': f1_voting,
        'AUC-ROC': auc_roc_voting,
        'AUPRC': auprc_voting,
    }

    # Append the dictionary to the list
    voting_fold_metrics.append(voting_metrics)

print("\n------------------")
print("Best parameters summary for each model:")
print("------------------")

# Collect best parameters from each fold
rf_best_params = {}
xgb_best_params = {}
knn_best_params = {}
logreg_best_params = {}

# Count parameter occurrences across folds
for fold_metrics in rf_fold_metrics:
    fold = fold_metrics['Fold']
    fold_params = grid_search_rf.best_params_
    for param, value in fold_params.items():
        if param not in rf_best_params:
            rf_best_params[param] = {}
        if value not in rf_best_params[param]:
            rf_best_params[param][value] = 0
        rf_best_params[param][value] += 1

for fold_metrics in xgb_fold_metrics:
    fold = fold_metrics['Fold']
    fold_params = grid_search_xgb.best_params_
    for param, value in fold_params.items():
        if param not in xgb_best_params:
            xgb_best_params[param] = {}
        if value not in xgb_best_params[param]:
            xgb_best_params[param][value] = 0
        xgb_best_params[param][value] += 1

for fold_metrics in knn_fold_metrics:
    fold = fold_metrics['Fold']
    fold_params = grid_search_knn.best_params_
    for param, value in fold_params.items():
        if param not in knn_best_params:
            knn_best_params[param] = {}
        if value not in knn_best_params[param]:
            knn_best_params[param][value] = 0
        knn_best_params[param][value] += 1

for fold_metrics in logreg_fold_metrics:
    fold = fold_metrics['Fold']
    fold_params = grid_search_logreg.best_params_
    for param, value in fold_params.items():
        if param not in logreg_best_params:
            logreg_best_params[param] = {}
        if value not in logreg_best_params[param]:
            logreg_best_params[param][value] = 0
        logreg_best_params[param][value] += 1

# Output the most common parameters for each model
print("\nRandom Forest - Most Common Parameters:")
for param, values in rf_best_params.items():
    most_common_value = max(values.items(), key=lambda x: x[1])
    print(f"  {param}: {most_common_value[0]} (selected in {most_common_value[1]}/{len(rf_fold_metrics)} folds)")

print("\nXGBoost - Most Common Parameters:")
for param, values in xgb_best_params.items():
    most_common_value = max(values.items(), key=lambda x: x[1])
    print(f"  {param}: {most_common_value[0]} (selected in {most_common_value[1]}/{len(xgb_fold_metrics)} folds)")

print("\nKNN - Most Common Parameters:")
for param, values in knn_best_params.items():
    most_common_value = max(values.items(), key=lambda x: x[1])
    print(f"  {param}: {most_common_value[0]} (selected in {most_common_value[1]}/{len(knn_fold_metrics)} folds)")

print("\nLogistic Regression - Most Common Parameters:")
for param, values in logreg_best_params.items():
    most_common_value = max(values.items(), key=lambda x: x[1])
    print(f"  {param}: {most_common_value[0]} (selected in {most_common_value[1]}/{len(logreg_fold_metrics)} folds)")

# Save the best parameters to a CSV file
best_params_data = []

for param_name, param_values in rf_best_params.items():
    for value, count in param_values.items():
        best_params_data.append({
            'Model': 'Random Forest',
            'Parameter': param_name,
            'Value': value,
            'Count': count,
            'Total Folds': len(rf_fold_metrics)
        })

for param_name, param_values in xgb_best_params.items():
    for value, count in param_values.items():
        best_params_data.append({
            'Model': 'XGBoost',
            'Parameter': param_name,
            'Value': value,
            'Count': count,
            'Total Folds': len(xgb_fold_metrics)
        })

for param_name, param_values in knn_best_params.items():
    for value, count in param_values.items():
        best_params_data.append({
            'Model': 'KNN',
            'Parameter': param_name,
            'Value': value,
            'Count': count,
            'Total Folds': len(knn_fold_metrics)
        })

for param_name, param_values in logreg_best_params.items():
    for value, count in param_values.items():
        best_params_data.append({
            'Model': 'Logistic Regression',
            'Parameter': param_name,
            'Value': value,
            'Count': count,
            'Total Folds': len(logreg_fold_metrics)
        })

best_params_df = pd.DataFrame(best_params_data)
best_params_df.to_csv(os.path.join(directory, 'Best_Parameters_Summary.csv'), index=False)
print(f"\nBest parameters summary saved to {os.path.join(directory, 'Best_Parameters_Summary.csv')}")

# Save ROC curve data to CSV files for each model and fold
# Fixed ROC data saving code block:
for model_name, roc_data in [
    ('RandomForest', rf_roc_data),
    ('XGBoost', xgb_roc_data),
    ('KNN', knn_roc_data),
    ('LogisticRegression', logreg_roc_data),
    ('VotingEnsemble', voting_roc_data)
]:
    # Create a directory for this model if it doesn't exist
    model_dir = os.path.join('roc_data', model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save ROC data for each fold
    for fold_data in roc_data:
        fold_num = fold_data['fold']
        
        # Make sure all arrays are the same length
        min_length = min(len(fold_data['fpr']), len(fold_data['tpr']))
        
        # Create separate DataFrames for each component of ROC data
        fpr_df = pd.DataFrame({'false_positive_rate': fold_data['fpr']})
        tpr_df = pd.DataFrame({'true_positive_rate': fold_data['tpr']})
        thresholds_df = pd.DataFrame({'thresholds': fold_data['thresholds']})
        
        # Save to separate CSV files
        fpr_df.to_csv(os.path.join(model_dir, f'fold_{fold_num}_fpr.csv'), index=False)
        tpr_df.to_csv(os.path.join(model_dir, f'fold_{fold_num}_tpr.csv'), index=False)
        thresholds_df.to_csv(os.path.join(model_dir, f'fold_{fold_num}_thresholds.csv'), index=False)

# Create DataFrames from the lists of dictionaries
rf_results = pd.DataFrame(rf_fold_metrics)
xgb_results = pd.DataFrame(xgb_fold_metrics)
knn_results = pd.DataFrame(knn_fold_metrics)
logreg_results = pd.DataFrame(logreg_fold_metrics)
voting_results = pd.DataFrame(voting_fold_metrics)

# Calculate average results across folds
rf_avg = rf_results.mean(numeric_only=True).to_frame().T
xgb_avg = xgb_results.mean(numeric_only=True).to_frame().T
knn_avg = knn_results.mean(numeric_only=True).to_frame().T
logreg_avg = logreg_results.mean(numeric_only=True).to_frame().T
voting_avg = voting_results.mean(numeric_only=True).to_frame().T

# Add model labels
rf_results['Model'] = 'Random Forest'
xgb_results['Model'] = 'XGBoost'
knn_results['Model'] = 'KNN'
logreg_results['Model'] = 'Logistic Regression'
voting_results['Model'] = 'Voting Ensemble'

rf_avg['Model'] = 'Random Forest (Average)'
xgb_avg['Model'] = 'XGBoost (Average)'
knn_avg['Model'] = 'KNN (Average)'
logreg_avg['Model'] = 'Logistic Regression (Average)'
voting_avg['Model'] = 'Voting Ensemble (Average)'

# Concatenate all results
all_results = pd.concat([
    rf_results, rf_avg,
    xgb_results, xgb_avg,
    knn_results, knn_avg,
    logreg_results, logreg_avg,
    voting_results, voting_avg
])

# Save results to CSV
all_results.to_csv(os.path.join(directory, 'CV_Results_SingleRun.csv'), index=False)

# Print average results
print("\nRandom Forest Average Results:")
for col in rf_avg.columns:
    if col != 'Model':
        print(f"{col}: {rf_avg[col].values[0]:.4f}")

print("\nXGBoost Average Results:")
for col in xgb_avg.columns:
    if col != 'Model':
        print(f"{col}: {xgb_avg[col].values[0]:.4f}")

print("\nKNN Average Results:")
for col in knn_avg.columns:
    if col != 'Model':
        print(f"{col}: {knn_avg[col].values[0]:.4f}")

print("\nLogistic Regression Average Results:")
for col in logreg_avg.columns:
    if col != 'Model':
        print(f"{col}: {logreg_avg[col].values[0]:.4f}")

print("\nVoting Ensemble Average Results:")
for col in voting_avg.columns:
    if col != 'Model':
        print(f"{col}: {voting_avg[col].values[0]:.4f}")

# # Initialize list to store all SHAP values
# shap_values_ordered = np.empty((X.shape[0], shap_values_list[0].shape[1], 2))

# # Match each fold's SHAP values to their corresponding indices
# for indices, shap_vals in zip(test_indices_list, shap_values_list):
#     shap_values_ordered[indices, :, :] = shap_vals

# # Rename variables for the SHAP Plot
# X.rename(
#     columns={
#         'AgeatFV': 'Age',
#         'SEX': 'Sex',
#         'RACE_DESC_American Indian or Alaska Native': 'Race (American Indian or Alaska Native)',
#         'RACE_DESC_Asian': 'Race (Asian)',
#         'RACE_DESC_Black': 'Race (Black)',
#         'RACE_DESC_More than one race': 'Race (More than one race)',
#         'RACE_DESC_Native Hawaiian or Other Pacific Islander': 'Race (Native Hawaiian or Other Pacific Islander)',
#         'RACE_DESC_White': 'Race (Caucasian)',
#         'RACE_DESC_East Asian and South-East': 'Race (East Asian and South-East)',
#         'RACE_DESC_South Asian': 'Race (South Asian)',
#         'FAMILY_MS_Y': 'Family History of MS',
#         'SMOKING_EVER': 'Smoking History',
#         'DiseasedurationatFV': 'Disease Duration in Years',
#         'EDSS_FV': 'EDSS at FV',
#         'DISEASE_CATEGORY_DESC_FV_Secondary Progressive MS': 'MS Subtype (SPMS)',
#         'DISEASE_CATEGORY_DESC_FV_Relapsing-Remitting MS': 'MS Subtype (RRMS)',
#         'DISEASE_CATEGORY_DESC_FV_Clinically Isolated Syndrome': 'MS Subtype (CIS)',
#         'DISEASE_CATEGORY_DESC_FV_Suspected MS': 'MS Subtype (Suspected MS)',
#         'TotalnumberofrelapsesbeforeFV': 'Total Relapses Before FV',
#         'Numberofrelapsesinthe3yearsbeforeFV': 'Relapse in the 3 Years Before FV',
#         'Numberofrelapsesinthe1yearbeforeFV': 'Relapse in the Year Before FV',
#         'NewNumberofrelapsesinthe1yearbeforeFV': 'Relapse in the Year Before FV',
#         'NewT2lesionYearBeforeFV_Y': 'New T2 Lesion Before FV',
#         'NewGadLesionYearBeforeFV_Y': 'New Enhancing Lesion Before FV',
#         'TreatmentBeforeFV': 'Treatment with Platform Therapy'
#     },
#     inplace=True
# )

# # Variables that are not wanted in the SHAP Plot
# vars_to_remove = [
#     "timeSinceLastAttack",
#     "FAMILY_MS_N",
#     "PYRAMIDAL_FUNCTION",
#     "BOWEL_BLADDER_FUNCTION",
#     "TreatmentBeforeFV",
#     "SENSORY_FUNCTION",
#     "BRAINSTEM_FUNCTION",
#     "FAMILY_MS_U",
#     "VISUAL_FUNCTION",
#     "CEREBELLAR_FUNCTION",
#     "MENTAL_FUNCTION",
#     "RACE_DESC_Unknown or not reported",
#     "NewT2lesionYearBeforeFV_N",
#     "NewT2lesionYearBeforeFV_U",
#     "NewGadLesionYearBeforeFV_N",
#     "NewGadLesionYearBeforeFV_U"
# ]

# # Drop those variables from X
# X_dropped = X.drop(columns=vars_to_remove)

# # Identify dropped column indices in the original X
# indices_to_remove = [X.columns.get_loc(col) for col in vars_to_remove]

# # Sort indices in descending order to ensure correct columns are removed
# indices_to_remove.sort(reverse=True)

# # Remove columns from SHAP value list
# shap_values_dropped = shap_values_ordered.copy()
# for idx in indices_to_remove:
#     shap_values_dropped = np.delete(shap_values_dropped, idx, axis=1)

# # Create a SHAP summary plot for the minority class (1)
# plt.figure(figsize=(12, 10))
# shap.summary_plot(shap_values_dropped[:, :, 1], X_dropped, feature_names=X_dropped.columns, max_display=41, show=False)

# # Get the current figure and adjust subplot parameters
# fig = plt.gcf()
# fig.subplots_adjust(left=0.33, right=1, top=.99, bottom=0.06)  # Adjust as needed

# # Save the adjusted plot to a PNG file
# output_path = os.path.join(directory, "SHAP_Summary_Plot_SingleRun.png")
# plt.savefig(output_path, dpi=300, bbox_inches="tight")
# plt.close()

print(f"\nAnalysis complete. Results saved to {os.path.join(directory, 'CV_Results_SingleRun.csv')}")
print(f"ROC curve data saved to {os.path.join(directory, 'roc_data')}")
# print(f"SHAP plot saved to {output_path}")