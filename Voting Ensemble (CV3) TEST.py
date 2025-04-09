import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV, KFold
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import make_pipeline
import shap
import matplotlib.pyplot as plt

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


################# Ensemble Method ######################
# Define the voting function
def voting_function(predictions, weights):
    # Calculate the weighted sum of the predictions
    weighted_sum = np.dot(predictions.T, weights)

    # Apply majority vote
    return (weighted_sum > 0).astype(int)

# Define the parameter grid for the random forest
param_grid_rf = {
    'randomforestclassifier__n_estimators': [100, 200, 300],  # 增加树的数量
    'randomforestclassifier__max_depth': [5, 7, 9, 11],       # 增加深度，提高复杂度
    'randomforestclassifier__min_samples_leaf': [3, 5, 10],   # 适度减少叶子节点样本数
}

# Define the parameter grid for XGBoost
param_grid_xgb = {
    'xgbclassifier__n_estimators': [100, 200, 300],  # 增加树的数量
    'xgbclassifier__max_depth': [5, 7, 9],          # 增加深度，提高复杂度
    'xgbclassifier__learning_rate': [0.05, 0.1, 0.2],  # 适度调整学习率
    'xgbclassifier__subsample': [0.8, 1],          # 保持默认
    'xgbclassifier__colsample_bytree': [0.8, 1]   # 保持默认
}

# Define the parameter grid for KNN
param_grid_knn = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan']
}

# Create a pipeline including SMOTE and each classifier
pipeline_rf = make_pipeline(SMOTE(), RandomForestClassifier())
pipeline_xgb = make_pipeline(SMOTE(), XGBClassifier(eval_metric='logloss'))
pipeline_knn = make_pipeline(SMOTE(), KNeighborsClassifier())

# Create GridSearchCV objects
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_xgb = GridSearchCV(pipeline_xgb, param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, scoring='roc_auc', n_jobs=-1)

# Create empty Data Frames to store results of each iteration of Cross Validation
rf_cv_results = pd.DataFrame()
xgb_cv_results = pd.DataFrame()
knn_cv_results = pd.DataFrame()
voting_cv_results = pd.DataFrame()

# Initialize lists to store all SHAP values across all test runs
all_shap_values_list = []

# Run 10 times
for i in range(0, 5):
    # Create lists to store metrics of each fold
    rf_fold_metrics = []
    xgb_fold_metrics = []
    knn_fold_metrics = []
    voting_fold_metrics = []

    # Initialize lists to store all Shap values and their test indices
    shap_values_list = []
    test_indices_list = []

    # Perform 10-fold cross-validation
    for train_index, test_index in KFold(n_splits=10, shuffle=True).split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Use GridSearchCV with SMOTE to find the best parameters
        grid_search_rf.fit(X_train, y_train)
        rf_classifier = grid_search_rf.best_estimator_

        grid_search_xgb.fit(X_train, y_train)
        xgb_classifier = grid_search_xgb.best_estimator_

        grid_search_knn.fit(X_train, y_train)
        knn_classifier = grid_search_knn.best_estimator_
        
        # Predict probabilities on the test set
        y_scores_rf = rf_classifier.predict_proba(X_test)[:, 1]
        y_scores_xgb = xgb_classifier.predict_proba(X_test)[:, 1]
        y_scores_knn = knn_classifier.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_scores_rf)
        fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_scores_xgb)
        fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_scores_knn)
        
        # Find the threshold that is closest to both specificity and recall being 70%
        #optimal_idx_rf = np.argmin(np.abs(1 - fpr_rf - 0.7) + np.abs(tpr_rf - 0.7))
        #optimal_threshold_rf = thresholds_rf[optimal_idx_rf]
        
        #optimal_idx_xgb = np.argmin(np.abs(1 - fpr_xgb - 0.7) + np.abs(tpr_xgb - 0.7))
        #optimal_threshold_xgb = thresholds_xgb[optimal_idx_xgb]

        #optimal_idx_knn = np.argmin(np.abs(1 - fpr_knn - 0.7) + np.abs(tpr_knn - 0.7))
        #optimal_threshold_knn = thresholds_knn[optimal_idx_knn]
        
        # 找到 Recall + Specificity 最大的索引
        optimal_idx_rf = np.argmax(tpr_rf + (1 - fpr_rf))
        optimal_idx_xgb = np.argmax(tpr_xgb + (1 - fpr_xgb))
        optimal_idx_knn = np.argmax(tpr_knn + (1 - fpr_knn))

        # 获取最佳阈值
        optimal_threshold_rf = thresholds_rf[optimal_idx_rf]
        optimal_threshold_xgb = thresholds_xgb[optimal_idx_xgb]
        optimal_threshold_knn = thresholds_knn[optimal_idx_knn]


        # Re-evaluate model performance using the optimal threshold
        y_pred_rf = (y_scores_rf >= optimal_threshold_rf).astype(int)
        y_pred_xgb = (y_scores_xgb >= optimal_threshold_xgb).astype(int)
        y_pred_knn = (y_scores_knn >= optimal_threshold_knn).astype(int)
        
        # Calculate evaluation metrics for RF for current fold
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)
        auc_roc_rf = roc_auc_score(y_test, y_pred_rf)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
        specificity_rf = tn / (tn + fp)
        precision_rf_curve, recall_rf_curve, _ = precision_recall_curve(y_test, y_pred_rf)
        auprc_rf = auc(recall_rf_curve, precision_rf_curve)

        # Store metrics in a dictionary for the current fold
        rf_metrics = {
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
        auc_roc_xgb = roc_auc_score(y_test, y_pred_xgb)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
        specificity_xgb = tn / (tn + fp)
        precision_xgb_curve, recall_xgb_curve, _ = precision_recall_curve(y_test, y_pred_xgb)
        auprc_xgb = auc(recall_xgb_curve, precision_xgb_curve)

        # Store metrics in a dictionary for the current fold
        xgb_metrics = {
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
        auc_roc_knn = roc_auc_score(y_test, y_pred_knn)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_knn).ravel()
        specificity_knn = tn / (tn + fp)
        precision_knn_curve, recall_knn_curve, _ = precision_recall_curve(y_test, y_pred_knn)
        auprc_knn = auc(recall_knn_curve, precision_knn_curve)

        # Store metrics in a dictionary for the current fold
        knn_metrics = {
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


    ################# Ensemble-Voting ######################
        # Define the predictions of individual classifiers (convert all 0s to 1s so weighted voted can be used)
        predictions = np.array([
            np.where(y_pred_rf == 0, -1, y_pred_rf),   
            np.where(y_pred_xgb == 0, -1, y_pred_xgb),
            np.where(y_pred_knn == 0, -1, y_pred_knn)
        ])

        # Define the weights for each classifier
        weights = np.array([2, 2, 1])  # Adjust weights based on classifier performance

        # Perform cross-validated predictions using the voting classifier
        y_pred_voting = voting_function(predictions, weights)

        # Evaluate the performance of the voting classifier
        accuracy_voting = accuracy_score(y_test, y_pred_voting)
        precision_voting = precision_score(y_test, y_pred_voting)
        recall_voting = recall_score(y_test, y_pred_voting)
        f1_voting = f1_score(y_test, y_pred_voting)
        auc_roc_voting = roc_auc_score(y_test, y_pred_voting)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_voting).ravel()
        specificity_voting = tn / (tn + fp)
        precision_voting_curve, recall_voting_curve, _ = precision_recall_curve(y_test, y_pred_voting)
        auprc_voting = auc(recall_voting_curve, precision_voting_curve)

        # Store metrics in a dictionary for the current fold
        voting_metrics = {
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

    # Create DataFrames from the lists of dictionaries
    rf_fold_results = pd.DataFrame(rf_fold_metrics)
    xgb_fold_results = pd.DataFrame(xgb_fold_metrics)
    knn_fold_results = pd.DataFrame(knn_fold_metrics)
    voting_fold_results = pd.DataFrame(voting_fold_metrics)

    # Calculate metric means across 10 folds
    avg_rf_fold_results = rf_fold_results.mean(axis=0).to_frame().T
    avg_xgb_fold_results = xgb_fold_results.mean(axis=0).to_frame().T
    avg_knn_fold_results = knn_fold_results.mean(axis=0).to_frame().T
    avg_voting_fold_results = voting_fold_results.mean(axis=0).to_frame().T

    # Concatenate average results to CV results list
    rf_cv_results = pd.concat([rf_cv_results, avg_rf_fold_results], ignore_index=True)
    xgb_cv_results = pd.concat([xgb_cv_results, avg_xgb_fold_results], ignore_index=True)
    knn_cv_results = pd.concat([knn_cv_results, avg_knn_fold_results], ignore_index=True)
    voting_cv_results = pd.concat([voting_cv_results, avg_voting_fold_results], ignore_index=True)

    # Initialize list to store all SHAP values from current run (samples, features, classes)
    shap_values_ordered = np.empty((X.shape[0], shap_values_list[0].shape[1], 2))

    # Match each fold’s SHAP values to their corresponding indices
    for indices, shap_vals in zip(test_indices_list, shap_values_list):
        shap_values_ordered[indices, :, :] = shap_vals

    # Append values to list of all runs
    all_shap_values_list.append(shap_values_ordered)

    # Print average results of 10 folds
    print("\nRandom Forest Cross Validation Results:")
    print(f"Accuracy: {avg_rf_fold_results['Accuracy'].values[0]}")
    print(f"Specificity: {avg_rf_fold_results['Specificity'].values[0]}")
    print(f"Precision: {avg_rf_fold_results['Precision'].values[0]}")
    print(f"Recall: {avg_rf_fold_results['Recall'].values[0]}")
    print(f"F1 Score: {avg_rf_fold_results['F1 Score'].values[0]}")
    print(f"AUC-ROC: {avg_rf_fold_results['AUC-ROC'].values[0]}")
    print(f"AUPRC: {avg_rf_fold_results['AUPRC'].values[0]}")

    # Print average results of 10 folds
    print("\nXGBoost Cross Validation Results:")
    print(f"Accuracy: {avg_xgb_fold_results['Accuracy'].values[0]}")
    print(f"Specificity: {avg_xgb_fold_results['Specificity'].values[0]}")
    print(f"Precision: {avg_xgb_fold_results['Precision'].values[0]}")
    print(f"Recall: {avg_xgb_fold_results['Recall'].values[0]}")
    print(f"F1 Score: {avg_xgb_fold_results['F1 Score'].values[0]}")
    print(f"AUC-ROC: {avg_xgb_fold_results['AUC-ROC'].values[0]}")
    print(f"AUPRC: {avg_xgb_fold_results['AUPRC'].values[0]}")

    # Print average results of 10 folds
    print("\nKNN Cross Validation Results:")
    print(f"Accuracy: {avg_knn_fold_results['Accuracy'].values[0]}")
    print(f"Specificity: {avg_knn_fold_results['Specificity'].values[0]}")
    print(f"Precision: {avg_knn_fold_results['Precision'].values[0]}")
    print(f"Recall: {avg_knn_fold_results['Recall'].values[0]}")
    print(f"F1 Score: {avg_knn_fold_results['F1 Score'].values[0]}")
    print(f"AUC-ROC: {avg_knn_fold_results['AUC-ROC'].values[0]}")
    print(f"AUPRC: {avg_knn_fold_results['AUPRC'].values[0]}")

    # Print average results of 10 folds
    print("\nVoting Ensemble Cross Validation Results:")
    print(f"Accuracy: {avg_voting_fold_results['Accuracy'].values[0]}")
    print(f"Specificity: {avg_voting_fold_results['Specificity'].values[0]}")
    print(f"Precision: {avg_voting_fold_results['Precision'].values[0]}")
    print(f"Recall: {avg_voting_fold_results['Recall'].values[0]}")
    print(f"F1 Score: {avg_voting_fold_results['F1 Score'].values[0]}")
    print(f"AUC-ROC: {avg_voting_fold_results['AUC-ROC'].values[0]}")
    print(f"AUPRC: {avg_voting_fold_results['AUPRC'].values[0]}")
    print("") # line feed

# Calculate metric means of 10 iterations
avg_rf_results = rf_cv_results.mean(axis=0).to_frame().T
avg_xgb_results = xgb_cv_results.mean(axis=0).to_frame().T
avg_knn_results = knn_cv_results.mean(axis=0).to_frame().T
avg_voting_results = voting_cv_results.mean(axis=0).to_frame().T

# Calculate metric stds of 10 iterations
std_rf_results = rf_cv_results.std(axis=0).to_frame().T
std_xgb_results = xgb_cv_results.std(axis=0).to_frame().T
std_knn_results = knn_cv_results.std(axis=0).to_frame().T
std_voting_results = voting_cv_results.std(axis=0).to_frame().T

# Create titles for the data frames
avg_rf_results_title = pd.DataFrame(['Average_RF:'], columns=['Title'])
std_rf_results_title = pd.DataFrame(['Std_RF:'], columns=['Title'])
avg_xgb_results_title = pd.DataFrame(['Average_XGB:'], columns=['Title'])
std_xgb_results_title = pd.DataFrame(['Std_XGB:'], columns=['Title'])
avg_knn_results_title = pd.DataFrame(['Average_KNN:'], columns=['Title'])
std_knn_results_title = pd.DataFrame(['Std_KNN:'], columns=['Title'])
avg_voting_results_title = pd.DataFrame(['Average_Voting:'], columns=['Title'])
std_voting_results_title = pd.DataFrame(['Std_Voting:'], columns=['Title'])

# Concatenate titles with data frames
avg_rf_results_with_title = pd.concat([avg_rf_results_title, avg_rf_results], axis=1)
std_rf_results_with_title = pd.concat([std_rf_results_title, std_rf_results], axis=1)
avg_xgb_results_with_title = pd.concat([avg_xgb_results_title, avg_xgb_results], axis=1)
std_xgb_results_with_title = pd.concat([std_xgb_results_title, std_xgb_results], axis=1)
avg_knn_results_with_title = pd.concat([avg_knn_results_title, avg_knn_results], axis=1)
std_knn_results_with_title = pd.concat([std_knn_results_title, std_knn_results], axis=1)
avg_voting_results_with_title = pd.concat([avg_voting_results_title, avg_voting_results], axis=1)
std_voting_results_with_title = pd.concat([std_voting_results_title, std_voting_results], axis=1)

# Concatenate all data frames
final_results = pd.concat([avg_rf_results_with_title, std_rf_results_with_title,
                            avg_xgb_results_with_title, std_xgb_results_with_title,
                            avg_knn_results_with_title, std_knn_results_with_title,
                            avg_voting_results_with_title, std_voting_results_with_title,    
                            ])

# Save combined data frames to the same CSV
output_path = os.path.join(directory, 'Final_Results_CV3.csv')
final_results.to_csv(output_path, index=False)

# Calculate average SHAP values
shap_values_average = np.mean(all_shap_values_list, axis=0)

# Rename variables for the SHAP Plot
X.rename(
    columns={
        'AgeatFV': 'Age',
        'SEX': 'Sex',
        'RACE_DESC_American Indian or Alaska Native': 'Race (American Indian or Alaska Native)',
        'RACE_DESC_Asian': 'Race (Asian)',
        'RACE_DESC_Black': 'Race (Black)',
        'RACE_DESC_More than one race': 'Race (More than one race)',
        'RACE_DESC_Native Hawaiian or Other Pacific Islander': 'Race (Native Hawaiian or Other Pacific Islander)',
        'RACE_DESC_White': 'Race (Caucasian)',
        'RACE_DESC_East Asian and South-East': 'Race (East Asian and South-East)',
        'RACE_DESC_South Asian': 'Race (South Asian)',
        'FAMILY_MS_Y': 'Family History of MS',
        'SMOKING_EVER': 'Smoking History',
        'DiseasedurationatFV': 'Disease Duration in Years',
        'EDSS_FV': 'EDSS at FV',
        'DISEASE_CATEGORY_DESC_FV_Secondary Progressive MS': 'MS Subtype (SPMS)',
        'DISEASE_CATEGORY_DESC_FV_Relapsing-Remitting MS': 'MS Subtype (RRMS)',
        'DISEASE_CATEGORY_DESC_FV_Clinically Isolated Syndrome': 'MS Subtype (CIS)',
        'DISEASE_CATEGORY_DESC_FV_Suspected MS': 'MS Subtype (Suspected MS)',
        'TotalnumberofrelapsesbeforeFV': 'Total Relapses Before FV',
        'Numberofrelapsesinthe3yearsbeforeFV': 'Relapse in the 3 Years Before FV',
        'Numberofrelapsesinthe1yearbeforeFV': 'Relapse in the Year Before FV',
        'NewT2lesionYearBeforeFV_Y': 'New T2 Lesion Before FV',
        'NewGadLesionYearBeforeFV_Y': 'New Enhancing Lesion Before FV',
        'Treatment with Injectable Med': 'Treatment with Platform Therapy'
    },
    inplace=True
)

# Variables that are not wanted in the SHAP Plot
vars_to_remove = [
    "timeSinceLastAttack",
    "FAMILY_MS_N",
    "PYRAMIDAL_FUNCTION",
    "BOWEL_BLADDER_FUNCTION",
    "TreatmentBeforeFV",
    "SENSORY_FUNCTION",
    "BRAINSTEM_FUNCTION",
    "FAMILY_MS_U",
    "VISUAL_FUNCTION",
    "CEREBELLAR_FUNCTION",
    "MENTAL_FUNCTION",
    "RACE_DESC_Unknown or not reported",
    "NewT2lesionYearBeforeFV_N",
    "NewT2lesionYearBeforeFV_U",
    "NewGadLesionYearBeforeFV_N",
    "NewGadLesionYearBeforeFV_U"
]

# Drop those variables from X
X_dropped = X.drop(columns=vars_to_remove)

# Identify dropped column indices in the original X
indices_to_remove = [X.columns.get_loc(col) for col in vars_to_remove]

# Sort indices in descending order to ensure correct columns are removed
indices_to_remove.sort(reverse=True)

# Remove columns from SHAP value list
shap_values_average_dropped = shap_values_average.copy()
for idx in indices_to_remove:
    shap_values_average_dropped = np.delete(shap_values_average_dropped, idx, axis=1)

# Create a SHAP summary plot for the minority class (1)
shap.summary_plot(shap_values_average_dropped[:, :, 1], X_dropped, feature_names=X_dropped.columns, max_display = 41, show=False)

# Get the current figure and adjust subplot parameters
fig = plt.gcf()
fig.subplots_adjust(left=0.33, right=1, top=.99, bottom=0.06)  # Adjust as needed

# Save the adjusted plot to a PNG file
output_path = os.path.join(directory, "SHAP_Summary_Plot_CV3.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")