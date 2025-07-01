import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

if __name__ == '__main__':
    print('Input one parameter\n 1. Drug name. (for now until we change)')

    if len(sys.argv) != 2:
        print('Incorrect outputs. Please rerun the code with the required inputs as suggested above.')
        sys.exit(1)

    print("Starting code...", flush=True)

    drug_name = str(sys.argv[1])

    # Importing the dataset - this is the logTPM file
    data = pd.read_csv("paad_logTPM_treatment_patient_protcode_jv.csv", index_col=0)
    
    # Load the response variable from the specific file
    response_data = pd.read_csv("survival.csv", index_col=0)
    y = response_data.loc[drug_name].values
    y = y.astype(float)

    # Subset 
    selected_features = pd.read_csv(f"{drug_name}/combined_list.csv")
    selected_feature_names = selected_features.iloc[:, 0].tolist()
    data_subset = data.loc[selected_feature_names]
    # Subset
    
    # Remove columns with missing values in y
    non_missing_indices = ~pd.isnull(y)
    y = y[non_missing_indices]
    data_subset = data_subset.loc[:, non_missing_indices]

    gene_names = data_subset.index  # Change if genes don't need to be extracted
    # print(gene_names)
    final_matrix = pd.DataFrame(data_subset.T)  # Change if genes don't need to be extracted
    observation_names = final_matrix.index

    X = final_matrix.values
    
    # Random states for multiple runs
    random_states = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    feature_rankings = pd.DataFrame(gene_names, columns=['Gene'])

    for rs in random_states:
        print(f"Running for random state {rs}", flush=True)

        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)

        # Standardizing data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SVC with fixed parameters
        svc = SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)

        # RFE
        rfe = RFE(estimator=svc, n_features_to_select=1)
        rfe.fit(X_train_scaled, y_train)
        ranks = rfe.ranking_

        feature_rankings[f'rs{rs}'] = ranks

    output_file = f"{drug_name}/biomarkers_firstrun.csv"
    feature_rankings.to_csv(output_file, index=False)
    print(f"Feature ranking saved to {output_file}.", flush=True)

    # Optional: Print CV and test AUC and accuracy
    cv_scores = []
    test_scores = []

    for rs in random_states:
        # Splitting data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=rs)

        # Standardizing data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SVC with fixed parameters
        svc = SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)

        # Cross-validation
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_auc_scores = []
        cv_acc_scores = []

        for train_index, test_index in kf.split(X_train_scaled, y_train):
            X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            svc.fit(X_train_fold, y_train_fold)
            y_pred_fold = svc.predict(X_test_fold)

            cv_auc_scores.append(roc_auc_score(y_test_fold, svc.decision_function(X_test_fold)))
            cv_acc_scores.append(accuracy_score(y_test_fold, y_pred_fold))

        cv_scores.append((np.mean(cv_auc_scores), np.mean(cv_acc_scores)))

        # Test set evaluation
        svc.fit(X_train_scaled, y_train)
        y_test_pred = svc.predict(X_test_scaled)
        test_auc = roc_auc_score(y_test, svc.decision_function(X_test_scaled))
        test_acc = accuracy_score(y_test, y_test_pred)
        test_scores.append((test_auc, test_acc))

    for i, rs in enumerate(random_states):
        print(f"Random State {rs} - CV AUC: {cv_scores[i][0]:.2f}, CV Accuracy: {cv_scores[i][1]:.2f}")
        print(f"Random State {rs} - Test AUC: {test_scores[i][0]:.2f}, Test Accuracy: {test_scores[i][1]:.2f}")

