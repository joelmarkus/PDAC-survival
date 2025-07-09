import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, accuracy_score

if __name__ == '__main__':
    print('Input one parameter for our final run\n 1. Drug name. (for now until we change)')

    if len(sys.argv) != 2:
        print('Incorrect inputs. Please rerun the code with the required inputs as suggested above.')
        sys.exit(1)

    print("Starting code...", flush=True)

    drug_name = str(sys.argv[1])

    # Load the ranks file
    ranks_file = f"{drug_name}/biomarkers_firstrun.csv"
    ranks_df = pd.read_csv(ranks_file)

    # Find features that appear in the top 150 ranks at least 3 times
    top_150_features = ranks_df.loc[:, 'rs10':'rs100'].apply(lambda x: (x <= 150).sum(), axis=1)
    selected_features = ranks_df['Gene'][top_150_features >= 5].tolist()

    # Importing the dataset
    data = pd.read_csv("paad_logTPM_treatment_patient_protcode_jv.csv", index_col=0)
    
    # Load the response variable from the specific file
    response_data = pd.read_csv("survival.csv", index_col=0)
    y = response_data.loc[drug_name].values
    y = y.astype(float)

    # Subset - for selected features
    data_subset = data.loc[selected_features]

    # Remove columns with missing values in y
    non_missing_indices = ~pd.isnull(y)
    y = y[non_missing_indices]
    data_subset = data_subset.loc[:, non_missing_indices]
    
    gene_names = data_subset.index
    final_matrix = pd.DataFrame(data_subset.T)
    observation_names = final_matrix.index

    X = final_matrix.values

    # Splitting data into train and test sets
    random_state = 200
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    # Standardizing data
    scaler = StandardScaler() #change if you need any other scaler - RobustScaler() works well too with this data, from my experience
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVC with fixed parameters
    svc = SVC(kernel='linear', C=1, class_weight='balanced', random_state=42) #or C=0.5

    # RFE
    rfe = RFE(estimator=svc, n_features_to_select=1)
    rfe.fit(X_train_scaled, y_train)
    ranks = rfe.ranking_

    # Save the final ranked list
    final_rankings = pd.DataFrame({'Gene': gene_names, 'Rank': ranks})
    final_output_file = f"{drug_name}/{drug_name}_ranked_list_final_run.csv"
    final_rankings.to_csv(final_output_file, index=False)
    print(f"Final feature ranking saved to {final_output_file}.", flush=True)

    # Optional: Print CV and test AUC and accuracy
    cv_scores = []
    test_scores = []

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

    print(f"Random State {random_state} - CV AUC: {cv_scores[0][0]:.2f}, CV Accuracy: {cv_scores[0][1]:.2f}")
    print(f"Random State {random_state} - Test AUC: {test_scores[0][0]:.2f}, Test Accuracy: {test_scores[0][1]:.2f}")
