import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, train_test_split, cross_val_predict

if __name__ == '__main__':
    print('Input one parameter: Drug name')

    if len(sys.argv) != 2:
        print('Incorrect inputs. Please rerun the code with the required input as suggested above.')
        sys.exit(1)

    print("Starting python code...", flush=True)

    drug_name = str(sys.argv[1])

    # Importing the dataset
    data = pd.read_csv("paad_logTPM_treatment_patient_protcode_jv.csv", index_col=0)

    # Load the response variable from the specific file
    response_data = pd.read_csv("survival.csv", index_col=0)
    y = response_data.loc[drug_name].values
    y = y.astype(float)

    # Subset - for selected features
    data_subset = data #No need to subset?

    # Remove columns with missing values in y
    non_missing_indices = ~pd.isnull(y)
    y = y[non_missing_indices]
    data_subset = data_subset.loc[:, non_missing_indices]
    
    gene_names = data_subset.index
    final_matrix = pd.DataFrame(data_subset.T)
    observation_names = final_matrix.index

    X = final_matrix.values
    
    random_state = 200

    # Splitting the dataset into training and testing sets with 30% for testing
    indices = np.arange(X.shape[0])

    # Initial split into training and test sets
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=random_state)

    print("Train-test split completed", flush=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reading the ranked features file and sorting to rank
    rankings = f"{drug_name}/{drug_name}_ranked_list_final_run.csv"
    biomarkers = pd.read_csv(rankings)
    num_rows = biomarkers.shape[0]
    features = biomarkers.iloc[:, 0].values
    ranks = biomarkers.iloc[:, 1].values
    sorted_indices = np.argsort(ranks)
    sorted_features = features[sorted_indices]
    
    print('Features in order: \n', sorted_indices, flush=True)

    results = []

    best_test_auc = 0
    best_num_features = 0
    best_model = None
    best_conf_matrix = None
    best_y_pred = y_test
    best_cv_conf_matrix = None

    for num_features in range(1, num_rows+1, 1): 
        selected_features = sorted_features[:num_features]
        selected_indices_gene = [np.where(gene_names == feature)[0][0] for feature in selected_features]
        X_train_selected = X_train_scaled[:, selected_indices_gene]
        X_test_selected = X_test_scaled[:, selected_indices_gene]

        model = SVC(kernel='linear', C=1, class_weight='balanced', random_state=42)

        kf = RepeatedStratifiedKFold(n_splits=20, n_repeats=1, random_state=42)
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)
        cv_mean_auc = cv_scores.mean()

        cv_scores_acc = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='accuracy', n_jobs=-1)
        cv_mean_acc = cv_scores_acc.mean()

        cv_predictions = cross_val_predict(model, X_train_selected, y_train, cv=kf, n_jobs=-1)
        cv_conf_matrix = confusion_matrix(y_train, cv_predictions)

        # To see which test model is best
        model.fit(X_train_selected, y_train)
        y_test_pred = model.predict(X_test_selected)
        test_auc = roc_auc_score(y_test, y_test_pred)
        test_accuracy = (y_test == y_test_pred).mean()

        results.append([num_features, cv_mean_auc, cv_mean_acc, test_auc, test_accuracy])

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_test_accuracy = test_accuracy
            best_num_features = num_features
            best_model = model
            best_y_pred = y_test_pred
            best_conf_matrix = confusion_matrix(y_test, y_test_pred)
            best_cv_conf_matrix = cv_conf_matrix

    print(f'Best Test AUC: {best_test_auc:.4f}')
    print(f'Best Test Accuracy: {best_test_accuracy:.4f}')
    print(f'Best Number of Features: {best_num_features}')
    print(f'Confusion Matrix (Test Set):\n{best_conf_matrix}')

    results_df = pd.DataFrame(results, columns=['Number of Features', 'CV AUC', 'CV Accuracy', 'Test AUC', 'Test Accuracy'])
    results_file = f"{drug_name}/{drug_name}_performance.csv"
    results_df.to_csv(results_file, index=False)

    # Plot test confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(best_conf_matrix, annot=True, cmap='YlGnBu', fmt='d', cbar=False, annot_kws={"size": 20})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{drug_name}: Confusion Matrix (Test Set) - Optimal Features: {best_num_features}')
    plt.xticks(ticks=[0.5, 1.5], labels=['Alive', 'Dead'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Alive', 'Dead'])
    confusion_matrix_file = f"{drug_name}/{drug_name}_confusion_matrix.png"
    plt.savefig(confusion_matrix_file)

    # Plot CV confusion matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(best_cv_conf_matrix, annot=True, cmap='YlGnBu', fmt='d', cbar=False, annot_kws={"size": 20})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Cross-Validation)')
    plt.xticks(ticks=[0.5, 1.5], labels=['Alive', 'Dead'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Alive', 'Dead'])
    cv_confusion_matrix_file = f"{drug_name}/{drug_name}_cv_confusion_matrix.png"
    plt.savefig(cv_confusion_matrix_file)

    print(f'Actual values: {y_test}')
    print(f'Predictions: {best_y_pred}')

    incorrect_indices = test_indices[y_test != best_y_pred]
    incorrect_observations = observation_names[incorrect_indices]
    print("Incorrectly predicted observations:", flush=True)
    for name in incorrect_observations:
        print(name)

    # Plot the line graph for performance metrics
    x = results_df.iloc[:, 0]
    y1 = results_df.iloc[:, 1]
    y2 = results_df.iloc[:, 2]
    y3 = results_df.iloc[:, 3]
    y4 = results_df.iloc[:, 4]

    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, linestyle='-', color='blue', label='CV AUC')
    plt.plot(x, y2, linestyle='--', color='blue', label='CV Accuracy')
    plt.plot(x, y3, linestyle='-', color='orange', label='Test AUC')
    plt.plot(x, y4, linestyle='--', color='orange', label='Test Accuracy')
    plt.xlabel('No. of genes')
    plt.ylabel('Metric value')
    plt.title(f'{drug_name}: TCGA PDAC: SVC on stratified 80-20 split:, n=129 on protein coding genes with std >= 1, Filtered (Top 150, >3 times/5), rs=200')
    plt.grid(True)
    plt.legend()

    max_y4 = max(y4)
    max_x4 = x[y4.idxmax()]
    plt.axvline(x=max_x4, color='r', linestyle='--', linewidth=1, label='Max Test AUC')

    performance_line_graph_file = f"{drug_name}/{drug_name}_performance_line_graph.png"
    plt.savefig(performance_line_graph_file)
    plt.show()
