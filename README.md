# PDAC Survival Prediction Pipeline

Contains a novel three-step SVM pipeline for predicting binary survival outcomes in PDAC patients using gene expression data.

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```


## Workflow

Run the scripts in the following order, passing the drug name as an argument each time (e.g., `Brigatinib`). All scripts are located in the `scripts/` folder. In our test data

### Step 1: Multi-seed RFE Ranking

```bash
python scripts/1_run_rfe_multiseed.py Brigatinib
```

This code ranks features (genes) using recursive feature elimination (RFE) across 10 random seeds and outputs a combined ranking file.

### Step 2: Consensus Filtering + Final RFE

```bash
python scripts/2_final_rfe_consensus.py DUMMY_DRUG
```

This code filters features that frequently appear in the top rankings and reruns RFE to produce a final ranked list.

### Step 3: Evaluate Feature Sets and Plot Results

```bash
python scripts/3_evaluate_ranking_curve.py DUMMY_DRUG
```

This iteratively evaluates models built with increasing numbers of top-ranked features and generates all final metrics and plots.

## Outputs

All outputs are saved inside a folder named after the drug (e.g., `DUMMY_DRUG/`). These include:

- `biomarkers_firstrun.csv`: RFE rankings from multiple seeds
- `[drug_name]_ranked_list_final_run.csv`: Final ranked gene list
- `[drug_name]_performance.csv`: Model performance across feature counts
- `[drug_name]_confusion_matrix.png`: Test confusion matrix
- `[drug_name]_cv_confusion_matrix.png`: Cross-validation confusion matrix
- `[drug_name]_performance_line_graph.png`: Performance vs. number of features plot
