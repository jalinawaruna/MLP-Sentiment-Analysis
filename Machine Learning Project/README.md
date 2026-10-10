# Amazon Reviews Sentiment Analysis – Final Evaluation (Implementation)
IT2011 Artificial Intelligence and Machine Learning · Group 2026-Y2-S1-MLB-B11G1-06

Binary sentiment classification (positive / negative) of Amazon product reviews
(Kaggle: *Amazon Reviews for Sentiment Analysis*, fastText format).

## Group members and models

| Notebook | Model | Member | IT Number |
|---|---|---|---|
| IT25102662_Linear_SVM.ipynb | Linear SVM | Sandeera R.A. | IT25102662 |
| IT25100100_Logistic_Regression.ipynb | Logistic Regression | Dissanayaka D.M.N.D.M. | IT25100100 |
| IT25102248_MLP.ipynb | MLP Neural Network | Waruna J.A.J. | IT25102248 |
| IT25101455_Naive_Bayes.ipynb | Naive Bayes | Ediriweera V.S. | IT25101455 |
| IT25100646_Random_Forest.ipynb | Random Forest | Uthsara S.J.I. | IT25100646 |
| IT25103215_Decision_Tree.ipynb | Decision Tree | Liyanage D.C.T. | IT25103215 |
| Group_Model_Comparison.ipynb | Comparison of all six models | Group | – |

## Common settings (same in all six notebooks)

- Random sample of 300,000 reviews from `train.ft.txt` (seed 42)
- Preprocessing: missing values and duplicates, text cleaning, IQR outlier removal on review length
  (299,983 reviews after preprocessing)
- 80/20 stratified train/test split: 239,986 training and 59,997 test reviews
- Hyperparameter tuning with 5-fold stratified cross-validation on a subset of the training data
- Each final model is evaluated once on the same test set

## Results (test set, 59,997 reviews)

| Rank | Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| 1 | Linear SVM | 0.9309 | 0.9310 | 0.9793 |
| 2 | Logistic Regression | 0.9242 | 0.9243 | 0.9764 |
| 3 | MLP Neural Network | 0.9170 | 0.9174 | 0.9740 |
| 4 | Naive Bayes | 0.8960 | 0.8957 | 0.9608 |
| 5 | Random Forest | 0.8896 | 0.8887 | 0.9571 |
| 6 | Decision Tree | 0.7791 | 0.7870 | 0.8653 |

## Folder structure

```
Final_Evaluation_Implementation/
├── README.md
├── requirements.txt
├── data/raw/                   put train.ft.txt here (not included, 1.5 GB)
├── notebooks/                  the six model notebooks + Group_Model_Comparison.ipynb
└── results/
    ├── final_metrics/          final test result of each model (one CSV per model)
    ├── outputs/                tuning results and summary tables (CSV)
    ├── eda_visualizations/     charts saved by the notebooks
    ├── figures/                group comparison chart
    ├── group_members.csv
    └── group_model_comparison.csv
```

## How to run

1. Install the libraries: `pip install -r requirements.txt`
2. Download `train.ft.txt` from Kaggle (bittlingmayer/amazonreviews) and put it in `data/raw/`.
3. Open a model notebook from the `notebooks` folder and run all cells
   (Kernel → Restart & Run All). Each notebook is independent: it loads and preprocesses
   the data itself with the same steps, so the notebooks can be run in any order.
4. After all six model notebooks have run, run `Group_Model_Comparison.ipynb`
   to compare the six models.

Approximate run times on a laptop: Logistic Regression, Linear SVM and Naive Bayes about
10 minutes each, Decision Tree about 20 minutes, Random Forest about 45 minutes and the MLP
about 50 minutes. All paths are relative, so the folder can be run on any PC.
