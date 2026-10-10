# 🧠 MLP Sentiment Analysis Using Neural Networks

A Machine Learning project that uses a **Multi-Layer Perceptron (MLP)** neural network to classify Amazon product reviews as positive or negative. The project uses TF-IDF text features, hyperparameter tuning, and model evaluation to build an effective sentiment classification model.

## 📌 Project Overview

This project explores how a feed-forward neural network can learn patterns in customer reviews and predict their sentiment.

- **Model:** Multi-Layer Perceptron (MLP)
- **Task:** Binary sentiment classification
- **Dataset:** Amazon Reviews for Sentiment Analysis
- **Feature Extraction:** TF-IDF
- **Framework:** Scikit-learn
- **Student ID:** IT25102248

## 🎯 Objectives

- Preprocess and clean text review data.
- Convert text into numerical features using TF-IDF.
- Experiment with different feature configurations.
- Improve model performance using GridSearchCV.
- Reduce overfitting using early stopping and L2 regularization.
- Evaluate the final model using standard classification metrics.
- Predict sentiment for custom review text.

## 🗂️ Project Structure

```text
MLP-Sentiment-Analysis/
├── data/
│   └── raw/
│       └── README.txt
├── notebooks/
│   └── IT25102248_MLP.ipynb
├── results/
│   ├── eda_visualizations/
│   │   └── mlp_curves_cm_roc.png
│   ├── final_metrics/
│   │   └── MLP_Neural_Network.csv
│   └── outputs/
│       ├── mlp_gridsearch_results.csv
│       └── mlp_model_summary.csv
├── requirements.txt
└── README.md
```

*The large dataset file is not included in this repository. Download it separately before running the notebook.*

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## ⚙️ Methodology

### 1. Data Loading and Preprocessing

- Randomly sample 300,000 reviews using random seed 42.
- Extract sentiment labels from the dataset.
- Handle missing values and duplicate reviews.
- Clean review text and prepare it for feature extraction.
- Remove review-length outliers using the Interquartile Range (IQR) method.
- Split the data into training and testing sets using an 80:20 stratified split.

After preprocessing, 299,983 reviews remain.

### 2. TF-IDF Feature Extraction

Term Frequency–Inverse Document Frequency (TF-IDF) converts text into numerical features that represent the importance of words in reviews.

Different configurations were evaluated, including:

- 1,000 TF-IDF features with stop words removed.
- 1,000 features with stop words retained.
- 10,000 features with stop words retained.
- 10,000 features with unigram and bigram features.

The configuration containing 10,000 features with stop words retained and bigrams achieved the strongest baseline cross-validation F1 score.

### 3. MLP Neural Network

The MLP model learns relationships between numerical text features and sentiment labels through interconnected neural-network layers.

The baseline configuration uses:

- Hidden layers: `(16, 8)`
- Activation function: ReLU
- Optimizer: Adam
- Batch size: 32
- Maximum baseline epochs: 15

### 4. Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to evaluate nine combinations of hidden-layer sizes and L2 regularization strength (`alpha`).

The tuning process used:

- Hidden-layer configurations: `(16, 8)`, `(32, 16)`, `(64, 32)`
- Alpha values: `0.0001`, `0.001`, `0.01`
- Early stopping to reduce overfitting.
- F1 score to select the best configuration.

The best configuration was `(16, 8)` with `alpha = 0.0001`.

### 5. Model Evaluation

The final tuned model was trained on the full training set and evaluated on the held-out test set. The test set was not used for hyperparameter selection.

## 📊 Final Model Performance

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 91.70% |
| Precision | 91.35% |
| Recall | 92.13% |
| F1 Score | 91.74% |
| ROC-AUC | 97.40% |

The model achieved an F1 score of **91.74%** and a ROC-AUC of **97.40%** on the held-out test set.

### Visualizations

The project generates visualizations to help understand model behaviour, including:

- Training loss curves.
- Confusion matrix.
- ROC curve.
- Comparisons between baseline and tuned models.

These visualizations help evaluate classification performance and identify potential overfitting.

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/jalinawaruna/MLP-Sentiment-Analysis.git
cd MLP-Sentiment-Analysis
```

Replace `YOUR_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the Amazon Reviews dataset from Kaggle:

https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

Place the `train.ft.txt` file in:

```text
data/raw/train.ft.txt
```

The dataset is large and is not included in this GitHub repository.

### Step 5: Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/IT25102248_MLP.ipynb
```

Run the notebook cells in order, using **Kernel → Restart & Run All** if necessary.

Allow sufficient time for preprocessing, cross-validation, and model training.

## 🔮 Future Improvements

- Experiment with different neural-network architectures.
- Explore advanced text representations and embedding techniques.
- Reduce training time through more efficient feature extraction.
- Test the model on additional review datasets.
- Deploy the trained model through a web application or API.

## 👨‍💻 Author

**Waruna J.A.J.**  
Student ID: `IT25102248`

## 📚 Dataset Reference

Amazon Reviews for Sentiment Analysis — Kaggle dataset:

https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

## 📄 License

This project was developed for educational and academic purposes. Refer to the dataset provider's terms for dataset usage and redistribution conditions.
