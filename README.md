# PhishGuard AI

**A machine learning project for phishing URL detection using Python and Random Forest.**

PhishGuard AI is a data science project that investigates how machine learning can be used to distinguish phishing URLs from legitimate URLs. The project uses exploratory data analysis (EDA), data visualization and a Random Forest classifier to identify patterns in URL and website characteristics.

## Project Objectives

The main objectives of this project are:

- Explore and understand a real-world phishing URL dataset.
- Identify patterns and differences between phishing and legitimate websites.
- Visualize important characteristics of the dataset.
- Train a machine learning model to classify URLs.
- Evaluate model performance using standard classification metrics.
- Investigate which features contribute most to the model's predictions.

## Dataset

This project uses the **PhiUSIIL Phishing URL Dataset**, obtained from the UCI Machine Learning Repository.

| Property | Value |
|---|---:|
| Total observations | 235,795 |
| Total columns | 55 |
| Input features | 54 |
| Legitimate URLs | 134,850 |
| Phishing URLs | 100,945 |
| Missing values | 0 |

The target variable is `label`:

- `0` = Phishing
- `1` = Legitimate

The dataset includes URL characteristics and website-related features, such as URL length, HTTPS usage, domain information and webpage structure.

**Dataset source:** [UCI Machine Learning Repository – PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)

## Technologies Used

- **Python** – Main programming language.
- **Pandas** – Data manipulation and analysis.
- **NumPy** – Numerical operations.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical visualizations.
- **Scikit-learn** – Machine learning and model evaluation.
- **ucimlrepo** – Retrieving the dataset from UCI.

## Exploratory Data Analysis

Several visualizations were created to investigate the dataset:

| Visualization | Description |
|---|---|
| Category Distribution | Compares the number of phishing and legitimate URLs. |
| URL Length Distribution | Examines differences in URL length between categories. |
| HTTPS Usage | Compares HTTPS usage between phishing and legitimate URLs. |
| Feature Importance | Shows which numerical features contribute most to the Random Forest model. |

All visualizations are available in the `plots/` directory.

## Machine Learning

A **Random Forest Classifier** was selected for the classification task.

The dataset was divided into:

- 80% training data
- 20% testing data

The model was trained using numerical features. The `URLSimilarityIndex` feature was excluded in an additional experiment to investigate its influence on model performance.

### Model Results

The experiment without `URLSimilarityIndex` produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 99.99% |
| Test observations | 47,159 |
| Correct predictions | 47,154 |
| Incorrect predictions | 5 |

### Confusion Matrix

| | Predicted phishing | Predicted legitimate |
|---|---:|---:|
| Actual phishing | 20,184 | 5 |
| Actual legitimate | 0 | 26,970 |

These results are preliminary. The unusually high performance requires further investigation, including checks for data leakage and validation on independent data. The results should therefore not be interpreted as proof of real-world phishing detection accuracy.

## Feature Importance

The feature importance analysis identified the following five most influential features:

| Feature | Importance |
|---|---:|
| LineOfCode | 15.58% |
| NoOfSelfRef | 13.77% |
| NoOfExternalRef | 13.44% |
| NoOfImage | 12.59% |
| NoOfJS | 6.19% |

This suggests that webpage structure and content-related characteristics contribute substantially to the model's decisions.

## Project Structure

```text
PhishGuard-AI/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── plots/
    ├── category_distribution.png
    ├── url_length.png
    ├── https_usage.png
    └── feature_importance.png
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Esmee0/PhishGuard-AI.git
cd PhishGuard-AI
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the project

```bash
python main.py
```

The dataset is retrieved from the UCI Machine Learning Repository when the program runs. The analysis results are printed in the terminal and visualizations are saved in the `plots/` directory.

## Limitations and Future Improvements

The current implementation is an experimental classification pipeline rather than a complete real-time phishing detection application.

Possible future improvements include:

- Investigating potential data leakage.
- Testing the model on independent and previously unseen data.
- Comparing Random Forest with other classification algorithms.
- Investigating feature selection and model optimization.
- Developing a user interface for interactive predictions.
- Exploring how to extract the required features from new URLs.

## Author

**Esmee Moller**

Individual Data Science / AI Project

**Project:** PhishGuard AI

**Repository:** [GitHub – PhishGuard AI](https://github.com/Esmee0/PhishGuard-AI)
