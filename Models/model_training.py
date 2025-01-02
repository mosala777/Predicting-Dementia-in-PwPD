import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import wilcoxon, kruskal
import argparse

# Set fixed random seed
RANDOM_SEED = 6

def load_and_preprocess_data(file_path, label_column):
    data = pd.read_csv(file_path)
    y = data[label_column]
    x = data.drop(columns=[label_column], errors='ignore')
    return x, y

def get_model_definitions():
    rf_model = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 10]
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        scoring='roc_auc',
        return_train_score=True
    )

    xgb_model = GridSearchCV(
        estimator=XGBClassifier(objective='binary:logistic'),
        param_grid={
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": np.linspace(0.01, 0.3, 5),
            "max_depth": [3, 5, 6, 10]
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        scoring='roc_auc',
        return_train_score=True
    )

    lr_model = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid={
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ['elasticnet'],
            "solver": ['saga'],
            'l1_ratio': np.linspace(0.1, 1.0, 10)
        },
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        scoring='roc_auc',
        return_train_score=True
    )

    return {"Random Forest": rf_model, "Gradient Boosting": xgb_model, "Logistic Regression": lr_model}

def nested_cross_validation(models, x, y, repeats=20):
    scores = {model_name: [] for model_name in models.keys()}
    weights = compute_sample_weight(class_weight="balanced", y=y)
    
    for n in range(1, repeats + 1):
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED + n)
        for model_name, model in models.items():
            x_to_use = x
            if model_name == "Logistic Regression":
                scaler = RobustScaler().fit(x)
                x_to_use = scaler.transform(x)

            nested_scores = cross_validate(
                estimator=model,
                X=x_to_use,
                y=y,
                cv=outer_cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                fit_params={"sample_weight": weights}
            )
            scores[model_name].append(statistics.mean(nested_scores['test_score']))
    return scores

def plot_scores(scores, output_file):
    data = list(scores.values())
    labels = list(scores.keys())

    sns.boxplot(data=data, width=0.3, showmeans=True).set_title("ROC AUC Scores")
    sns.stripplot(data=data, jitter=True, color='black', size=5, alpha=0.6)
    plt.xticks(range(len(labels)), labels)
    plt.savefig(output_file)
    plt.close()

def save_results(scores, output_file):
    with open(output_file, "w") as file:
        for model_name, model_scores in scores.items():
            file.write(f"{model_name}: Mean AUC {statistics.mean(model_scores):.2f}, Std {np.std(model_scores):.5f}, Median {statistics.median(model_scores):.2f}\n")

def main():
    parser = argparse.ArgumentParser(description="Nested CV for classification models")
    parser.add_argument("file_path", type=str, help="Path to the CSV data file")
    parser.add_argument("label_column", type=str, help="Name of the label column")
    args = parser.parse_args()

    # Load data
    x, y = load_and_preprocess_data(args.file_path, args.label_column)

    # Define models
    models = get_model_definitions()

    # Perform nested CV
    scores = nested_cross_validation(models, x, y, repeats=20)

    # Plot and save results
    plot_scores(scores, "nested_cv_results.png")
    save_results(scores, "nested_cv_results.txt")

if __name__ == "__main__":
    main()
