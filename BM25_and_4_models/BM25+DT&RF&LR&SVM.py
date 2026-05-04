import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from datasets import load_dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


#Load PubMedQA dataset

dataset = load_dataset("pubmed_qa", "pqa_labeled")

# combine multiple context sentences into one text
def build_context(example):
    example["combined_context"] = " ".join(example["context"]["contexts"])
    return example

dataset["train"] = dataset["train"].map(build_context)

#BM25 Transformer

class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, X, y=None):
        X = sparse.csr_matrix(X)
        n_samples, n_features = X.shape

        # document frequency
        df = np.bincount(X.indices, minlength=n_features)
        # BM25 idf
        self.idf_ = np.log((n_samples - df + 0.5) / (df + 0.5) + 1)

        # average document length
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        self.avgdl_ = doc_len.mean()

        return self

    def transform(self, X):
        X = sparse.csr_matrix(X).astype(np.float64)
        doc_len = np.asarray(X.sum(axis=1)).ravel()

        rows, cols = X.nonzero()
        data = X.data.copy()

        denom = data + self.k1 * (
            1 - self.b + self.b * doc_len[rows] / self.avgdl_
        )
        data = data * (self.k1 + 1) / denom
        data = data * self.idf_[cols]

        return sparse.csr_matrix((data, (rows, cols)), shape=X.shape)



#Prepare data
df = pd.DataFrame({
    "question": dataset["train"]["question"],
    "context": dataset["train"]["combined_context"],
    "label": dataset["train"]["final_decision"]
})

df["text"] = df["question"].fillna("") + " [SEP] " + df["context"].fillna("")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.3,
    random_state=42,
    stratify=df["label"]
)

label_order = ["yes", "no", "maybe"]



#Shared BM25 text pipeline

def make_pipeline(clf):
    return Pipeline([
        ("count", CountVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000
        )),
        ("bm25", BM25Transformer(k1=1.5, b=0.75)),
        ("clf", clf)
    ])


models = {
    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(
            max_depth=40,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        )
    ),
    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=60,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    ),
    "Logistic Regression": make_pipeline(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    ),
    "SVM": make_pipeline(
        LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    )
}



#Train + evaluate

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=label_order)

    results[name] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "cm": cm,
        "y_pred": y_pred
    }

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, labels=label_order))



#Confusion matrix

for name, info in results.items():
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=info["cm"],
        display_labels=label_order
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{name} - Confusion Matrix")
    plt.show()