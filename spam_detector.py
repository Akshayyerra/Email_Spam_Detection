import os
import zipfile
import requests
import io
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------- Config -------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
MODEL_PATH = "spam_model.joblib"
RANDOM_STATE = 42
# ----------------------------------

def download_and_load_sms_dataset(url=DATA_URL):
    print("Downloading dataset...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # The dataset file inside zip is "SMSSpamCollection"
    with z.open("SMSSpamCollection") as f:
        df = pd.read_csv(f, sep="\t", names=["label", "text"], quoting=3, header=None)
    print(f"Loaded {len(df)} rows.")
    return df

def basic_clean(text):
    # Lowercase
    text = text.lower()
    # Replace URLs and emails with tokens
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_data(df):
    df = df.copy()
    # Map labels: 'spam' -> 1, 'ham' -> 0
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    # Clean text
    df["clean_text"] = df["text"].astype(str).apply(basic_clean)
    return df

def build_and_train(df):
    X = df["clean_text"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ("clf", LogisticRegression(solver="saga", max_iter=2000, random_state=RANDOM_STATE)),
    ])

    # Optional: small grid search for C
    param_grid = {
        "clf__C": [0.1, 1.0, 3.0]
    }
    # Use GridSearchCV with cv=3 (fast). Remove grid search to train faster.
    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    print("Training model (GridSearchCV)...")
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)

    # Evaluate
    print("\nEvaluating on test set:")
    preds = gs.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save model
    joblib.dump(gs.best_estimator_, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    return gs.best_estimator_

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Train first.")
    return joblib.load(path)

def predict_messages(model, messages):
    cleaned = [basic_clean(m) for m in messages]
    preds = model.predict(cleaned)
    probs = model.predict_proba(cleaned)[:, 1] if hasattr(model, "predict_proba") else None
    results = []
    for msg, p, prob in zip(messages, preds, probs if probs is not None else [None]*len(preds)):
        results.append({
            "message": msg,
            "is_spam": bool(p),
            "spam_score": float(prob) if prob is not None else None
        })
    return results

def main():
    # 1) Download and load data
    df = download_and_load_sms_dataset()

    # 2) Prepare
    df = prepare_data(df)

    # 3) Train and evaluate
    model = build_and_train(df)

    # 4) Try some examples
    examples = [
        "Free entry in 2 a weekly competition to win FA Cup final tickets. Text WIN to 12345",
        "Hey, are we meeting for lunch today?",
        "URGENT! Your account has been compromised, click http://phishy.example to secure it",
        "Please find attached the minutes from yesterday's meeting."
    ]
    results = predict_messages(model, examples)
    print("\nExample predictions:")
    for r in results:
        print(f"- '{r['message'][:70]}...' -> spam={r['is_spam']} (score={r['spam_score']:.3f})")

if __name__ == "__main__":
    main()
