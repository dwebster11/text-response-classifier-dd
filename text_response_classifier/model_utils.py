# model_utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

def train_model(df_train: pd.DataFrame):
    # Convert text to string, fill NaN
    X = df_train["text_response"].fillna("").astype(str)
    y = df_train["label"].fillna("").astype(str)
    
    # Train classifier for 3 categories
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model

def predict_responses(model, df_test: pd.DataFrame):
    df_test["text_response"] = df_test["text_response"].fillna("").astype(str)
    df_test["predicted_label"] = model.predict(df_test["text_response"])
    
    # Ensure only three categories
    df_test["predicted_label"] = df_test["predicted_label"].apply(
        lambda x: x if x in ["opt_out", "suppress"] else "other"
    )
    return df_test
