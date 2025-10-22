from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

def train_model(df_train: pd.DataFrame):
    X = df_train["text_response"]
    y = df_train["label"]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    model.fit(X, y)
    return model

def predict_responses(model, df_test: pd.DataFrame):
    df_test["predicted_label"] = model.predict(df_test["text_response"])
    return df_test
