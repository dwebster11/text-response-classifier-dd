from transformers import pipeline
import pandas as pd

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Candidate labels
CANDIDATE_LABELS = ["opt_out", "suppress"]

def predict_responses(df_test: pd.DataFrame):
    df_test["text_response"] = df_test["text_response"].fillna("").astype(str)
    predicted_labels = []
    for text in df_test["text_response"]:
        result = classifier(text, CANDIDATE_LABELS)
        top_label = result["labels"][0]
        score = result["scores"][0]
        if score < 0.5:
            predicted_labels.append("unclassified")
        else:
            predicted_labels.append(top_label)
    df_test["predicted_label"] = predicted_labels
    return df_test
