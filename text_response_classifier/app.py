import streamlit as st
import pandas as pd
import joblib
from model_utils import train_model, predict_responses

st.set_page_config(page_title="Text Response Classifier", page_icon="📱")

st.title("📱 Text Response Classifier")
st.markdown("""
Train a model to categorize text message replies into:
- ✅ **Opt-Out**
- 🚫 **Suppression**
- ☎️ **Wrong Number**
- 📂 **Other**
---
""")

# Upload training data
train_file = st.file_uploader("📄 Upload labeled training CSV", type=["csv"])

if train_file:
    df_train = pd.read_csv(train_file)
    st.write("### Preview of training data", df_train.head())

    if "text_response" not in df_train.columns or "label" not in df_train.columns:
        st.error("Your training CSV must have 'text_response' and 'label' columns.")
    else:
        if st.button("🚀 Train Model"):
            model = train_model(df_train)
            joblib.dump(model, "text_model.pkl")
            st.success("✅ Model trained and saved successfully!")

# Upload new responses
test_file = st.file_uploader("📄 Upload new CSV to classify", type=["csv"])

if test_file and st.button("🔍 Classify Responses"):
    try:
        model = joblib.load("text_model.pkl")
    except:
        st.error("No trained model found. Please train a model first.")
    else:
        df_test = pd.read_csv(test_file)
        if "text_response" not in df_test.columns:
            st.error("Your CSV must have a 'text_response' column.")
        else:
            df_result = predict_responses(model, df_test)
            st.write("### Classification Preview", df_result.head())

            # Offer categorized downloads
            st.markdown("### 📥 Download Categorized CSVs")
            for label in df_result["predicted_label"].unique():
                subset = df_result[df_result["predicted_label"] == label]
                st.download_button(
                    label=f"Download '{label}' responses",
                    data=subset.to_csv(index=False).encode("utf-8"),
                    file_name=f"{label}_responses.csv",
                    mime="text/csv"
                )
