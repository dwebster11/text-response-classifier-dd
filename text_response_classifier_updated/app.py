import streamlit as st
import pandas as pd
from model_utils import predict_responses

st.set_page_config(page_title="Text Response Classifier", page_icon="📱")

st.title("📱 Text Response Classifier")
st.markdown("""
Classify text responses into:
- ✅ opt_out
- 🚫 suppress
- 📂 unclassified
---
""")

# Upload new responses
test_file = st.file_uploader("📄 Upload new CSV to classify", type=["csv"])

if test_file and st.button("🔍 Classify Responses"):
    df_test = pd.read_csv(test_file)

    if "text_response" not in df_test.columns:
        st.error("Your CSV must have a 'text_response' column.")
    else:
        df_result = predict_responses(df_test)
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
