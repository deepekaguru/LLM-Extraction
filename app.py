import streamlit as st
import pandas as pd
from openai import OpenAI

# Load API key from secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Customer Feedback Extractor", layout="centered")
st.title("🧠 LLM-Powered Customer Feedback Extractor")

st.markdown("Extract structured insights (product, sentiment, issue, request) from unstructured reviews using GPT.")

# Upload CSV or enter a review manually
upload = st.file_uploader("📄 Upload a CSV file with a 'Text' column (optional)", type=['csv'])
manual_input = st.text_area("✍️ Or paste a single customer review below")

def extract_from_llm(review_text):
    prompt = f"""
Extract the following details from the customer review:
- Product
- Sentiment (positive, neutral, negative)
- Issue Reported (if any)
- Feature Request (if any)

Review: "{review_text}"
Respond in JSON format.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# Manual input processing
if manual_input:
    with st.spinner("Extracting insights from your review..."):
        structured = extract_from_llm(manual_input)
        st.subheader("🔍 Extracted Insights")
        st.json(structured)

# CSV upload processing
elif upload:
    df = pd.read_csv(upload)
    if 'Text' not in df.columns:
        st.error("❌ CSV must contain a 'Text' column.")
    else:
        results = []
        with st.spinner("Processing reviews..."):
            for review in df['Text'].dropna().head(10):  # limit to 10 reviews
                try:
                    structured = extract_from_llm(review)
                    results.append({"Review": review, "LLM Output": structured})
                except Exception as e:
                    results.append({"Review": review, "LLM Output": f"Error: {str(e)}"})

        result_df = pd.DataFrame(results)
        st.subheader("📄 Extracted Review Data")
        st.dataframe(result_df)
