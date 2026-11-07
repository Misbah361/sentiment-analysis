import pandas as pd
import streamlit as st
from transformers import pipeline

# Start with the ready-made model. After you push your fine-tuned model to the
# Hugging Face Hub, replace this with "your-username/imdb-distilbert".
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(page_title="IMDB sentiment analysis", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading model (first time takes a minute)...")
def load_model():
    return pipeline("sentiment-analysis", model=MODEL_ID, truncation=True, max_length=512)


def to_label(raw):
    return "Positive" if raw.upper() in ("POSITIVE", "LABEL_1") else "Negative"


clf = load_model()

st.title("IMDB sentiment analysis")
st.caption("Classifies movie reviews as positive or negative using DistilBERT.")

tab1, tab2, tab3 = st.tabs(["Analyze a review", "Analyze a CSV", "Model results"])

with tab1:
    examples = {
        "Positive": "An absolutely wonderful film with brilliant acting.",
        "Negative": "A boring, predictable mess. I wanted my money back.",
        "Sarcasm": "Oh great, another two hours of my life gone. Just what I wanted!",
    }
    choice = st.selectbox("Load an example (optional)", ["Write my own"] + list(examples))
    default = examples.get(choice, "")
    review = st.text_area("Review text", value=default, height=140)

    if st.button("Analyze review", type="primary"):
        if not review.strip():
            st.warning("Type or paste a review first.")
        else:
            out = clf(review)[0]
            label = to_label(out["label"])
            conf = out["score"]
            if label == "Positive":
                st.success(f"{label} ({conf:.0%} confident)")
            else:
                st.error(f"{label} ({conf:.0%} confident)")
            st.progress(float(conf))
            st.caption("Confidence is how sure the model is about its answer, not a guarantee it is right.")

with tab2:
    st.write("Upload a CSV file with a column named `review` (the first column is used if there is none).")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        col = "review" if "review" in df.columns else df.columns[0]
        df = df.head(200).copy()
        texts = df[col].astype(str).tolist()
        with st.spinner(f"Analyzing {len(texts)} reviews..."):
            results = clf(texts, batch_size=16)
        df["sentiment"] = [to_label(r["label"]) for r in results]
        df["confidence"] = [round(r["score"], 3) for r in results]
        st.caption("Only the first 200 rows are analyzed.")
        st.bar_chart(df["sentiment"].value_counts())
        st.dataframe(df)
        st.download_button("Download results", df.to_csv(index=False), "sentiment_results.csv", "text/csv")

with tab3:
    st.subheader("Accuracy on 1,000 test reviews")
    scores = pd.DataFrame(
        {"Accuracy": [0.849, 0.881, 0.897]},
        index=["TF-IDF + Logistic Regression", "Pretrained DistilBERT", "Fine-tuned DistilBERT"],
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline", "84.9%")
    c2.metric("Pretrained", "88.1%", "+3.2")
    c3.metric("Fine-tuned", "89.7%", "+1.6")
    st.bar_chart(scores)
    st.subheader("Fine-tuning progress")
    st.table(pd.DataFrame({"Epoch": [1, 2], "Validation loss": [0.410, 0.325], "Accuracy": [0.835, 0.897]}))
    st.subheader("Limitations")
    st.markdown(
        "- Sarcasm uses positive words for a negative opinion.\n"
        "- Mixed and neutral reviews are often misclassified.\n"
        "- Only a subset of the data (5,000 train, 1,000 test) was used."
    )
