import re
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()

stop_words = set(stopwords.words("english"))
negation_words = {"not", "no", "nor", "never"}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean_review(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)


st.title("🎬 Movie Sentiment Analyzer")
st.write(
    "Enter a movie review and the model will predict whether its sentiment is positive or negative."
)

review = st.text_area(
    "Movie review",
    placeholder="Example: I loved the acting, story, and music in this movie.",
    height=160
)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a movie review first.")
    else:
        cleaned_review = clean_review(review)
        review_vector = vectorizer.transform([cleaned_review])

        prediction = model.predict(review_vector)[0]
        probabilities = model.predict_proba(review_vector)[0]

        confidence = probabilities.max()

        if prediction == "pos":
            st.success("Positive sentiment 😊")
        else:
            st.error("Negative sentiment 😞")

        st.metric("Model confidence", f"{confidence:.2%}")

        with st.expander("See processed review"):
            st.write(cleaned_review)