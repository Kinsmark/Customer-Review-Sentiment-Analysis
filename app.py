"""
Customer Review Sentiment Analysis - Streamlit app.

Run train_and_save_model.py first to generate the model files this app loads.
Then launch with: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('sentiment_model.joblib')
    return vectorizer, model


vectorizer, model = load_artifacts()

st.title("Customer Review Sentiment Analyzer")
st.write(
    "Paste a product review below and the model will classify it as positive, "
    "negative, or neutral. Trained on real Amazon Alexa product reviews as a "
    "portfolio demo."
)

st.divider()

review_text = st.text_area(
    "Review text",
    placeholder="Type or paste a review here, e.g. 'Setup was a pain but sound quality is amazing once it's running.'",
    height=120
)

example_reviews = {
    "Choose an example...": "",
    "Clearly positive": "Absolutely love this thing. Easy to set up and the sound is fantastic for the price.",
    "Clearly negative": "Terrible experience. Stopped working after two weeks and customer service was no help at all.",
    "Genuinely mixed": "It's fine I guess. Does what it's supposed to but nothing special, wouldn't go out of my way to recommend it.",
}
example_choice = st.selectbox("Or try an example", list(example_reviews.keys()))
if example_choice != "Choose an example...":
    review_text = example_reviews[example_choice]
    st.text_area("Selected example:", value=review_text, height=80, disabled=True)

if st.button("Analyze sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Enter a review first.")
    else:
        X = vectorizer.transform([review_text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        class_order = model.classes_

        st.subheader("Result")

        emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
        st.markdown(f"### {emoji.get(prediction, '')} {prediction.capitalize()}")

        st.write("Confidence breakdown:")
        for cls, prob in sorted(zip(class_order, probabilities), key=lambda x: -x[1]):
            st.progress(float(prob), text=f"{cls.capitalize()}: {prob:.1%}")

        if prediction == "neutral":
            st.caption(
                "Neutral reviews are the hardest case for this model, since "
                "the language often overlaps with mildly positive reviews. "
                "Treat this prediction with a bit more caution than the others."
            )
