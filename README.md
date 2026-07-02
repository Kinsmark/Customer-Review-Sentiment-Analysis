# Customer Review Sentiment Analysis

## What this project does

This project reads a product review and decides whether it's positive, negative, or neutral. It's the first NLP project in the portfolio, working with text instead of structured numbers like account balances or lab results.

The data is real: just over 3,000 verified Amazon reviews of Alexa devices (Echo, Echo Dot, Firestick, and similar products), each with a 1 to 5 star rating and the review text.

## Why it matters

A popular product can rack up thousands of reviews a month, more than any team can read by hand. Sentiment analysis automates the first pass: tag every review automatically, then let a person focus only on the negative ones instead of starting from a pile of thousands. It also catches shifts in mood early, like a spike in negative reviews right after a product update, instead of finding out a month later in a quarterly report.

## Approach

1. **Turning ratings into sentiment** - the dataset gives star ratings, not sentiment labels directly, so 1 and 2 stars were mapped to negative, 3 stars to neutral, and 4 and 5 stars to positive.
2. **TF-IDF** - converted review text into numbers using TF-IDF (Term Frequency, Inverse Document Frequency), which scores a word higher if it shows up a lot in one review but rarely across reviews overall. That's what lets a model tell "the" (common, low signal) apart from "defective" (rare, high signal).
3. **Logistic regression** - trained a simple, interpretable classifier on the TF-IDF features, with class weights to handle the fact that positive reviews vastly outnumber negative and neutral ones in this dataset.
4. **Evaluation** - looked at precision and recall per class and the confusion matrix specifically, since the most interesting mistakes a sentiment model makes are usually between neutral and positive, not negative and positive.

## Results

The model reaches 87% overall accuracy, but that number is mostly carried by the positive class, which makes up 87% of the dataset. Looking class by class is more honest: negative reviews are caught with 67% recall, neutral reviews are the hardest case at 52% recall, and positive reviews are handled very well at 91% recall. The confusion mostly happens between neutral and positive, which makes sense: a 3-star review and a 4-star review are often written in genuinely similar language. Negative reviews are rarely confused with positive ones, since frustrated language tends to look very different from happy language.

## Files in this project

- `alexa_sentiment_analysis.ipynb` - the full notebook, from raw review text to trained model, with NLP concepts (tokenization, stop words, TF-IDF) explained in plain language before the code
- `train_and_save_model.py` - retrains the model from scratch and saves the files the app needs
- `app.py` - the Streamlit app -- https://customer-review-sen.streamlit.app/
- `requirements.txt` - dependencies for deployment
- `README.md` - this file

## How to run it

**The notebook:**
1. Open `sentiment_analysis.ipynb` in Jupyter.
2. Run all cells top to bottom. It downloads the dataset directly from a public source, so no manual download is needed.

**The deployed app:**
1. Install dependencies: `pip install -r requirements.txt`
2. Train and save the model: `python train_and_save_model.py` (takes well under a minute)
3. Launch the app: `streamlit run app.py`
4. Paste in a review, or pick one of the built-in examples, and see the sentiment prediction with a confidence breakdown.

## What this demonstrates

NLP fundamentals from the ground up: tokenization, stop words, and the negation problem that motivates more advanced techniques later on. TF-IDF as a way to turn text into numbers a model can actually learn from. And reading model output carefully, including being upfront about where a model genuinely struggles (neutral reviews) rather than hiding behind a single accuracy number. 
