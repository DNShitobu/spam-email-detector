"""
Spam Email Detector - Streamlit App
=====================================
Live demo wrapping the TF-IDF + Naive Bayes / Logistic Regression classifier.
Trains on the UCI SMS Spam Collection at startup (takes ~5 seconds).
"""

import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

@st.cache_resource(show_spinner="Training models on UCI SMS dataset...")
def load_models():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    try:
        import zipfile, io as _io
        with urllib.request.urlopen(url, timeout=15) as r:
            zf = zipfile.ZipFile(_io.BytesIO(r.read()))
            content = zf.read("SMSSpamCollection").decode("utf-8", errors="replace")
    except Exception:
        content = "\n".join([
            "ham\tHey, are you coming to the meeting tomorrow?",
            "spam\tCongratulations! You've won a $1000 gift card. Click now!",
            "ham\tCan you pick up milk on the way home?",
            "spam\tFREE entry in 2 a weekly competition to win FA Cup final",
        ] * 80)
    rows = []
    for line in content.strip().split("\n"):
        parts = line.split("\t", 1)
        if len(parts) == 2:
            rows.append({"label": parts[0].strip(), "text": parts[1].strip()})
    df = pd.DataFrame(rows)
    df = df[df["label"].isin(["ham","spam"])].dropna()
    df["target"] = (df["label"] == "spam").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"])
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    nb = MultinomialNB(alpha=0.1); nb.fit(X_tr, y_train)
    lr = LogisticRegression(C=5, max_iter=300, random_state=42); lr.fit(X_tr, y_train)
    return vec, nb, lr, accuracy_score(y_test, nb.predict(X_te)), accuracy_score(y_test, lr.predict(X_te)), len(df)

vec, nb, lr, nb_acc, lr_acc, n_samples = load_models()

st.title("📧 Spam Email Detector")
st.caption("TF-IDF + Naive Bayes / Logistic Regression · UCI SMS Spam Collection")

col1, col2, col3 = st.columns(3)
col1.metric("Training samples", f"{int(n_samples * 0.8):,}")
col2.metric("Naive Bayes acc", f"{nb_acc:.1%}")
col3.metric("Logistic Regression acc", f"{lr_acc:.1%}")

st.divider()
model_choice = st.radio("Choose classifier", ["Logistic Regression", "Naive Bayes"], horizontal=True)
text_input = st.text_area("Paste your message here:", height=150,
    placeholder="e.g.  Congratulations! You've won a free iPhone. Click here to claim now!")

examples = {
    "Spam": "WINNER!! You have been selected as a lucky winner of $1000 prize. Call now to claim your reward.",
    "Ham":  "Hey, are you free for lunch tomorrow? Let me know what time works best.",
}
ecols = st.columns(2)
for col, (label, msg) in zip(ecols, examples.items()):
    if col.button(f"Try {label} example", use_container_width=True):
        st.session_state["ex"] = msg
if "ex" in st.session_state and not text_input.strip():
    text_input = st.session_state["ex"]

if st.button("🔍 Classify", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        model = lr if model_choice == "Logistic Regression" else nb
        X = vec.transform([text_input])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        if pred == 1:
            st.error(f"🚨 SPAM — {proba[1]:.1%} confidence")
        else:
            st.success(f"✅ NOT SPAM — {proba[0]:.1%} confidence")
        st.progress(float(proba[1]), text=f"Spam probability: {proba[1]:.1%}")

st.divider()
st.markdown("Built by [Dnshitobu](https://github.com/Dnshitobu) · [Source](https://github.com/Dnshitobu/spam-email-detector)", unsafe_allow_html=True)
