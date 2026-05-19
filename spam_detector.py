"""
Spam Email Detector
===================
Classifies SMS messages as spam/ham using TF-IDF + scikit-learn.
Auto-downloads UCI SMS Spam Collection dataset.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import re, urllib.request, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import warnings; warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
if not os.path.exists("sms_spam.tsv"):
    urllib.request.urlretrieve(DATA_URL, "sms_spam.tsv")

df = pd.read_csv("sms_spam.tsv", sep="\t", header=None, names=["label","text"])
print(f"Shape: {df.shape}  Spam rate: {(df['label']=='spam').mean():.2%}")

df["text_length"] = df["text"].str.len()
fig, axes = plt.subplots(1,3,figsize=(14,4))
df["label"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue","salmon"], rot=0)
axes[0].set_title("Class Distribution")
df[df["label"]=="ham"]["text_length"].hist(ax=axes[1], bins=40, alpha=0.6, label="ham", color="steelblue")
df[df["label"]=="spam"]["text_length"].hist(ax=axes[1], bins=40, alpha=0.6, label="spam", color="salmon")
axes[1].set_title("Message Length"); axes[1].legend()
plt.tight_layout(); plt.savefig("eda_spam.png", dpi=150, bbox_inches="tight"); plt.close()

def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]"," ", text.lower())).strip()

df["clean_text"] = df["text"].apply(clean)
df["label_enc"]  = (df["label"] == "spam").astype(int)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df["clean_text"], df["label_enc"], test_size=0.2, random_state=42, stratify=df["label_enc"])
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
X_train = tfidf.fit_transform(X_train_raw)
X_test  = tfidf.transform(X_test_raw)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=300, C=1.0, random_state=42),
    "Linear SVM": LinearSVC(C=1.0, random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"acc":acc, "auc":auc}
    print(f"{name}: Acc={acc:.4f} AUC={auc:.4f}")

best = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest: {best}")
print(classification_report(y_test, models[best].predict(X_test), target_names=["ham","spam"]))
print("\n✅ Done!")
