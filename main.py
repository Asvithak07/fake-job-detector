import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import joblib

# ---------- 0) Setup ----------
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    # keep letters/digits/space; drop other punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    # remove stopwords but KEEP domain cues like 'visa','fee','deposit','earn'
    kept = []
    preserve = {"visa", "deposit", "fee", "fees", "sponsorship", "earn", "daily", "per", "day", "whatsapp", "telegram"}
    for w in text.split():
        if (w not in STOP_WORDS) or (w in preserve):
            kept.append(w)
    return " ".join(kept)

# ---------- 1) Load ----------
data = pd.read_csv("data/fake_job_postings.csv")
for col in ["title", "description", "requirements"]:
    data[col] = data[col].fillna("")
data["text"] = (data["title"] + " " + data["description"] + " " + data["requirements"])
data["clean_text"] = data["text"].apply(clean_text)

X_text = data["clean_text"].values
y = data["fraudulent"].values

# ---------- 2) Vectorizers (word + char) ----------
tfidf_word = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
    max_features=20000
)
tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    min_df=3,
    sublinear_tf=True,
    max_features=10000
)

Xw = tfidf_word.fit_transform(X_text)
Xc = tfidf_char.fit_transform(X_text)
X = hstack([Xw, Xc])

# ---------- 3) Split (stratified) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------- 4) SVM + calibrated probabilities ----------
base_svc = LinearSVC(C=1.0, class_weight="balanced")
svm_cal = CalibratedClassifierCV(base_svc, method="sigmoid", cv=5)
svm_cal.fit(X_train, y_train)

# ---------- 5) Evaluate (default 0.5 threshold) ----------
y_proba = svm_cal.predict_proba(X_test)[:, 1]  # probability of class 1 (fake)
y_pred_default = (y_proba >= 0.50).astype(int)

print("Confusion Matrix @0.50:\n", confusion_matrix(y_test, y_pred_default))
print("\nReport @0.50:\n", classification_report(y_test, y_pred_default, digits=3))

# ---------- 6) Pick a better threshold (favor recall for class=1) ----------
prec, rec, th = precision_recall_curve(y_test, y_proba)

# prefer recall >= 0.85 if possible, else max F1 for class 1
target_recall = 0.85
idx = np.where(rec[:-1] >= target_recall)[0]  # thresholds has len-1
if len(idx) > 0:
    # among those, choose the one with highest precision
    best_i = idx[np.argmax(prec[idx])]
    best_thresh = float(th[best_i])
else:
    # maximize F1 for the positive class
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_i = int(np.nanargmax(f1[:-1]))
    best_thresh = float(th[best_i])

y_pred_best = (y_proba >= best_thresh).astype(int)
print(f"\nChosen threshold: {best_thresh:.3f}")
print("Confusion Matrix @best:\n", confusion_matrix(y_test, y_pred_best))
print("\nReport @best:\n", classification_report(y_test, y_pred_best, digits=3))

# ---------- 7) Simple rule-based booster keywords ----------
HARD_KEYWORDS = [
    "registration fee","processing fee","security deposit","application fee",
    "training fee","upfront","pay first","payable before joining","non-refundable"
]
SOFT_KEYWORDS = [
    "earn per day","earn daily","no experience","work from home",
    "quick money","limited slots","whatsapp","telegram","visa sponsorship guaranteed"
]

# ---------- 8) Save everything as one bundle ----------
bundle = {
    "model": svm_cal,
    "tfidf_word": tfidf_word,
    "tfidf_char": tfidf_char,
    "threshold": best_thresh,
    "hard_keywords": HARD_KEYWORDS,
    "soft_keywords": SOFT_KEYWORDS,
}
joblib.dump(bundle, "outputs/fake_job_detector_bundle.joblib")
print("\nSaved -> outputs/fake_job_detector_bundle.joblib")
