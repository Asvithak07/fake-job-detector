import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack

# ---------- Load artifacts ----------
BUNDLE = joblib.load("outputs/fake_job_detector_bundle.joblib")
model = BUNDLE["model"]
tfidf_word = BUNDLE["tfidf_word"]
tfidf_char = BUNDLE["tfidf_char"]
THRESH = float(BUNDLE["threshold"])
HARD = [s.lower() for s in BUNDLE["hard_keywords"]]
SOFT = [s.lower() for s in BUNDLE["soft_keywords"]]

# ---------- Preprocess (same as training) ----------
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    preserve = {"visa", "deposit", "fee", "fees", "sponsorship", "earn", "daily", "per", "day", "whatsapp", "telegram"}
    kept = []
    for w in text.split():
        if (w not in STOP_WORDS) or (w in preserve):
            kept.append(w)
    return " ".join(kept)

def find_keywords(raw_text: str):
    t = raw_text.lower()
    matched_hard = [k for k in HARD if k in t]
    matched_soft = [k for k in SOFT if k in t]
    return matched_hard, matched_soft

def predict_with_rules(raw_text: str):
    clean = clean_text(raw_text)
    Xw = tfidf_word.transform([clean])
    Xc = tfidf_char.transform([clean])
    X = hstack([Xw, Xc])
    proba_fake = float(model.predict_proba(X)[0, 1])

    # Rule-based booster
    matched_hard, matched_soft = find_keywords(raw_text)
    boosted = proba_fake
    reason = []

    if matched_hard:
        boosted = max(boosted, 0.90)   # strong override
        reason.append(f"Hard keywords: {', '.join(matched_hard)}")

    if matched_soft:
        # small boost per soft keyword
        boosted = min(boosted + 0.15 * len(matched_soft), 0.99)
        reason.append(f"Soft keywords: {', '.join(matched_soft)}")

    label = 1 if boosted >= THRESH else 0
    return label, proba_fake, boosted, reason

# ---------- UI ----------
st.title("🕵️‍♀️ Fake Job Posting Detector (SVM + Rules)")
st.write("Paste a job description to check if it's likely **Fake** or **Real**. The app also shows confidence and the reasons.")

txt = st.text_area("Job description / message", height=200, placeholder="Paste the job text here...")

if st.button("Analyze"):
    if not txt.strip():
        st.warning("Please paste some text.")
    else:
        label, p_raw, p_final, reasons = predict_with_rules(txt)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model probability (fake)", f"{p_raw*100:.1f}%")
        with col2:
            st.metric("Final probability (after rules)", f"{p_final*100:.1f}%")

        if label == 1:
            st.error("🚨 Prediction: **FAKE**")
        else:
            st.success("✅ Prediction: **REAL**")

        if reasons:
            st.info("Why: " + " | ".join(reasons))
        else:
            st.caption("No scam-specific keywords detected. Decision based on model.")
