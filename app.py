import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
import os

# ======================
# 🔹 PAGE CONFIG
# ======================
st.set_page_config(page_title="AI Spam Classifier", page_icon="📧")

# ======================
# 🔑 GEMINI API SETUP
# ======================
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except:
    api_key = None

# Models (stable ones)
MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# ======================
# 🔹 LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ======================
# 🎨 UI STYLE
# ======================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.chatbox {
    background: rgba(0,0,0,0.05);
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📧 Spam Email Classifier + 🤖 AI</div>', unsafe_allow_html=True)

# ======================
# 🎯 SESSION STATE
# ======================
if "result" not in st.session_state:
    st.session_state.result = None

if "prob" not in st.session_state:
    st.session_state.prob = None

if "show_history" not in st.session_state:
    st.session_state.show_history = False

if "ai_reply" not in st.session_state:
    st.session_state.ai_reply = ""

# ======================
# 🤖 AI FUNCTION
# ======================
def get_ai_reply(user_text, is_spam):

    if not api_key:
        return "⚠️ AI not configured."

    prompt = f"""
    You are a smart AI assistant.

    User message: {user_text}
    Spam result: {"Spam" if is_spam else "Not Spam"}

    Rules:
    - If spam → warn clearly
    - If not spam → reply friendly
    - Keep short
    """

    for model_name in MODELS:
        for attempt in range(3):
            try:
                model_ai = genai.GenerativeModel(model_name)
                response = model_ai.generate_content(prompt)
                return response.text
            except Exception as e:
                if "503" in str(e):
                    time.sleep(2)
                else:
                    break

    return "⚠️ AI busy. Try again later."

# ======================
# ✉️ INPUT
# ======================
text = st.text_area("Enter your message")

# ======================
# 🚀 ANALYZE
# ======================
if st.button("Analyze Message"):

    if text.strip() == "":
        st.warning("⚠️ Please enter a message")

    else:
        vector = vectorizer.transform([text])
        result = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        st.session_state.result = result
        st.session_state.prob = prob

        label = "Spam" if result == 1 else "Not Spam"

        # Save history
        new_data = pd.DataFrame({
            "Time": [datetime.now()],
            "Message": [text],
            "Result": [label],
            "Confidence": [max(prob)*100]
        })

        try:
            if os.path.exists("history.csv"):
                old = pd.read_csv("history.csv")
                updated = pd.concat([old, new_data], ignore_index=True)
            else:
                updated = new_data

            updated.to_csv("history.csv", index=False)
        except:
            pass

        # AI reply
        st.session_state.ai_reply = get_ai_reply(text, result == 1)

# ======================
# 📊 RESULT
# ======================
if st.session_state.result is not None:

    result = st.session_state.result
    prob = st.session_state.prob

    if result == 1:
        st.error(f"🚫 Spam ({prob[1]*100:.2f}%)")
    else:
        st.success(f"✅ Not Spam ({prob[0]*100:.2f}%)")

    # Pie chart
    st.markdown("### 📊 Analysis")

    fig, ax = plt.subplots()
    ax.pie(
        [prob[1]*100, prob[0]*100],
        labels=["Spam", "Not Spam"],
        autopct="%1.1f%%"
    )
    st.pyplot(fig)

    # AI reply
    st.markdown("### 🤖 AI Assistant")
    st.markdown(f'<div class="chatbox">{st.session_state.ai_reply}</div>', unsafe_allow_html=True)

# ======================
# 📜 HISTORY TOGGLE
# ======================
if st.button("📜 Show / Hide History"):
    st.session_state.show_history = not st.session_state.show_history

# ======================
# 📜 HISTORY
# ======================
if st.session_state.show_history:
    try:
        history = pd.read_csv("history.csv")
        st.markdown("### 📜 History")
        st.dataframe(history, use_container_width=True)
    except:
        st.info("No history yet")

