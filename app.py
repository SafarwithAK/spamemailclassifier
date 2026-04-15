import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import google.generativeai as genai

import time
import os
from dotenv import load_dotenv
load_dotenv()

# ======================
# 🔑 GEMINI API
# ======================

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 🔥 Multiple models (fallback system)
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-lite-latest"
]

# ======================
# 🔹 Load ML Model
# ======================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ======================
# 🔹 Page Config
# ======================
st.set_page_config(page_title="AI Spam Classifier", page_icon="📧")

# ======================
# 🎨 UI CSS
# ======================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141e30, #243b55);
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
textarea {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 10px !important;
    min-height: 150px !important;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
}
.result {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    margin-top: 20px;
}
.chatbox {
    background: rgba(255,255,255,0.1);
    padding: 12px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 🏷️ TITLE
# ======================
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
# 🤖 AI FUNCTION (Retry + Fallback)
# ======================
def get_ai_reply(user_text, is_spam):

    prompt = f"""
    You are a smart AI assistant.

    User message: {user_text}
    Spam result: {"Spam" if is_spam else "Not Spam"}

    Rules:
    - If spam → warn clearly
    - If not spam → reply friendly
    - Keep answer short (2-3 lines)
    """

    for model_name in MODELS:
        for attempt in range(3):  # retry 3 times
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text

            except Exception as e:
                if "503" in str(e):
                    time.sleep(2)
                else:
                    break  # try next model

    return "⚠️ All AI models are busy. Try again later."

# ======================
# ✉️ INPUT
# ======================
text = st.text_area("Enter your message")

# ======================
# 🚀 ANALYZE
# ======================
if st.button("Analyze Message"):
    if text.strip() != "":
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
            old = pd.read_csv("history.csv")
            updated = pd.concat([old, new_data], ignore_index=True)
        except:
            updated = new_data

        updated.to_csv("history.csv", index=False)

        # 🤖 AI reply
        st.session_state.ai_reply = get_ai_reply(text, result == 1)

    else:
        st.warning("⚠️ Enter a message")

# ======================
# 📊 RESULT + PIE CHART
# ======================
if st.session_state.result is not None:

    prob = st.session_state.prob
    result = st.session_state.result

    if result == 1:
        st.markdown(
            f'<div class="result" style="background:#ff4b4b;">🚫 Spam ({prob[1]*100:.2f}%)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result" style="background:#00c853;">✅ Not Spam ({prob[0]*100:.2f}%)</div>',
            unsafe_allow_html=True
        )

    # 📊 Pie Chart
    st.markdown("### 📊 Analysis")

    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(
        [prob[1]*100, prob[0]*100],
        labels=["Spam", "Not Spam"],
        autopct="%1.1f%%"
    )
    st.pyplot(fig)

    # 🤖 AI Chat
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
