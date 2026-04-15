# 📧 Spam Email Classifier + 🤖 AI Assistant

An intelligent **Spam Email Detection System** built using **Machine Learning, NLP, and Streamlit**, integrated with **AI (Gemini API)** to provide smart responses.

---

## 🚀 Project Overview

Spam emails are a major issue in modern communication, causing security threats and inbox clutter. This project presents an **automated Spam Email Classifier** that detects whether a message is spam or not with high accuracy.

The system uses:
- Machine Learning (Naive Bayes)
- Natural Language Processing (NLP)
- TF-IDF Feature Extraction
- Streamlit Web Interface
- Gemini AI Integration

---

## 🎯 Features

- ✅ Spam / Not Spam Classification  
- 📊 Probability Visualization (Pie Chart)  
- 🤖 AI Assistant Response (Gemini API)  
- 📜 Message History Tracking  
- 🎨 Beautiful UI (Custom CSS + Streamlit)  
- ⚡ Fast & Real-time Prediction (~97% Accuracy)  

---

## 🛠️ Tech Stack

### Frontend
- Streamlit  
- HTML (via Streamlit)  
- CSS  

### Backend
- Python  
- Scikit-learn  
- NLTK  

### Libraries Used
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Pickle  
- WordCloud  
- python-dotenv  

### AI Integration
- Google Gemini API  

---

## 🧠 Machine Learning Model

- Algorithm: **Multinomial Naive Bayes**
- Feature Extraction: **TF-IDF**
- Accuracy: **~97%**
- Precision: **~1.0**

---

## 📂 Project Structure

```bash
SpamEmailClassifier/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── spam.csv
├── history.csv
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add API Key

Create `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### 4️⃣ Run App
```bash
streamlit run app.py
```

---

## 📊 How It Works

1. User enters a message  
2. Text preprocessing (NLP)  
3. TF-IDF converts text into vector  
4. Model predicts spam or not spam  
5. Result shown with confidence  
6. AI assistant gives response  
7. Data stored in history  

---

## 🧪 Example

**Input:**
```
Congratulations! You won ₹10,00,000. Click now!
```

**Output:**
```
🚫 Spam (98%)
```

---

## 🔐 Security

- API key stored in `.env`  
- No sensitive data exposed  

---

## 🚧 Future Improvements

- Gmail/Outlook Integration  
- Deep Learning Models (LSTM, BERT)  
- Mobile App  
- Multi-language support  

---

## 👨‍💻 Author

**Ajit Kumar**  
B.Tech CSE  
Sandip University  

---

## ⭐ Support

If you like this project:
- Star ⭐ the repo  
- Fork 🍴 it  
- Share 🔥  

