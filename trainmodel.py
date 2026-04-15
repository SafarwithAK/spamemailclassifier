# =========================================================
# 📌 SPAM EMAIL CLASSIFIER WITH DATA VISUALIZATION
# =========================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd
import string
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# =========================
# 2. DOWNLOAD NLTK DATA
# =========================
nltk.download('punkt')
nltk.download('stopwords')

# =========================
# 3. LOAD DATASET
# =========================
df = pd.read_csv('email.csv', encoding='latin-1')

# =========================
# 4. DATA CLEANING
# =========================

# Remove unnecessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Convert labels (ham=0, spam=1)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicate data

df = df.drop_duplicates(keep='first')

# =========================
# 5. DATA VISUALIZATION
# =========================

# 🔹 Spam vs Ham Distribution
plt.figure()
sns.countplot(x='target', data=df)
plt.title("Spam vs Ham Distribution")
plt.xlabel("0 = Ham, 1 = Spam")
plt.ylabel("Count")
plt.show()

# 🔹 Message Length Analysis
df['num_characters'] = df['text'].apply(len)

# Ham messages
plt.figure()
sns.histplot(df[df['target'] == 0]['num_characters'], bins=50)
plt.title("Ham Message Length")
plt.show()

# Spam messages

plt.figure()
sns.histplot(df[df['target'] == 1]['num_characters'], bins=50)
plt.title("Spam Message Length")
plt.show()

# 🔹 Average Length Comparison

plt.figure()
sns.barplot(x='target', y='num_characters', data=df)
plt.title("Average Message Length")
plt.show()

# =========================
# 6. TEXT PREPROCESSING
# =========================

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords & punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Apply preprocessing

df['transformed_text'] = df['text'].apply(transform_text)

# =========================
# 7. WORDCLOUD VISUALIZATION
# =========================
from wordcloud import WordCloud

# Spam WordCloud

spam_wc = WordCloud(width=500, height=500, background_color='white')
spam_words = spam_wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

plt.figure()
plt.imshow(spam_words)
plt.axis("off")
plt.title("Spam WordCloud")
plt.show()

# Ham WordCloud

ham_wc = WordCloud(width=500, height=500, background_color='white')
ham_words = ham_wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))

plt.figure()
plt.imshow(ham_words)
plt.axis("off")
plt.title("Ham WordCloud")
plt.show()

# =========================
# 8. FEATURE ENGINEERING (TF-IDF)
# =========================

tfidf = TfidfVectorizer(max_features=3000)

# Convert text to numerical vectors

X = tfidf.fit_transform(df['transformed_text']).toarray()
Y = df['target'].values

# =========================
# 9. TRAIN-TEST SPLIT
# =========================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# =========================
# 10. MODEL TRAINING
# =========================

model = MultinomialNB()
model.fit(X_train, Y_train)

# =========================
# 11. MODEL EVALUATION
# =========================

Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))

# =========================
# 12. CONFUSION MATRIX
# =========================

cm = confusion_matrix(Y_test, Y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# 13. SAVE MODEL & VECTORIZER
# =========================

pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model & Vectorizer saved successfully!")
