import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score
import joblib

# NLTK Downloads
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------------------------------------
# 1. Load Real Industry Dataset (UCI SMS/Email Spam Collection)
# -------------------------------------------------------------
print("Downloading and loading real dataset...")
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'email_text'])

print(f"Total Rows in Real Dataset: {df.shape[0]}")
print(df.head())

# -------------------------------------------------------------
# 2. Advanced Text Preprocessing (Clean & Lemmatize)
# -------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags, special chars, numbers
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Tokenize and Lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

print("\nPreprocessing real dataset texts...")
df['clean_text'] = df['email_text'].apply(preprocess_text)

# -------------------------------------------------------------
# 3. Vectorization & Data Splitting
# -------------------------------------------------------------
X = df['clean_text']
y = df['label'].map({'ham': 0, 'spam': 1}) # Convert labels to 0 and 1

# Extracting Features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------
# 4. Multi-Model Evaluation (Comparing Models on Real Data)
# -------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_precision = 0.0

print("\n--- Real Dataset Model Evaluation ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    print(f"{name} -> Accuracy: {acc * 100:.2f}%, Precision: {prec * 100:.2f}%")
    
    # Save best model based on Precision
    if prec >= best_precision:
        best_precision = prec
        best_model = model

# -------------------------------------------------------------
# 5. Model Saving (For Web App)
# -------------------------------------------------------------
joblib.dump(best_model, 'spam_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("\nReal Dataset Best Model & TF-IDF Vectorizer saved successfully!")