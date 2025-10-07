
import pandas as pd
import numpy as np
import zipfile
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


zip_path = r"C:\fake news\archive.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(r"C:\fake news")


fake_df = pd.read_csv(r"C:\fake news\Fake.csv")
real_df = pd.read_csv(r"C:\fake news\True.csv")


fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'
df = pd.concat([fake_df, real_df], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------
# 4. Preprocessing Function
# ------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()                              # lowercase
    text = re.sub(r'[^a-z\s]', '', text)            # remove punctuation/numbers
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing on 'text' column
df['clean_text'] = df['text'].apply(preprocess_text)

# ------------------------------------------
# 5. Train-Test Split
# ------------------------------------------
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# 6. TF-IDF Vectorization
# ------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ------------------------------------------
# 7. Train Logistic Regression
# ------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ------------------------------------------
# 8. Evaluate Model
# ------------------------------------------
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------------------
# 9. Predict on new text (example)
# ------------------------------------------
def predict_news(text):
    text = preprocess_text(text)
    vect = tfidf.transform([text])
    pred = model.predict(vect)[0]
    return pred

sample_text = "Donald Trump launches new social media platform."
print("Prediction:", predict_news(sample_text))
