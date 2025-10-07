# ==========================================
# Fake News Detection: CPU-Compatible Tiny-BERT
# ==========================================

import pandas as pd
import numpy as np
import zipfile
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import gc

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

# ------------------------------------------
# 1. NLTK Download
# ------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------------------------------
# 2. Extract Dataset
# ------------------------------------------
zip_path = r"C:\fake news\archive.zip"
extract_folder = r"C:\fake news"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

fake_df = pd.read_csv(extract_folder + r"\Fake.csv")
real_df = pd.read_csv(extract_folder + r"\True.csv")

fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------
# 3. Preprocessing Function
# ------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# ------------------------------------------
# 4. TF-IDF + Logistic Regression
# ------------------------------------------
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Save TF-IDF model
joblib.dump(lr_model, extract_folder + r"\tfidf_lr_model.pkl")
joblib.dump(tfidf, extract_folder + r"\tfidf_vectorizer.pkl")

# Evaluate TF-IDF
y_pred = lr_model.predict(X_test_tfidf)
print("-----TF-IDF + Logistic Regression Results-----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ------------------------------------------
# 5. Tiny-BERT (CPU-friendly)
# ------------------------------------------
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# Reduce dataset for CPU training
sample_size = min(len(df), 5000)
df_sample = df.sample(n=sample_size, random_state=42)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sample['text'].tolist(),
    df_sample['label_enc'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tiny-BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-tiny')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

# Free memory
del train_encodings, test_encodings
gc.collect()

# Force CPU
device = torch.device("cpu")

bert_model = BertForSequenceClassification.from_pretrained(
    'prajjwal1/bert-tiny', num_labels=2
).to(device)

# Minimal TrainingArguments compatible with all versions
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train Tiny-BERT
trainer.train()

# Save Tiny-BERT
bert_model.save_pretrained(extract_folder + r"\bert_tiny_model")
tokenizer.save_pretrained(extract_folder + r"\bert_tiny_model")

# Evaluate Tiny-BERT
print("-----Tiny-BERT Results-----")
trainer.evaluate()

# ------------------------------------------
# 6. Unified Prediction Function
# ------------------------------------------
def predict(text, model_type="bert"):
    text = str(text)
    if model_type.lower() == "tfidf":
        vect = tfidf.transform([preprocess_text(text)])
        return lr_model.predict(vect)[0]
    elif model_type.lower() == "bert":
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=64).to(device)
        outputs = bert_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        return le.inverse_transform([pred])[0]
    else:
        raise ValueError("model_type must be 'bert' or 'tfidf'")

# ------------------------------------------
# 7. Test Predictions
# ------------------------------------------
sample_text = "Donald Trump launches new social media platform."
print("TF-IDF Prediction:", predict(sample_text, model_type="tfidf"))
print("Tiny-BERT Prediction:", predict(sample_text, model_type="bert"))
