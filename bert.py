# ==========================================
# Fake News Detection: TF-IDF + Logistic Regression + BERT (Windows Ready)
# ==========================================

# ------------------------------------------
# 1. Imports
# ------------------------------------------
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

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------
# 2. NLTK Download
# ------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------------------------------
# 3. Extract CSV from ZIP (Windows Path)
# ------------------------------------------
zip_path = r"C:\fake news\archive.zip"  # your local zip file path
extract_folder = r"C:\fake news"        # folder to extract

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Load CSVs
fake_df = pd.read_csv(extract_folder + r"\Fake.csv")
real_df = pd.read_csv(extract_folder + r"\True.csv")

fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------------------
# 4. TF-IDF + Logistic Regression
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

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

y_pred = lr_model.predict(X_test_tfidf)
print("-----TF-IDF + Logistic Regression Results-----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_lr(text):
    text = preprocess_text(text)
    vect = tfidf.transform([text])
    return lr_model.predict(vect)[0]

# ------------------------------------------
# 5. BERT / DistilBERT
# ------------------------------------------
# Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label_enc'].tolist(), test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
).to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    report_to="none"
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train BERT
trainer.train()

# Evaluate BERT
print("-----BERT / DistilBERT Results-----")
trainer.evaluate()

def predict_bert(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
    outputs = bert_model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return le.inverse_transform([pred])[0]

# ------------------------------------------
# 6. Test Predictions
# ------------------------------------------
sample_text = "Donald Trump launches new social media platform."
print("TF-IDF Prediction:", predict_lr(sample_text))
print("BERT Prediction:", predict_bert(sample_text))
