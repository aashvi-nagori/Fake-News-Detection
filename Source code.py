# Fake News Detection Using Machine Learning

# Install Libraries
!pip install pandas numpy scikit-learn

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_excel("news.csv.xlsx")
df

# Check the Data
df.head()

# Fix column spaces
df.columns = df.columns.str.strip().str.lower()
print(df.columns)

# Select input/output
X = df['text']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df['text'] = df['text'].fillna("")
df['title'] = df['title'].fillna("")
df['label'] = df['label'].fillna("")
df.isnull().sum()

# Convert to numbers (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train/Test Split and Train the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Report
print(classification_report(y_test, y_pred))
