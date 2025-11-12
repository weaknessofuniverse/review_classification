import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv('data/processed/train.csv')
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.1, stratify=df['label'], random_state=42
)

pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), max_features=50000),
    LogisticRegression(max_iter=200, class_weight='balanced', multi_class='ovr')
)

pipe.fit(X_train, y_train)
preds = pipe.predict(X_val)
print(classification_report(y_val, preds))
print(confusion_matrix(y_val, preds))

joblib.dump(pipe, 'models/baseline.joblib')
