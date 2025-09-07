import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv("IMDB Dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
def clean_text(text):
    text = text.lower()                               
    text = re.sub(r"<.*?>", "", text)                
    text = text.translate(str.maketrans("", "", string.punctuation))  
    text = re.sub(r"\d+", "", text)                  
    return text

df['review'] = df['review'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Shape of training data:", X_train_vec.shape)
print("Shape of testing data:", X_test_vec.shape)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

lr = LogisticRegression(max_iter=200)
lr.fit(X_train_vec, y_train)
y_pred_lr = lr.predict(X_test_vec)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

print("\nNaive Bayes Classification Report:\n")
print(classification_report(y_test, y_pred_nb))

print("\nLogistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_lr))

sample_reviews = [
    "The movie was absolutely fantastic, I loved every moment!",
    "Terrible film. Waste of time. Worst acting ever.",
    "It was okay, not the best but not the worst either.",
    "The storyline was dull but the visuals were stunning."
]

sample_reviews_clean = [clean_text(r) for r in sample_reviews]
sample_vec = vectorizer.transform(sample_reviews_clean)

predictions = lr.predict(sample_vec)
print("\nCustom Review Predictions (1 = Positive, 0 = Negative):")
for review, pred in zip(sample_reviews, predictions):
    print(f"Review: {review}\nPredicted Sentiment: {'Positive' if pred == 1 else 'Negative'}\n")
