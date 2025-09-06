# 🎬 Movie Review Sentiment Analysis

## 📌 Overview
This project uses **Machine Learning** to classify IMDB movie reviews as **Positive (1)** or **Negative (0)**.  
It demonstrates the process of **Natural Language Processing (NLP)** — text cleaning, feature extraction with **TF-IDF**, and training models like **Naive Bayes** and **Logistic Regression**.

---

## ⚙️ Technologies Used
- Python 3
- Pandas
- Scikit-learn
- Regular Expressions (Regex) for text cleaning
- TF-IDF Vectorizer

---

## 📂 Dataset
- **IMDB Large Movie Review Dataset** (50,000 reviews).  
- Each review is labeled as **positive** or **negative**.  
- Download here: [Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

⚠️ Note: The dataset is **not uploaded in this repo** due to size. Please download manually.

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
Install dependencies

pip install -r requirements.txt


Run the script

python sentiment_analysis.py

📊 Results

Naive Bayes Accuracy: ~84%

Logistic Regression Accuracy: ~88%

🔮 Future Work

Use Deep Learning (LSTM, Transformers)

Add word embeddings (Word2Vec, GloVe, BERT)