# 🎬 Movie Review Sentiment Analysis

This project performs **Sentiment Analysis** on IMDB Movie Reviews using Python and Machine Learning.  
It classifies reviews as **Positive 😊** or **Negative 😞** based on the text content.

---

## 📂 Project Structure
IMDB_Sentiment/
│── IMDB Dataset.csv # Dataset (movie reviews + sentiment labels)
│── sentiment_analysis.py # Main Python script for training & testing
│── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Requirements

Before running, install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
🚀 How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/Prasanna4941/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis
Run the Python script:

bash
Copy code
python sentiment_analysis.py
The script will:

Load the IMDB dataset

Preprocess the text (cleaning, tokenizing, vectorizing)

Train a Machine Learning model (Logistic Regression / Naive Bayes)

Evaluate accuracy

Show visualizations

📊 Sample Output
✅ Accuracy Report:

yaml
Copy code
Model Accuracy: 88.5%
Classification Report:
              precision    recall  f1-score   support

    Negative       0.87      0.90      0.88      5000
    Positive       0.90      0.87      0.88      5000

    accuracy                           0.88     10000
✅ Confusion Matrix Heatmap:

✅ Word Cloud of Positive Reviews:

✅ Word Cloud of Negative Reviews:

📌 Example Prediction
Input:

arduino

"This movie was absolutely wonderful, I loved the acting!"
Output:

makefile
Prediction: Positive 😊
Input:

arduino
"The movie was boring and too long. Waste of time."
Output:

makefile
Prediction: Negative 😞
📷 Screenshots
Dataset preview

Training results

Accuracy/Confusion Matrix graphs

Word Clouds

(Add your screenshots inside a screenshots/ folder and link them here)

✨ Features
Preprocessing of raw text

TF-IDF Vectorization

Model training and evaluation

Visualization (accuracy, confusion matrix, word clouds)

Easy to extend with deep learning (LSTMs, BERT, etc.)

📌 Future Improvements
Deploy model using Flask or Streamlit

Add live user review input

Use Deep Learning models for better accuracy

👨‍💻 Author
Prasanna Kumar
📧 Email:prasannakumarnerella1@gmail.com
🔗 GitHub: Prasanna4941
