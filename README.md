Email Spam Classification using NLP and Machine Learning
This project classifies emails as  spam or ham using advanced Natural Language Processing (NLP) and machine learning techniques. It preprocesses raw email text and trains a Multinomial Naive Bayes model for effective spam detection.
Features
Text Preprocessing:
  - Tokenization to split text into words.
  - Removal of stopwords to eliminate common, irrelevant words.
  - Cleaning special characters for cleaner input data.
  - Lemmatization to normalize words to their base forms.
Vectorization:
  - Converted text into numerical features using CountVectorizer.
Model Training:
  - Trained a Multinomial Naive Bayes model, ideal for text-based classification tasks.
Dataset
The dataset used for this project contains spam and ham email samples:  
- Source: [Kaggle - Spam Emails Dataset](https://www.kaggle.com/datasets/abdallahwagih/spam-emails)  
- Downloaded using Kaggle CLI:
  !kaggle datasets download -d abdallahwagih/spam-emails
  !unzip spam-emails.zip
Libraries Used
NLTK: For tokenization, stopword removal, and lemmatization.
Pandas: For data manipulation and analysis.
Scikit-learn: For CountVectorizer, model training, and evaluation.
Workflow
Dataset Loading:
Loaded the Kaggle dataset using pandas.read_csv() after extraction.
Text Preprocessing:
Tokenized the email text and removed stopwords.
Cleaned unwanted characters and lemmatized text.
Feature Extraction:
Transformed preprocessed text into numerical features using CountVectorizer.
Model Training:
Trained a Multinomial Naive Bayes classifier to identify spam and ham emails.
Evaluation:
Evaluated model performance using metrics like accuracy, precision, recall, and F1-score.

Install dependencies:
pip install nltk pandas scikit-learn
Download and extract the dataset:
kaggle datasets download -d abdallahwagih/spam-emails
unzip spam-emails.zip
Run the script:
python email_classification.py
Results:The Multinomial Naive Bayes model achieved high accuracy in distinguishing between spam and ham emails, demonstrating its effectiveness for text-based classification tasks.

