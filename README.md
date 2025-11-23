# Fake News Detection Using Machine Learning
## Overview
Fake news spreads quickly on social media and online platforms, misleading people and causing confusion. This project uses Natural language processing (NLP) and machine learning classification algorithms for detecting whether a news article is real or fake automatically. 
The system can help users identify trustworthy news efficiently.
## Features
 - It classify if the news article is Real or Fake automatically
 - Text preprocessing: lowercasing, stopword removal, stemming
 - TF-IDF vectorization of news text
 - Machine Learning models: Logistic Regression, Naive Bayes
 - Displays prediction results clearly (Real/Fake)
## Technologies / Tools Used
 - Python 3
 - Google Colab / Jupyter Notebook
 - Pandas, Numpy
 - scikit-learn
 - NLTK (Natural Language Toolkit)
 - Matplotlib / Seaborn (for visualization)
## Steps to Install & Run the Project
 1. Clone the repository:
    git clone
    https://github.com/aashvi-nagori/Fake-News-Detection.git
 2. Open the project in Google Colab or Jupyter Notebook
 3. Make sure the following libraries are installed:
    pip install pandas numpy
    scikit-learn nltk matplotlib seaborn
 4. Open the notebook 'FakeNewsDetection.ipynb'.
 5. Run each cell step by step to:
 - Load the dataset
 - Preprocess the text
 - Train the model
 - Test predictions
## Instructions for testing
 1. The project contains a test dataset ('test.csv') to check model performance.
 2. Users can also input custom news text in the notebook.
 3. After running the notebook, the model predicts whether the input news is Real or Fake.
 4. The notebook also displays accuracy, confusion matrix, and other evaluation matrices. 
     
