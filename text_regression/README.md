# Text Regression - Assignment 1  

## Overview  
This project implements a text classification system that categorizes English texts into two classes: **geographic** or **non-geographic**. 
The solution leverages Wikipedia data and annotations, along with  **Logistic Regression**, to classify input texts accurately.  

## Features  
- **Text Classification**: Predicts whether a given text is geographic or non-geographic.  
- **Preprocessing Options**:  
  - **Stopword Removal**: Uses the SnowBall stopword list.  
  - **Stemming**: Implements Porter’s algorithm.  
  - **Lemmatization**: Uses WordNet Lemmatizer.  
- **Feature Extraction**:  
  - **TF-IDF**: With preprocessing.
- **Model Choices**:
  - **Logistic Regression**   

## Requirements  
### Python Version (NLTK/Spacy):  
- Python 3.6+  
- Libraries:  
  - `nltk` (for preprocessing & Naive Bayes)  
  - `spacy` (for advanced NLP pipelines)  
  - `scikit-learn` (for Logistic Regression)  
  - `wikipedia-api` (for fetching Wikipedia data)  
   ```python
   pip install -r requirements.txt
   ```

## Usage  

1. Training the Model
bash
python train.py \
    --data_source wikipedia \
    --model_type logesticregression \
    --preprocess lemma \
    --output_model model.pkl
Arguments:

Argument	Description	Options
--data_source	Training data source	wikipedia or path to CSV/JSON file
--model_type	Classification algorithm	naive_bayes or logistic_regression
--preprocess	Text preprocessing method	none, stopwords, stem, or lemma
--output_model	Path to save trained model	Any filename with .pkl extension

2. Classifying Text
bash
python predict.py \
    --model model.pkl \
    --text "The Amazon River flows through South America"

Output:
Input text: "obama is a president of the united states."
Classification: Geographic 

## Project Structure
text_regression/
├── data_files/                          # Raw and processed datasets
│   ├── wiki_dataset.csv           # Raw Wikipedia dataset
│   ├── processed_dataset.csv      # Cleaned/preprocessed data
│   └── tfidf_matrix.csv         # Feature matrices  
│   └── data_loader.py # Wikipedia data fetcher
│
├── src/                           # Source code
│   ├── preprocessing.py            # Data preprocessing scripts
│   ├── features.py          # Feature visualization
│   ├── model.py                   # Model training (Naive Bayes/Logistic Regression)
│   ├── predict.py                 # Prediction script
│   ├── utils.py                   # Helper functions
│   └── model.pkl                  # Trained model (serialized)
│   └── confusion_matrix.png       # Performance metrics
│  
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

## Implementation Pipeline  
1. **Data Collection**:  
   - Fetch annotated Wikipedia articles using the MediaWiki API.  
2. **Preprocessing**:  
   - Tokenization, stopword removal, stemming/lemmatization.  
3. **Feature Extraction**:  
   - TF-IDF  
4. **Model Training**:  
   -  Logistic Regression.  
5. **Classification**:  
   - Predict class (geographic/non-geographic) for input text.  


## Dataset  
- **Source**: Wikipedia articles with geographic/non-geographic annotations.