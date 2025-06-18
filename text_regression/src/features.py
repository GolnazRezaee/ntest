import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')


df = pd.read_csv(r'C:\Users\Asus\PycharmProjects\Golnaz\text_regression\data_files\processed_dataset.csv')


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])


tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df.to_csv(r'C:\Users\Asus\PycharmProjects\Golnaz\text_regression\data_files\tfidf_matrix.csv', index=False)


print("TF-IDF matrix shape:", tfidf_matrix.shape)
print(tfidf_df.head())
