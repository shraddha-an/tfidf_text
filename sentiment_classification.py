# Exploring different NLP models for sentiment classification
# 1) NLP: Preprocessing + TF IDF Matrix
# 2) Model Selection: Loop through multiple classifiers for selection
# 3) Metrics: Classification Report

# Importing the libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sb

# Importing dataset
yelp_ds = pd.read_csv('yelp_labelled.txt', sep ='\t', header = None, names = ['reviews', 'rating'])
amazon_ds = pd.read_csv('amazon_cells_labelled.txt', sep ='\t', header = None, names = ['reviews', 'rating'])
imdb_ds = pd.read_csv('imdb_labelled.txt', sep ='\t', header = None, names = ['reviews', 'rating'])

dataset = pd.concat([yelp_ds, amazon_ds, imdb_ds], ignore_index = True)

# Manually setting the rating for 2 reviews with NaN values
dataset.fillna(1, inplace = True)

# ============================ Part 1: NLP ================================
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def text_preprocess(ds: pd.Series) -> pd.Series:
    """
    Apply NLP Preprocessing Techniques to the reviews.
    """
    for m in range(len(ds)):
        main_words = re.sub('[^a-zA-Z]', ' ', ds[m])
        main_words = (main_words.lower()).split()
        main_words = [w for w in main_words if not w in set(stopwords.words('english'))]
        lem = WordNetLemmatizer()
        main_words = [lem.lemmatize(w) for w in main_words if len(w) > 1]
        main_words = ' '.join(main_words)
        ds[m] = main_words

    return ds

dataset['reviews'] = text_preprocess(dataset['reviews'])

# Word Cloud generator for the 3 datasets + complete dataset
# Corpus of all reviews to plot a BIG word cloud
corpus = " ".join(review for review in dataset.reviews)
yelp = ' '.join(review for review in yelp_ds.reviews)
amazon = ' '.join(review for review in amazon_ds.reviews)
imdb = ' '.join(review for review in imdb_ds.reviews)

from PIL import Image
from wordcloud import WordCloud

def generateWordCloud(corpus: str, cmap: str) -> wordcloud:
    """
    Return a Word Cloud object generated from the corpus and color map parameter.
    """
    wordcloud = WordCloud(background_color = 'black', width = 800, height = 400,
                      colormap = cmap, max_words = 180, contour_width = 3,
                      max_font_size = 80, contour_color = 'steelblue',
                      random_state = 0)

    wordcloud.generate(corpus)

    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.figure()

    return wordcloud

generateWordCloud(corpus = corpus, cmap = 'viridis').to_file('dataset_wc.png')
generateWordCloud(corpus = yelp, cmap = 'hsv').to_file('yelp_wc.png')
generateWordCloud(corpus = amazon, cmap = 'tab20b').to_file('amazon_wc.png')
generateWordCloud(corpus = imdb, cmap = 'Accent').to_file('imdb_wc.png')


# Spliiting into X & y
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Building a TF IDF matrix out of the corpus of reviews
from sklearn.feature_extraction.text import TfidfVectorizer
td = TfidfVectorizer(max_features = 4500)
X = td.fit_transform(X).toarray()

# Splitting into training & test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)

# =========================  Part 3: Model Selection  =======================
# Classifiers
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# ===========================  Part 4: Metrics  =============================
# Classification metrics
from sklearn.metrics import accuracy_score, classification_report
from pprint import pprint
classification_report = classification_report(y_test, y_pred)

print('\n Accuracy: ', accuracy_score(y_test, y_pred))
print('\nClassification Report')
print('======================================================')
print('\n', classification_report)


