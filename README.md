# Text Classification using a TF-IDF Matrix

NLP Project where I build a TF-IDF matrix to classify the sentiment of user reviews from Amazon, Yelp and IMDB.

## What is TF-IDF?

Term Frequency-Inverse Document Frequency (TF-IDF) determines how important a word is by weighing its frequency of occurence in the document and computing how often the same word occurs in other documents. If a word occurs many times in a particular document but not in others, then it might be highly relevant to that particular document and is therefore assigned more importance.

## Word Clouds

I built word clouds out of the most commonly appearing words from each of the 3 datasets, as well as the combined one.

#### 1) Entire Dataset -

![Dataset Word Cloud](https://github.com/shraddha-an/tfidf_text/blob/master/dataset_wc.png)

#### 2) Amazon Word Cloud -

![Amazon Word Cloud](https://github.com/shraddha-an/tfidf_text/blob/master/amazon_wc.png)

#### 3) IMDB Word Cloud - 

![IMDB Word Cloud](https://github.com/shraddha-an/tfidf_text/blob/master/imdb_wc.png)

#### 4) Yelp Word Cloud - 

![Yelp Word Cloud](https://github.com/shraddha-an/tfidf_text/blob/master/yelp_wc.png)

## Dataset

Download the dataset from here: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences

## Medium Article

I've written a detailed article of this project on Medium. If you're interested, you can check it out [here](https://medium.com/swlh/text-classification-using-tf-idf-7404e75565b8?source=friends_link&sk=c2be0898d36bc48c4a54c9062c471ca1).

## Acknowledgements

This dataset was created for the paper, ‘From Group to Individual Labels using Deep Features’, Kotzias et. al,. KDD 2015.
