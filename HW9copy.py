import requests
from bs4 import BeautifulSoup
import nltk
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from newspaper import Article, Config




url = "https://www.dailynews.com/2024/05/17/construction-work-begins-in-warner-center-to-build-rams-temporary-practice-facility/"

my_article = Article(url, language = "en")

# download the article

my_article.download()
print(my_article)

# parse the article

my_article.parse()

# Extract the title
print ("Title : ", my_article.title)


# Extract the authors

print("Authors: ", my_article.authors)

# Get the publishing date

print("Published date:", my_article.publish_date)

# NLP on the article

my_article.nlp()

# Extract summary
print("Summary: ", my_article.summary)

# Extract keywords
print("Keyword: ", my_article.keywords)