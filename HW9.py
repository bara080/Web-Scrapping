import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def scrape_article(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    article_links = soup.find_all('a', href=True)
    articles = []

    for link in article_links:
        article_url = link['href']
        if article_url.startswith('http'):
            article_resp = requests.get(article_url)
            article_soup = BeautifulSoup(article_resp.content, 'html.parser')
            article_text = article_soup.get_text()
            articles.append(article_text)

    return articles

url = "https://www.cnn.com"
url2 = "https://nypost.com"
url3 = "https://www.nbcnews.com"

articles1 = scrape_article(url)
articles2 = scrape_article(url2)
articles3 = scrape_article(url3)

def convert_to_tfidf(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(documents)
    return tfidf_vectorizer, tfidf_vectors

# Convert articles from each source to TF-IDF vectors
tfidf_vectorizer1, tfidf_vectors1 = convert_to_tfidf(articles1)
tfidf_vectorizer2, tfidf_vectors2 = convert_to_tfidf(articles2)
tfidf_vectorizer3, tfidf_vectors3 = convert_to_tfidf(articles3)

# Perform clustering for each source separately
cluster_labels = []
cluster_centers = []
for tfidf_vectors in [tfidf_vectors1, tfidf_vectors2, tfidf_vectors3]:
    kmeans = KMeans(n_clusters=10)
    labels = kmeans.fit_predict(tfidf_vectors)
    cluster_labels.append(labels)
    cluster_centers.append(kmeans.cluster_centers_)

# Get feature names (words) from TF-IDF vectorizers
feature_names1 = tfidf_vectorizer1.get_feature_names_out()
feature_names2 = tfidf_vectorizer2.get_feature_names_out()
feature_names3 = tfidf_vectorizer3.get_feature_names_out()

# Visualize top 100 words for each cluster
def visualize_top_words_per_cluster(cluster_centers, feature_names, num_words=100):
    for cluster_center in cluster_centers:
        # Sort cluster center by descending order of TF-IDF scores
        sorted_indices = np.argsort(cluster_center)[::-1]
        top_indices = sorted_indices[:num_words]
        top_words = [feature_names[i] for i in top_indices]
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_words))
        
        # Plot word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Top 100 Words in Cluster')
        plt.axis('off')
        plt.show()

# Visualize top 100 words for each cluster for each source
visualize_top_words_per_cluster(cluster_centers[0], feature_names1)
visualize_top_words_per_cluster(cluster_centers[1], feature_names2)
visualize_top_words_per_cluster(cluster_centers[2], feature_names3)
