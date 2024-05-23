from bs4 import BeautifulSoup
from datetime import date
import requests
from newspaper import Article
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Get today's date
today = date.today()
d = today.strftime("%d/%m/%Y")

cnn_url = "https://www.bbc.com/news/newsbeat-44124396"

print(cnn_url)

# Fetch the HTML content from the URL
html = requests.get(cnn_url)

# Parse the HTML content using BeautifulSoup
cnn_bs4 = BeautifulSoup(html.content, "lxml")

# Collect data in lists
headlines = []
articles = []

# Extract and print headlines
for headline in cnn_bs4.find_all("h2"):
    headlines.append(headline.text)

# Extract and print news articles with the correct class name
for news in cnn_bs4.find_all("article", {"class": "sc-bwzfXH sc-gwVKww kDvhCz"}):
    articles.append(news.text.strip())

# Extract keywords from meta tags
meta_keywords = []
for meta_tag in cnn_bs4.find_all("meta"):
    if 'name' in meta_tag.attrs and meta_tag.attrs['name'].lower() == 'keywords':
        keywords_content = meta_tag.attrs.get('content', '')
        meta_keywords = [keyword.strip() for keyword in keywords_content.split(',')]
        break

print("Meta Keywords: ", meta_keywords)

# Use newspaper3k to perform NLP on the article
article = Article(cnn_url)
article.download()
article.parse()
article.nlp()

# Extract summary and keywords
summary_nlp = article.summary
keywords_nlp = article.keywords

print("Summary_nlp: ", summary_nlp)
print("Keyword_nlp: ", keywords_nlp)

# Combine all text data for TF-IDF Vectorization
all_documents = headlines + articles + [summary_nlp]

# Convert documents to TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(all_documents)

# Perform K-Means clustering with thread limits
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

with threadpool_limits(limits=1, user_api='blas'):
    kmeans.fit(tfidf_vectors)

labels = kmeans.labels_

# Convert TF-IDF vectors to array and create DataFrame
tfidf_array = tfidf_vectors.toarray()
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_array, columns=tfidf_feature_names)

# Add meta keywords, summary, NLP keywords, and cluster labels to the DataFrame
additional_data = {
    "Meta Keywords": [", ".join(meta_keywords)] * len(all_documents),
    "Summary NLP": [summary_nlp] * len(all_documents),
    "Keywords NLP": [", ".join(keywords_nlp)] * len(all_documents),
    "Cluster Label": labels
}

additional_df = pd.DataFrame(additional_data)

# Combine TF-IDF DataFrame with additional data
final_df = pd.concat([tfidf_df, additional_df], axis=1)

# Export DataFrame to Excel
output_file = "cnn_data_tfidf_clusters.xlsx"
final_df.to_excel(output_file, index=False)

print(f"Data successfully exported to {output_file}")

# Perform PCA to reduce the dimensions of the TF-IDF vectors to 2D
pca = PCA(n_components=2)
reduced_tfidf_vectors = pca.fit_transform(tfidf_vectors.toarray())

# Create a DataFrame for the reduced vectors and cluster labels
pca_df = pd.DataFrame(reduced_tfidf_vectors, columns=['PC1', 'PC2'])
pca_df['Cluster Label'] = labels

# Plot the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster Label',
    palette=sns.color_palette('hsv', num_clusters),
    data=pca_df,
    legend='full',
    alpha=0.7
)
plt.title('K-means Clustering of CNN Articles (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.show()
