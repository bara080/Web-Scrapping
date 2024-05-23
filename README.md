# Web-Scrapping

# CNN Articles Clustering and Analysis

## Overview
This project demonstrates how to extract and analyze news articles from CNN using web scraping, natural language processing (NLP), and clustering techniques. The project includes code to visualize the clustering results using PCA (Principal Component Analysis) and export the processed data to an Excel file.

## Dataset
The dataset is fetched directly from a CNN news URL using web scraping. The content is then processed and analyzed.

## Files
- `main.py`: The main Python script that performs web scraping, NLP, clustering, and visualization.
- `cnn_data_tfidf_clusters.xlsx`: The output Excel file containing the TF-IDF vectors, meta keywords, summary, NLP keywords, and cluster labels.
- `images/`: Directory to store the generated plot images (if needed).

## Installation
To run this project, you need to have Python installed along with the necessary libraries. You can install the required libraries using pip:
```bash
pip install beautifulsoup4 requests newspaper3k pandas scikit-learn threadpoolctl matplotlib seaborn lxml


Visualization

The script generates a scatter plot showing the K-means clusters reduced to 2D using PCA. The plot helps visualize the distribution of articles across different clusters.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements

BeautifulSoup
Newspaper3k
scikit-learn
pandas
matplotlib
seaborn


This README file provides a clear overview of the project, instructions on how to install dependencies and run the code, and details about the functionality of the main script. Adjust the repository URL and any other specific details as necessary.

