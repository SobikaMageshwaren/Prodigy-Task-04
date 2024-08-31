Sentiment Analysis on IMDB Dataset 🎬📊

Overview 🌟
This project performs sentiment analysis on the IMDB Dataset using advanced Natural Language Processing (NLP) techniques and machine learning models. Our goal is to clean the text data, transform it into numerical features using TF-IDF, and train a logistic regression model to predict sentiments (positive/negative). The project also includes various visualizations to help you understand the model's performance and the underlying data.

Features 🔍
Data Preprocessing: Clean text data by removing punctuation, stopwords, and performing lemmatization.
TF-IDF Vectorization: Convert text data into numerical features.
Logistic Regression Model: Train a machine learning model to predict sentiment.
Evaluation Metrics: Generate classification reports and confusion matrices.
Visualizations: Create word clouds for positive and negative sentiments and a count plot for sentiment distribution.

Project Structure 📁
IMDB Dataset.csv: The dataset used for training and testing the model.
sentiment_analysis.py: Main Python script with code for data preprocessing, model training, and visualizations.
README.md: Detailed documentation for the project (this file).

Installation 🚀
To get started with this project, follow these steps:
Clone the Repository:

git clone https://github.com/yourusername/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb

Install Required Libraries: Ensure you have Python 3.x installed. Install the required packages by running:

pip install -r requirements.txt

Dependencies:
pandas
nltk
scikit-learn
matplotlib
seaborn
wordcloud
Download NLTK Data: The script will automatically download the necessary NLTK data (stopwords, wordnet, and punkt) when you run the script for the first time.

Usage 💻
Prepare Your Dataset: Ensure your dataset is in CSV format with two columns: review (text data) and sentiment (labels: positive/negative).

Run the Script: Execute the following command to perform sentiment analysis on your dataset:

python sentiment_analysis.py

Visualizations:
Sentiment Distribution: Displays a bar chart showing the distribution of positive and negative reviews.
Confusion Matrix: Generates a confusion matrix to evaluate the logistic regression model's performance.
Word Clouds: Shows word clouds for positive and negative reviews, highlighting the most frequent words in each category.

Sample Output 📈
Sentiment Distribution:
![Screenshot (202)](https://github.com/user-attachments/assets/25e36805-2d5a-40ba-b0ca-97368a5b70fd)


Confusion Matrix:
![Screenshot (203)](https://github.com/user-attachments/assets/c1c2cf15-1335-4f3b-9fee-6b64bcdc0469)


Word Cloud for Positive Sentiment:
![Screenshot (204)](https://github.com/user-attachments/assets/a822fa42-a807-47df-b349-d4a6ccff13e4)


Word Cloud for Negative Sentiment:
![Screenshot (205)](https://github.com/user-attachments/assets/c7543afc-cfc8-41ec-bfe7-4d5a30fcfe38)


Customization 🎨
You can customize the script to suit your needs:

Dataset: Replace IMDB Dataset.csv with your own dataset. Ensure it has the same structure (review and sentiment columns,Download the IBDM Dataset from kaggle)

Model: Experiment with other machine learning models such as SVM, Random Forest, etc.

Visualization: Tailor the visualizations (e.g., color palettes, word cloud properties) to your preferences.

Contributing 🤝
Contributions are welcome! If you have ideas, suggestions, or issues, please open a pull request or an issue.

License 📝
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements 🙌
The dataset used in this project is from the IMDB Dataset.
Special thanks to the developers of scikit-learn, nltk, matplotlib, and seaborn for their powerful libraries.
