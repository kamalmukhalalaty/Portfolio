# [About Me](https://www.linkedin.com/in/kamalmukhalalaty/)

- 👋 Hi, I’m Kamal Mukhalalaty 

- 👀 I’m interested in Data Science and Analytics

- 🌱 I’m currently completing my Master of Engineering at the University of Toronto specalizing in data science
  - Here is a short list of what I have learned thus far:
    - Programming Languages:
      - Python
      - C
      - C#
      - VBA
    - Machine Learning & Data Analytics tools and techniques:
      - Scikit-learn
      - Pandas
      - Numpy
      - NLP (RegEx NLTK)
      - Time Series Forcasting & Analysis  
    - Deep Learning, Nural Networks (CNN, RNN, LSTM, ResNet, GAN):
      - Keras
      - TensorFlow
    - Big Data
      - Hadoop MapReduce
      - Hadoop via Python Streaming
      - Apache Spark
        - DataBricks
        - Pyspark
        - RDD & Dataframe manipulation
        - MLlib
    - Cloud Computing
      - Azure
        - Azure ML
        - Azure SQL
      - Data Engineering
        - Batch: Azure Data Factory, Azure Synapse Analytics
        - Stream: Azure IoT Hubs, Event Hubs, Stream Analytics

- 📫 How to reach me:
  - Email: [kamalmukhalalaty@gmail.com]()
  - Linkedin: [https://www.linkedin.com/in/kamalmukhalalaty/]()

# Projects:

## [Salary Prediction Challenge Kaggle](https://github.com/kamalmukhalalaty/Kaggle-Salary-Predictions)

The challenge objective: tell a data story about a subset of the data science community represented in the given survey, through a combination of both narrative text and data exploration. A “story” could be defined any number of ways, and that’s deliberate. The challenge is to deeply explore (through data) the impact, priorities, or concerns of a specific group of data science and machine learning practitioners. 

I decided to do both the data exploration and build a predictive model using logistic regression to predict the salaries of the survey participants.

This project involved:
- Data cleaning, wrangling and manipulation
- Imputation of missing values 
- Statistical Analysis 
- Data Exploration & Visualization
- Hyperparameter Tuning
- Bias-Variance Tradeoff Analyisis

For more details on the challenge itself and the to source the input data please visit: https://www.kaggle.com/c/kaggle-survey-2019

## [Sentiment Analysis on Generic Tweets & US Election Specific Posts (NLP)]()


For this project I thoroughly explore two labeled datasets, one contains sentiment labeled generic tweets while the other contains 2020 US Election relevant I built a NLP tweet sentiment analysis classifier.

The binary classifier is trained and validated on generic tweets from the file named sentiment_analysis.csv.

The extracted corpus is modeled via both Bag of words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF) embeddings.
top 200 features were selected based on frequency and the following models were applied using scikit's default settings:
LogisticRegression()
RandomForestClassifier()
DecisionTree
SVM
KNN
XGBoost W/Logistic objective -TF-IDF Embedding W/ Logistic Regression & Random Forest showed some promise so I conducted hyperparameter tuning which yeilded that Logistic Regresion in default setting has the highest validation accuracy of aproximatley 86%.
The previously defined best classifer is then applied on the entirety of the of US_Election_2020.csv dataset containing labeled 2020 US election relevant tweets and its performance is not as good at 58%. This is primarily due to computational contraints and dimensionality reduction requierments, the top 200 features from the generic tweets were used to train the model, these features are not as informative when in comes to dictating sentiment in the US election tweets as they are insufficiently diverse and unable to effectivly explain the relativly specific feature to sentiment mappings in the election relevant tweets.

A Multi-Class Classification model is created using the same steps as above to try and learn feature to negative sentiment reason mappings.

Finnaly an MLP-3 is Built using Keras and TF in an attempt to build an even more compatant classifier however the va;idation accuracy is only 1% higher so the idea is scrapped.

This was my first portfolio worthy project within the realm of NLP, model performance could be improved in the following ways:

getting access to massively parallel processing (MPP) to speed things up and allow me to use the whole generic tweet set and more features as opposed to randomly sampling 1/8th of the overall index and only taking the top 200 most frequent features. (Can try DataBricks Pyspark)
Using techniques such as word2vec or Glovo word embeddings to allow the model to better put sequence of words into context and improve prediction.
This will be the goal in my next NLP Project.




## [Project 3: Time-Series Forcasting of Covid-19 Data via manually tuned ARIMA]()

Performed time-series forecasting on NY State’s Covid-19 data with manually tuned ARIMA, SARIMA and SARIMAX (Seasonal ARIMA with exogenous data) models. The SARIMAX model included hypothesis testing of the exogenous data’s correlations.

## [Project 3 A data driven approach to redisigning U of T's MEng. Analytics Emphasis]()

Web Scraped LinkedIn Data Science job descriptions pre-processed and parsed the data before running Agglomerative Clustering on the mentioned skills based on inter-job-description term frequencies and produced a dendrogram to showcase results. 

## [Project 4: Big Data Hadoop MapReduce Program via Hadoop Streaming]()

Built a Hadoop MapReduce k-means clustering program from scratch using Hadoop’s Streaming API.  Orchestrated training iterations via a Bash script to pass data between my Python written mapper and reducer via STDIN and STDOUT.

## [Project 5&6: Machine Learning on Big Data via Spark, PySpark, MLlib]()

- Created a big data movie recommender system using Alternating Least Square (ALS) on Databricks in PySpark leveraging MLlib.]

- Created a big data raud detection system system on Databricks in PySpark leveraging MLlib.


## [Project 6: Neural Network design from the control flow graph to backpropagation to coding the algorithms from scratch for MLP, RNN, GANs and ResNet in Python using only Numpy.]()





<!---
kamalmukhalalaty/kamalmukhalalaty is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
