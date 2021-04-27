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
      - NLP (RegEx & NLTK)
      - Time Series Forcasting & Analysis  
    - Deep Learning, Nural Networks (CNN, RNN, LSTM, ResNet, GAN):
      - Keras
      - TensorFlow
    - Big Data & Data Engineering:
      - Hadoop MapReduce
      - Hadoop via Python Streaming
      - Apache Spark
        - DataBricks
        - Pyspark
        - RDD & Dataframe manipulation
        - MLlib
    - Cloud Computing:
      - Azure
        - Azure ML
        - Azure SQL
      - Data Engineering
        - Batch: Azure Data Factory, Azure Synapse Analytics
        - Stream: Azure IoT Hubs, Event Hubs, Stream Analytics

- 📫 How to reach me:
  - Email: [kamalmukhalalaty@gmail.com]()
  - Linkedin: [https://www.linkedin.com/in/kamalmukhalalaty/]()

# [Data Science Projects:](https://github.com/kamalmukhalalaty)

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

## [Sentiment Analysis on Generic Tweets & US Election Specific Posts (NLP)](https://github.com/kamalmukhalalaty/NLP_twitter_Sentiment_Analysis)

For this project I thoroughly explore two labeled datasets, one contains sentiment labeled generic tweets while the other contains 2020 US Election relevant I built a NLP tweet sentiment analysis classifier.

In this repository I showcase a notebook where I built a NLP tweet sentiment analysis classifier. 

- The binary classifier is trained and validated on generic tweets from the file named sentiment_analysis.csv. 
  - The extracted corpus is modeled via both Bag of words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF) embeddings.
  - top 200 features were selected based on frequency and the following models were applied using scikit's default settings:
    - LogisticRegression()
    - RandomForestClassifier()
    - DecisionTree
    - SVM
    - KNN
    - XGBoost W/Logistic objective
  - **TF-IDF Embedding W/ Logistic Regression & Random Forest showed some promise so I conducted hyperparameter tuning which yeilded that Logistic Regresion in default setting has the highest validation accuracy of aproximatley 86%.**
  
- The previously defined **best classifer** is then applied on the entirety of the of US_Election_2020.csv dataset containing labeled 2020 US election relevant tweets and its **performance is sub optimal at 58%**. This is primarily due to computational contraints and dimensionality reduction requierments, the top 200 features from the generic tweets were used to train the model and only a randomly sampled eigth of the total dataset index was used, these features are not as informative when it comes to dictating sentiment in the US election tweets as they are insufficiently diverse and unable to effectivly explain the feature to sentiment mappings in the election relevant tweets.

- A Multi-Class Classification model is then created using the same steps as above to try and learn feature to negative sentiment reason mappings on the 2020 US election relevant tweets and its. The highest accuracy random forrest classification mod--el had an accuracy at 36% but overfit the data extremely. The logistic regression model had a similar accuracy with less overfitting characteristics but still at unreasonable levels.
  - The model did poorly in my opinion for the following reasons:
    - Unequal distribution of the labeled reasons with Covid significantly outnumbering the others
      - Scoring metric could have been changed to have a weighted accuracy however the class imbalance is too low to justify this. 
    - the sample size of the negative sentement labeled tweets with reasons is small and therefore models have a hard time generalizing on new data from the little they have learned from the small training set.

- Finnaly an MLP-3 is Built using Keras and TF in an attempt to build an even more compatant classifier however the validation accuracy is only 1% higher so the idea is scrapped. 

This was my first portfolio worthy project within the realm of NLP, model performance could be improved in the following ways:

- getting access to massively parallel processing (MPP) to speed things up and allow me to use the whole generic tweet set and more features as opposed to randomly sampling 1/8th of the overall index and only taking the top 200 most frequent features. (Can try DataBricks Pyspark)
- Using techniques such as word2vec or Glovo word embeddings to allow the model to better put sequence of words into context and improve prediction.
  - This will be the goal in my next NLP Project.


![Sentiment Analysis Data Word Cloud](https://github.com/kamalmukhalalaty/Portfolio/blob/main/images/Sentiment%20Analysis%20Word%20Cloud.png| width=100)

![US Election Data Word Cloud](https://github.com/kamalmukhalalaty/Portfolio/blob/main/images/US%20Election%202020%20Word%20Cloud.png| width=50)



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

# Mechatronics Projects:

## Robotics, Control Theory & More:

In my past life, I worked on some interesting projects in the realm of Mechatronics.

### [Autonomous Robot Project](https://portfolium.com/entry/autonomous-robot-project)

### [Experimental Helicopter PID Control System Design](https://portfolium.com/entry/httpsyoutube3tm-tcbhyu8)
- Created a dynamic rig to test the one-dimensional motion of a replicated helicopter rotor system.
- Wrote the PID controller code and iteratively tuned control gains using the Ziegler-Nichols method to meet system response requirements, achieving <2% overshoot.
### [Print farm friendly 3D printer Design](https://portfolium.com/entry/print-farm-friendly-3d-printer)



<!---
kamalmukhalalaty/kamalmukhalalaty is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
