<img width="469" alt="Screen Shot 2021-05-03 at 1 30 55 PM" src="https://user-images.githubusercontent.com/72153772/116910692-f46be580-ac13-11eb-95c6-a7c241af6acd.png">



# [About Me](https://www.linkedin.com/in/kamalmukhalalaty/)

- 👋 Hi, I’m Kamal

- 👀 I’m interested in Data Science, Engineering & Analytics

- 🌱 I’m currently completing my Master of Engineering at the University of Toronto specializing in data science
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
      - Time Series Forecasting & Analysis  
    - Deep Learning, Neural Networks (CNN, RNN, LSTM, ResNet, GAN):
      - Keras
      - TensorFlow
    - Big Data & Data Engineering:
      - Hadoop MapReduce
      - Hadoop via Python Streaming
      - Apache Spark
        - Data Bricks
        - Pyspark
        - RDD & Dataframe manipulation
        - MLlib
    -  Cloud Computing:
      - Azure
        - Azure ML
        - Azure SQL
      - Data Engineering
        - Batch: Azure Data Factory, Azure Synapse Analytics
        - Stream: Azure IoT Hubs, Event Hubs, Stream Analytics

- 🦾 Mechanical Engineering Background (BASc. Completed @ UofT 2019)
  - Specialized in Mechatronics & Renewable Energy

- 📫 How to reach me:
  - Email: [kamalmukhalalaty@gmail.com](kamalmukhalalaty@gmail.com)
  - Linkedin: [https://www.linkedin.com/in/kamalmukhalalaty/](https://www.linkedin.com/in/kamalmukhalalaty/)

# [Data Science Projects:](https://github.com/kamalmukhalalaty)

## [Time-Series Forecasting of Covid-19 Data via ARIMA Family of Models](https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forecasting)

**For This Project I Performed time-series forecasting on NY State’s Covid-19 data during the start of the second wave (December 2020).**

I experiment with:
- Auto-Regressive models, 
- Weighted exponential smoothing, 
- manually & Autotuned ARIMA & SARIMA models 
- SARIMAX models (Seasonal ARIMA with supporting exogenous data)
  - The SARIMAX model training process includes hypothesis testing of the exogenous data’s fitment to the covid case data

The End Goal is to develop a model for a 1 month out of sample prediction (1-month forecast) of covid-19 cases for the state with an upper and lower bound defining best and worst case.

**Supporting data used as exogenous for SARIMAX Model:**

Oxfords covid-19 data hub tracks policy measures across 19 indicators. I used the database to pull a variety of Policy related features and examine some of the response indices created by Oxford (indices are simple averages of the individual component indicators).
- General info: [https://github.com/OxCGRT/covid-policy-tracker#subnational-data](https://github.com/OxCGRT/covid-policy-tracker#subnational-data)
- Indices: [https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md)

For example:

<img width="899" alt="An Example of the Oxford Database Indicators" src="https://user-images.githubusercontent.com/72153772/116601504-94fe9480-a8f8-11eb-8563-524bcb1a1e63.png">

The Project Workflow was as follows:

1. ARIMA
- Auto
- Manual
Results were not good enough. 

2. SARIMA
- Auto
  - Results were good but due to the use of the automatic function, freedom in tuning and addressing caveats was missing.
- Manual 
  - mse 1132324.045
  - rmse 1064.107
  - mape 0.109
  
  <img width="923" alt="Best 2-Week Prediction (Manually Tuned SARIMA)" src="https://user-images.githubusercontent.com/72153772/116601547-a21b8380-a8f8-11eb-94db-1f3eaa2d5a6a.png">
  
3. SARIMAX
- Auto
- Started by selecting the top 5 indices most Correlated to covid cases before training my SARIMAX model on them:
  - ConfirmedDeaths
  - 3_Cancel public events
  - C4_Restrictions on gatherings
  - H2_Testing policy
  - H6_Facial Coverings
- Training & Validation
  - mse 2103829.276
  - rmse 1450.458
  - mape 0.112
  
  <img width="907" alt="SARIMAX UnSig" src="https://user-images.githubusercontent.com/72153772/116601615-b0699f80-a8f8-11eb-9a27-194b00b8e1a0.png">  

- Training yields that the only Statistically Significant indices W/ large coefficients (Influence of SARIMAX model) are
  - C4_Restrictions on gatherings
  - H2_Testing policy
-  Model trained with only statistically significant indices
  - mse 2142885.617
  - rmse 1463.860
  - mape 0.113

  <img width="910" alt="SARIMAX Top 5" src="https://user-images.githubusercontent.com/72153772/116601739-d000c800-a8f8-11eb-8798-25651c53afed.png">

 - Despite all this work on trying to build a SARIMAX model and finding the perfect exogenous data set to support its predictions, the manually tuned SARIMA Model previously built (2) outperforms all SARIMAX models. For that reason, I will use that model to do my one month out prediction.

4. Apply winning SARIMA model for 1-month prediction.

<img width="909" alt="1 Month out prediction using Best Model" src="https://user-images.githubusercontent.com/72153772/116601894-ff173980-a8f8-11eb-9191-1a9e16d2dba7.png">

Forecasting for one month out is as expected a very difficult task as 1 month is a long time however it is reassuring that the forecast's lower bound is very close to actual cases.

A reminder of the 2-week forecast's outcome:

<img width="923" alt="Best 2-Week Prediction (Manually Tuned SARIMA)" src="https://user-images.githubusercontent.com/72153772/116601547-a21b8380-a8f8-11eb-94db-1f3eaa2d5a6a.png">

## [Salary Exploration & Prediction Challenge Kaggle](https://github.com/kamalmukhalalaty/Kaggle-Salary-Predictions)

### The challenge objective: tell a data story about a subset of the data science community represented in this survey, through a combination of both narrative text and data exploration. A “story” could be defined any number of ways, and that’s deliberate. The challenge is to deeply explore (through data) the impact, priorities, or concerns of a specific group of data science and machine learning practitioners. 

I decided to do both the data exploration and build a predictive model using logistic regression to predict the salaries of survey participants.

This project involved:
- Data cleaning, wrangling and manipulation
  - Ordinal & Nominal encoding
- Imputation of missing values 
- Statistical Analysis 
- Data exploration
- Visualization
- Feature Reduction 
  - Tree-Based Feature Importance 
  - Lasso for feature selection
- Hyperparameter Tuning
- Bias-Variance Tradeoff Analysis

For more details on the challenge itself and the to source the input data please visit: [https://www.kaggle.com/c/kaggle-survey-2019](https://www.kaggle.com/c/kaggle-survey-2019)

### Key Findings & Results

<img width="716" alt="Overall Distribution of Salaries" src="https://user-images.githubusercontent.com/72153772/116602016-22da7f80-a8f9-11eb-83e5-412d8f97311e.png">

The distribution of salaries is very skewed with a very high number of data points for the lowest salary bracket as well as a "bump" in observations at the 100-125k salary bracket. This is due to a high number of respondents from various developing countries with a lower average/median salary (eg. India, which has a large number of survey participants as well while the bump at the 100-125k salary bracket can be seen as a somewhat normal distribution of salaries for the North America/USA alone.

To investigate this further I have created the following plots:
<img width="950" alt="US vs India Salary Distributions" src="https://user-images.githubusercontent.com/72153772/116602062-35ed4f80-a8f9-11eb-97d4-c023133e71ad.png">
<img width="1150" alt="US vs India Salaries W:R:T Education" src="https://user-images.githubusercontent.com/72153772/116602083-3e458a80-a8f9-11eb-9920-94f3f0c1a59a.png">
Looking at this we can validate that there is a somewhat normal distribution of salaries around 125-150k for the US alone with what could be some outliers in the 0-9,999 salary bracket; and a skewed distribution at 0-10000 for Indian respondents alone with some outliers at the >250,000$ salary bracket.
Geography has a large impact on salary, but due to the high variability in the number of samples from each geographic region, this impact will be difficult for our models to learn and explain. Additionally, as geography is not an ordinal categorical feature, it will need to be one-hot-encoded, this will increase model coefficients dramatically with little to no required increased expressivity(prone to overfitting).

## [Sentiment Analysis on Generic Tweets & US Election Specific Posts (NLP)](https://github.com/kamalmukhalalaty/NLP_twitter_Sentiment_Analysis)

For this project I thoroughly explore two labelled datasets, one contains sentiment labelled generic tweets while the other contains 2020 US Election relevant I built an NLP tweet sentiment analysis classifier. In the repository, I showcase a notebook where I built an NLP tweet sentiment analysis classifier. 

Generic Tweets Word Cloud: 
<img width="450" alt="Sentiment Analysis Word Cloud" src="https://user-images.githubusercontent.com/72153772/116602117-4bfb1000-a8f9-11eb-9983-d063497e79ca.png">
US Election Relevant tweets Word Cloud:
<img width="452" alt="US Election 2020 Word Cloud" src="https://user-images.githubusercontent.com/72153772/116602143-53221e00-a8f9-11eb-8107-58f08f961505.png">

This project involved:
- A binary classifier is trained and validated on generic tweets from the file named sentiment_analysis.csv. 
  - The extracted corpus is modelled via both Bag of words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF) embeddings.
  - top 200 features were selected based on the frequency and the following models were applied using Scikit's default settings:
    - LogisticRegression
    - RandomForestClassifier
    - decision tree
    - SVM
    - KNN
    - XGBoost W/Logistic objective
  - **TF-IDF Embedding W/ Logistic Regression & Random Forest showed some promise so I conducted hyperparameter tuning which yielded that Logistic Regression in default setting has the highest validation accuracy of approximately 86%.** 
- The previously defined **best classifier** is then applied on the entirety of the of US_Election_2020.csv dataset containing labeled 2020 US election relevant tweets and its **performance is suboptimal at 58%**. This is primarily due to computational constraints and dimensionality reduction requirements, the top 200 features from the generic tweets were used to train the model and only a randomly sampled eight of the total dataset index was used, these features are not as informative when it comes to dictating sentiment in the US election tweets as they are insufficiently diverse and unable to effectively explain the feature to sentiment mappings in the election relevant tweets.

- A Multi-Class Classification model is then created using the same steps as above to learn feature - negative sentiment reason mappings on the 2020 US election relevant tweets and its. The highest accuracy random forest classification model had an accuracy at 36% but overfit the data extremely. The logistic regression model had a similar accuracy with less overfitting characteristics but still at unreasonable levels.
  - The model did poorly in my opinion for the following reasons:
    - Unequal distribution of the labelled reasons with Covid significantly outnumbering the others
      - Scoring metric could have been changed to have a weighted accuracy however the class imbalance is too low to justify this. 
    - the sample size of the negative sentiment labelled tweets with reasons is small and therefore models have a hard time generalizing on new data from the little they have learned from the small training set.

- Finally, an MLP-3 is Built using Keras and TF in an attempt to build an even more competent classifier however the validation accuracy is only 1% higher so the idea is scrapped. 

This was my first portfolio-worthy project within the realm of NLP, model performance could be improved in the following ways:

- getting access to massively parallel processing (MPP) to speed things up and allow me to use the whole generic tweet set and more features as opposed to randomly sampling 1/8th of the overall index and only taking the top 200 most frequent features. (Can try DataBricks Pyspark)
- Using techniques such as word2vec or Glovo word embeddings to allow the model to better put a sequence of words into context and improve prediction.
  - This will be the goal in my next NLP Project.

## [A data-driven approach to Designing a Data Science Masters Curriculum](https://github.com/kamalmukhalalaty/Data-Driven-Design-of-a-Data-Science-Masters-Curriculum/blob/main/README.md)

This was a group project that involved many parts, I have attached the presentation in the repository linked to the title.

My personal contribution to the overall project:

- Web scraped LinkedIn Data Science Job Postings 
- Exploratory Data Analysis was performed on 
  - Data Camp scraped & Coursera scraped data (Scraping done by my colleagues)
    - Extracting popular topics
    
    <img width="743" alt="datacamp_skills" src="https://user-images.githubusercontent.com/72153772/116602815-35a18400-a8fa-11eb-8325-4a50ceb2469c.png">
    <img width="792" alt="coursera_skills" src="https://user-images.githubusercontent.com/72153772/116602845-3df9bf00-a8fa-11eb-9617-ab2abb0bc02b.png">
    
  - Scraped job descriptions
    - Extracting important skills mentioned in job postings
    
    <img width="824" alt="skills_from_jobs" src="https://user-images.githubusercontent.com/72153772/116602929-536ee900-a8fa-11eb-8375-e6d835c675c1.png">
    
  - Looking at importance via most Frequent Words (Keywords) and Bigrams

- Hierarchical clustering of keywords in the job posting data to validate important topics:

The following dendrogram was created to cluster skills based on inter-job description term frequencies or more concretely, which words or groups of words appear most frequently in each job description. The words we chose to analyze are The top 10 most popular general skills found in job descriptions, And the most popular programming languages and software packages found from analysis of popular Coursera courses and job descriptions.

Based on this, the clusters shown on the dendrogram were derived. 

<img width="714" alt="dendrogram_1" src="https://user-images.githubusercontent.com/72153772/116602944-5a95f700-a8fa-11eb-9ce4-653d88383585.png">

Due to the large amount of terms analysed the dendrogram could be better interpreted by
Zeroing in on the most popular languages and software platforms, 

<img width="641" alt="dendrogram_2" src="https://user-images.githubusercontent.com/72153772/116602963-5ff34180-a8fa-11eb-9db0-3fe6acaff1b9.png">

Here we see 6 clear clusters and it is nice to see that the clusters make sense. We also see languages like r and c tying into larger clusters that contain more specific skills. 

All in all this reduced dendrogram was very informative and will be used in designing a relevant curriculum for the master's program.

## [Big Data Hadoop MapReduce Program via Hadoop Streaming](https://github.com/kamalmukhalalaty/Big-Data-Hadoop-MapReduce-Program-via-Hadoop-Streaming)

Built a Hadoop MapReduce k-means clustering program from scratch using Hadoop’s Streaming API.  Orchestrated training iterations via a Bash script to pass data between my Python written mapper and reducer via STDIN and STDOUT.

## [Machine Learning on Big Data via Spark, PySpark, MLlib](https://github.com/kamalmukhalalaty/Machine-Learning-on-Big-Data-via-Spark-PySpark-MLlib)

- Created a big data movie recommender system using Alternating Least Square (ALS) on Databricks in PySpark leveraging MLlib.
- Created a big data fraud detection system on Databricks in PySpark leveraging MLlib.

## [Deep Learning Projects](https://github.com/kamalmukhalalaty/Deep-Learning-Projects)

Neural Network design from the control flow graph to backpropagation to coding the algorithms from scratch for MLP, RNN, GANs and ResNet in Python using only Numpy

The Linked repository is in the process of being updated


# [Data Analytics Projects:](https://github.com/kamalmukhalalaty/Data-Analytics)

## [Shopify Data Science Internship Challenge](https://github.com/kamalmukhalalaty/Data-Analytics/blob/main/Shopify_Data_Science_Internship_Challenge_Kamal_Mukhalalaty.ipynb)

# [Mechatronics Projects:](https://portfolium.com/kamalmukhalala/portfolio)

## Robotics, Control Theory & More:

During my Mechanical Engineering undergrad at UofT, I worked on some interesting projects in the realm of Mechatronics.

### [Autonomous Robot Project](https://portfolium.com/entry/autonomous-robot-project)

<img width="377" alt="MAGGA (ROBOT)" src="https://user-images.githubusercontent.com/72153772/116603011-6b466d00-a8fa-11eb-8e2a-e19482013e07.png">

- Led the mechanical design of the robot’s chassis, robotic arm and gripper using SolidWorks.
- Integrated Bluetooth hardware to enable wireless flashing/uploading of code to the microprocessors.
- Programmed obstacle avoidance, localization and path-determination algorithms that interrogated
sensor data using C++ (Arduino IDE) to achieve localization in under 30 seconds.

### [Experimental Helicopter PID Control System Design](https://portfolium.com/entry/httpsyoutube3tm-tcbhyu8)

<img width="228" alt="PID Control" src="https://user-images.githubusercontent.com/72153772/116603372-da23c600-a8fa-11eb-9088-bc9e6246eaf5.png">

- Created a dynamic rig to test the one-dimensional motion of a replicated helicopter rotor system.
- Wrote the PID controller code and iteratively tuned control gains using the Ziegler-Nichols method to meet system response requirements, achieving <2% overshoot.

### [Print farm friendly 3D printer Design](https://portfolium.com/entry/print-farm-friendly-3d-printer)

![](https://user-images.githubusercontent.com/72153772/116603137-9630c100-a8fa-11eb-84fc-296c05720fde.png)

- Led the mechanical design of the printer using SolidWorks.

<!---
kamalmukhalalaty/kamalmukhalalaty is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

