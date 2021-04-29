# [About Me](https://www.linkedin.com/in/kamalmukhalalaty/)

- 👋 Hi, I’m Kamal

- 👀 I’m interested in Data Science, Engineering & Analytics

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

## [Time-Series Forcasting of Covid-19 Data via manually tuned ARIMA](https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forcasting)

**For This Project I Performed time-series forecasting on NY State’s Covid-19 data during the start of the second wave (December 2020).**

I experiemnt with:
- Auto Regressive models, 
- Weighted exponential smoothing, 
- manually & Auto tuned ARIMA & SARIMA models 
- SARIMAX models (Seasonal ARIMA with supporting exogenous data)
  - The SARIMAX model training process includes hypothesis testing of the exogenous data’s fitment to the covid case data

The End Goal is to develop a model for a 1 month out of sample prediction (1 month forcast) of covid-19 cases for the state with an uper and lower bound defining best and worst case.

**Supporting data used as exogenous for SARIMAX Model:**

Oxfords covid-19 data hub tracks policy measures across 19 indicators. I used the database to pull a variety of Policy related features and examine some of the response indicies created by Oxford (indices are simple averages of the individual component indicators).
- General info: [https://github.com/OxCGRT/covid-policy-tracker#subnational-data](https://github.com/OxCGRT/covid-policy-tracker#subnational-data)
- Indicies: [https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/index_methodology.md)
For example:
<img src="https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forcasting/blob/main/Images/An%20Example%20of%20the%20Oxford%20Database%20Indicators.png" width="500" height="300">

The Project Workflow was as follows:

1. ARIMA
- Auto
- Manual
Results were not good enough. 

2. SARIMA
- Auto
  - Results were good but due to the use of the automatic function, freedom in tuning and adressing caveats was missing.
- Manual 
  - mse 1132324.045
  - rmse 1064.107
  - mape 0.109
  
  <img src="https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forcasting/blob/main/Images/Best%202-Week%20Prediction%20(Manually%20Tuned%20SARIMA).png" width="500" height="300">
  
3. SARIMAX
- Auto
- Started by selecting the Top 5 indicies most Correlated to covid cases before training my SARIMAX model on them:
  - ConfirmedDeaths
  - 3_Cancel public events
  - C4_Restrictions on gatherings
  - H2_Testing policy
  - H6_Facial Coverings
- Training & Validation
  - mse 2103829.276
  - rmse 1450.458
  - mape 0.112
  
  <img src="https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forcasting/blob/main/Images/SARIMAX%20UnSig.png" width="500" height="300">
  
- Training yeilds that the only Statistically significant indicies W/ large coefficients (Influence of SARIMAX model) are
  - C4_Restrictions on gatherings
  - H2_Testing policy
-  Model trained with only Statsically significant 
  - mse 2142885.617
  - rmse 1463.860
  - mape 0.113

  <img src="https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forcasting/blob/main/Images/SARIMAX%20Top%205.png" width="500" height="300">

 - Despite all this work on trying to build a SARIMAX model and finding the perfect exogenous data set to support it's predictions, the mannually tuned SARIMA Model previously built (2) outperforms all SARIMAX models. for that reason I will use that model to do my one month out prediction.

4. Apply wining SARIMA model for 1-month prediction.
<img src="https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forcasting/blob/main/Images/1%20Month%20out%20prediction%20using%20Best%20Model.png" width="500" height="300">
Forcasting for one month out is as expected a very difficult task as 1 month is a long time however it is reassuring that the forcast's lower bound is very close to actual cases.

A reminder of the 2-week forcast's outcome:

<img src="https://github.com/kamalmukhalalaty/Covid-19-Cases-Time-Series-Forcasting/blob/main/Images/Best%202-Week%20Prediction%20(Manually%20Tuned%20SARIMA).png" width="500" height="300">

## [Salary Exploration & Prediction Challenge Kaggle](https://github.com/kamalmukhalalaty/Kaggle-Salary-Predictions)

### The challenge objective: tell a data story about a subset of the data science community represented in this survey, through a combination of both narrative text and data exploration. A “story” could be defined any number of ways, and that’s deliberate. The challenge is to deeply explore (through data) the impact, priorities, or concerns of a specific group of data science and machine learning practitioners. 

I decided to bo both the data exploration and build a predictive model using logistic regression to predict the salaries of survey participants.

This project involved:
- Data cleaning, wrangling and manipulation
  - Ordinal & Coordinal encoding 
- Imputation of missing values 
- Statistical Analysis 
- Data exploration
- Visualization
- Feature Reduction 
  - Tree Based Feature Importance 
  - Lasso for feature selection
- Hyperparameter Tuning
- Bias-Variance Tradeoff Analyisis

For more details on the challenge itself and the to source the input data please visit: [https://www.kaggle.com/c/kaggle-survey-2019](https://www.kaggle.com/c/kaggle-survey-2019)

### Key Findings & Results

<img src="https://github.com/kamalmukhalalaty/Kaggle-Salary-Predictions/blob/main/Overall%20Distribution%20of%20Salaries.png" width="500" height="300">

The distribution of salaries is very skewed with a very high number of data points for the lowest salary bracket as well as a "bump" in observations at the 100-125k salary bracket. This is due to a high number of respondents from various developping countries with a lower average/median salary (eg. India, which has a large number of survey participants as well while the bump at the 100-125k salary bracket can be seen as a somehwat normal distribution of salaries for the North America/USA alone.

To investgiate this further I have created the following plots:
<img src="https://github.com/kamalmukhalalaty/Kaggle-Salary-Predictions/blob/main/US%20vs%20India%20Salary%20Distributions.png" width="1500" height="300">
<img src="https://github.com/kamalmukhalalaty/Kaggle-Salary-Predictions/blob/main/US%20vs%20India%20Salaries%20W:R:T%20Education.png" width="1500" height="300">

Looking at this we can validate that there is somewhat normal distribution of salaries around 125-150k for the US alone with what could be some outliers in the 0-9,999 salrary bracket; and a skewed distrubtion at 0-10000 for indian respondants alone with some outliers at the >250,000$ salary bracket.

Geography has a large impact on salary, but due to the high variability in number of samples from each geographic region, this impact will be difficult for our models to learn and explain. Additionnaly, as geography is not an ordinal catigiorical feature, it will need to be one-hot-encoded, this will increase model coefficients dramatically with little to no requiered increased expressivity(prone to to overfitting).

## [Sentiment Analysis on Generic Tweets & US Election Specific Posts (NLP)](https://github.com/kamalmukhalalaty/NLP_twitter_Sentiment_Analysis)

For this project I thoroughly explore two labeled datasets, one contains sentiment labeled generic tweets while the other contains 2020 US Election relevant I built a NLP tweet sentiment analysis classifier. In the repository I showcase a notebook where I built a NLP tweet sentiment analysis classifier. 

Generic Tweets Word Cloud: 

<img src="https://github.com/kamalmukhalalaty/Portfolio/blob/main/images/Sentiment%20Analysis%20Word%20Cloud.png" width="400" height="250">

US Election Relevant tweets Word Cloud:

<img src="https://github.com/kamalmukhalalaty/Portfolio/blob/main/images/US%20Election%202020%20Word%20Cloud.png" width="400" height="250">

This project involved:
- A binary classifier is trained and validated on generic tweets from the file named sentiment_analysis.csv. 
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

- A Multi-Class Classification model is then created using the same steps as above to learn feature - negative sentiment reason mappings on the 2020 US election relevant tweets and its. The highest accuracy random forrest classification model had an accuracy at 36% but overfit the data extremely. The logistic regression model had a similar accuracy with less overfitting characteristics but still at unreasonable levels.
  - The model did poorly in my opinion for the following reasons:
    - Unequal distribution of the labeled reasons with Covid significantly outnumbering the others
      - Scoring metric could have been changed to have a weighted accuracy however the class imbalance is too low to justify this. 
    - the sample size of the negative sentement labeled tweets with reasons is small and therefore models have a hard time generalizing on new data from the little they have learned from the small training set.

- Finnaly an MLP-3 is Built using Keras and TF in an attempt to build an even more compatant classifier however the validation accuracy is only 1% higher so the idea is scrapped. 

This was my first portfolio worthy project within the realm of NLP, model performance could be improved in the following ways:

- getting access to massively parallel processing (MPP) to speed things up and allow me to use the whole generic tweet set and more features as opposed to randomly sampling 1/8th of the overall index and only taking the top 200 most frequent features. (Can try DataBricks Pyspark)
- Using techniques such as word2vec or Glovo word embeddings to allow the model to better put sequence of words into context and improve prediction.
  - This will be the goal in my next NLP Project.

## [A data driven approach to Disigning a Data Science Masters Curriculum](https://github.com/kamalmukhalalaty/Data-Driven-Design-of-a-Data-Science-Masters-Curriculum/blob/main/README.md)

This was a group project that involved many parts, I have attached the presentation in the repositiry linked to the title.

My personal contribution to the overall project:

- Web scraped linkedin Data Science Job Postings 
- Exploritory Data Analysis was performed on 
  - Data Camp scraped & Coursera scraped data (Scraping done by my colleagues)
    - Extracting popular topics
    
    <img src="https://github.com/kamalmukhalalaty/Data-Driven-Design-of-a-Data-Science-Masters-Curriculum/blob/main/Images/datacamp_skills.png">
    <img src="https://github.com/kamalmukhalalaty/Data-Driven-Design-of-a-Data-Science-Masters-Curriculum/blob/main/Images/coursera_skills.png">
    
  - Scraped job descriptions
    - Extracting important skills mentioned in job postings
    
    <img src="https://github.com/kamalmukhalalaty/Data-Driven-Design-of-a-Data-Science-Masters-Curriculum/blob/main/Images/skills_from_jobs.png">
    
  - Looking at importance via most Frequent Words (Key words) and Bigrams

- Hierarchical clustering of keywords in job posting data to validate important topics:

The following dendrogram was created to to clusters skills based on inter-job description term frequencies or more concretely, which words or groups of words appear most frequently in each job description. The words we chose to analyze are The top 10 most popular general  skills found in job descriptions, And the most popular programming languages and software packages found from analysis of popular coursera courses and job descriptions.

Based on this, the clusters shown on the dendrogram were derived. 

<img src="https://github.com/kamalmukhalalaty/Data-Driven-Design-of-a-Data-Science-Masters-Curriculum/blob/main/Images/dendrogram_1.png">

Due to the large amount of terms analysed the dendrogram could be better interpreted by
Zeroing in on the most popular languages and software platforms, 


<img src="https://github.com/kamalmukhalalaty/Data-Driven-Design-of-a-Data-Science-Masters-Curriculum/blob/main/Images/dendrogram_2.png">

Here we see 6 clear clusters and it is nice to see that the clusters make sense. We also see languages like r and c tying into larger clusters that contain more specific skills. 

All in all this reduced dendrogram was very informative and will be used in designing a relevant curriculum for the masters program.

## [Project 4: Big Data Hadoop MapReduce Program via Hadoop Streaming]()

Built a Hadoop MapReduce k-means clustering program from scratch using Hadoop’s Streaming API.  Orchestrated training iterations via a Bash script to pass data between my Python written mapper and reducer via STDIN and STDOUT.

## [Project 5&6: Machine Learning on Big Data via Spark, PySpark, MLlib]()

- Created a big data movie recommender system using Alternating Least Square (ALS) on Databricks in PySpark leveraging MLlib.]

- Created a big data raud detection system system on Databricks in PySpark leveraging MLlib.


## [Project 6: Neural Network design from the control flow graph to backpropagation to coding the algorithms from scratch for MLP, RNN, GANs and ResNet in Python using only Numpy.]()

# [Mechatronics Projects:](https://portfolium.com/kamalmukhalala/portfolio)

## Robotics, Control Theory & More:

During my mechanical Engineering undergrad at UofT, I worked on some interesting projects in the realm of Mechatronics.

### [Autonomous Robot Project](https://portfolium.com/entry/autonomous-robot-project)

<img src="https://github.com/kamalmukhalalaty/Portfolio/blob/main/images/MAGGA%20(ROBOT).png" width="500" height="500">

- Led the mechanical design of the robot’s chassis, robotic arm and gripper using SolidWorks.
- Integrated Bluetooth hardware to enable wireless flashing/uploading of code to the microprocessors.
- Programmed obstacle avoidance, localization and path-determination algorithms that interrogated
sensor data using C++ (Arduino IDE) to achieve localization in under 30 seconds.

### [Experimental Helicopter PID Control System Design](https://portfolium.com/entry/httpsyoutube3tm-tcbhyu8)

<img src="https://github.com/kamalmukhalalaty/Portfolio/blob/main/images/PID%20Control%202.png" width="300" height="500">

- Created a dynamic rig to test the one-dimensional motion of a replicated helicopter rotor system.
- Wrote the PID controller code and iteratively tuned control gains using the Ziegler-Nichols method to meet system response requirements, achieving <2% overshoot.

### [Print farm friendly 3D printer Design](https://portfolium.com/entry/print-farm-friendly-3d-printer)

<img src="https://github.com/kamalmukhalalaty/Portfolio/blob/main/images/3D%20Printer%20Isometric%20View.png" width="500" height="500">

- Led the mechanical design of the printer using SolidWorks.



<!---
kamalmukhalalaty/kamalmukhalalaty is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
