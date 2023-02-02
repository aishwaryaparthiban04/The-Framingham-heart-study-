**THE FRAMINGHAM HEART STUDY**

DATASET- You can access the dataset within this repository.

**1.Data description and Research question**

In this case study, the Framingham heart study cohort data set is used to train and test the models for heart disease classification. The data is collected by examining the town residents from Framingham, Massachusetts, from an ongoing cardiovascular study (Kaggle,2021). This dataset contains over 4,000 records and 16 columns. This dataset was collected initially to study the prevalence of several cardiovascular diseases and the risk factors associated with them. This dataset helps us explore the patterns of CVD and the risk factors over time that affect our lives. The Framingham study was started in 1948 under the supervision of the U.S. Public health Service, which was then transferred under the control of the new National Heart Institute. Participants, including men and women from the town of Framingham, were sampled, and studied to identify the concept of risk factors and the joint effects of CVD, which also facilitates specialized studies.

Based on the problem statement, I have framed a research question that I am going to answer with the help of the is project is given down below.
To identify the most relevant/risk factors of heart disease as well as predict the overall risk of whether the patient has a 10-year risk of future coronary heart disease (CHD).

**2. Data preparation and cleaning**
  
  Some of the steps that we will follow in data preprocessing are given below.
  
  1.Checking for the column with wrong data types assigned and fixing typos errors.
  
  2.Checking for duplicate values in the dataset. To identify duplicate values duplicated function is used, and then we remove it.
  
  3.Checking for the missing values in the dataset and imputations on missing values. 
  
  4.Check for anomalies in the dataset and remove them. To identify the anomalies, a boxplot is used, and then the Interquartile range's value is used to remove them (Ngare and Kennedy, 2019).


**3- Exploratory data analysis**

In this section, EDA was conducted to uncover the hidden pattern and extract important features from the dataset. The data analysis helps improve the analytical process's transparency (Ho Yu, C, 2010).

In this project, some of the steps taken in doing EDA are given below.

1.Performing univariate and bivariate data analysis on the numerical and categorical variables with the help of scatter plots, density plots, bar charts, and many other graphs were used.

2.A correlation plot was used to check the relationship between the variables.

3.The Shapiro test is used to check the normality of data distribution.

4.I am using the unsupervised learning technique like principal component analysis for doing the exploratory data analysis.

**4- Implementation of KNN**

After the **exploratory data analysis** is completed, the next step is to normalize the numerical column for that feature scaling is used.
The second step is to split the dataset into training and testing in the ratio of 75:25. 
The third step is to implement the KNN algorithm. But before that K value is computed with the help of an elbow plot. I have observed that K ___ gives the minimum error. 
The fourth step is to call the KNN model with parameters like k, distance measure, and train on the training data.
The model is evaluated on the testing data with the caret Package.
The target column is imbalanced in nature, so to resolve this issue, downsampling was used. It reduces the no of training samples that falls under the majority class. At k=41, it gives the best accuracy.

**5- High performance computational implementation**

It stands for high-performance computations implementation. It is very beneﬁcial when we have to solve complex Computational problems like predicting accurate diagnosis for diabetes patients. Due to parallel processing, the speed is relatively high. One of the applications of the HPCI technique is fraud detection. 

**5.1- Implementation**

1.First, pyspark and findspark are installed in the Google Colab environment through the pip command.

2.Second, I have created a Spark session with the help of SparkSession.builder() and set the app name as Colab.

3.Third, spark.read.csv is used to read the CSV files and set the header to True, and printSchema() is used to get the schema of each column in tree format. From the tree, you can observe that most columns are in string format.

4.Fourth, in this, you have to drop the irrelevant values from the dataset created at the time of loading data.

5.All the columns have string types, so to correct the data types, I have defined a list of the numerical column and categorical columns.

6.The dataframe is converted to pandas, and I have encoded the categorical column except for the target variable. Then feature scaling is used.

7.a vector assembler combines all the input columns and creates a single vector column named features.

8.StringIndexer is used to encode the target column, which is TenYearCHD, and renamed it as a target. Both vector assembler and stringIndexer are added to the pipeline, created at the seventh step.

9.Ninth, the dataset is split into training and testing in the ratio of 75:25.
Some of the following steps have been taken to implement the K nearest neighbors. First, to reduce the dimensions of the features, principal component analysis is used; after that, extracted features are converted to an array and broadcast to each node. After that KNN algorithm is implemented, which sorts the matrix by row and outputs the first k labels. The label appears the most as the predicted label for the test point.
Evaluate the model using the metrics accuracy, precision, recall, and f1 score.

**6- Performance evaluation and comparison of the methods**

Now it's time to evaluate the models. For this work, classification metrics like confusion matrix, Accuracy, Precision, and recall were used.
The KNN algorithm performed with K value set to 10. It gives an accuracy of 88.41% on the test dataset, the recall value of model is 1 it means the model is correctly detecting all the data points whereas the true negative rate (specificity) is very low around 0.13 which means our model is too eager to find the positive result.
The target variables are imbalanced. So, to tune the parameters trainControl method was used in which 5-fold cross validation was used to balance the target class down sampling. The KNN model runs for 20 different K values and finds that the k=29 model performs the best. It gives an accuracy of 86.7% on the test dataset.

**7- Discussion of the findings**

**7.1- Discussion**

CHD is a common disease and often leads to death due to various complications. In this work, we identify the most crucial factors that lead to CHD and predict the patient's overall risk. Features like age, prevalentHyp, sysBP are the major factors in predicting the coronary heart disease. To obtain this result various types of plots were made like histogram, bar plot and the scatter plot. Additionally a statistical Shapiro wilk test is conducted to find whether the data is normally distributed or not. From the Shapiro result I have found that our data is not normally distributed so to solve this I have used scale function which will scale the feature in the range -1 to 1. Then after that k means clustering is used to group all the features into two clusters. To find this optimal no of cluster values we have used a line chart was made. To find the most important factors with respect to the targeted variable a correlation heatmap was made. After that, a predictive model was made to predict CHD among patients. The target column is imbalanced, so the oversampling technique is performed. Then the KNN model is used which achieves an accuracy of 88.41% with the k value set to 10. The model is tuned and 5-fold cross-validation is used which achieves an accuracy of around 87% on the testing dataset. For the HPCI same algorithm is also used and it achieves lower performance than the traditional knn algorithm with an accuracy of 85.11%. This model also takes less time as compared to the machine learning model.

**7.2- Limitations **

There are only a few features are given for the patient. Therefore, to improve, you can also add additional features like medications history and heart imaging. You can also use an autoencoder for feature extraction and then apply a classification algorithm to improve the performance. I have used only a machine learning model on the imbalance data in this work.
Finally I can conclude that the features like age, prevalentHyp, sysBP are the primary factors in predicting the heart disease and the K nearest algorithm performed well in predicting the heart disease. 

