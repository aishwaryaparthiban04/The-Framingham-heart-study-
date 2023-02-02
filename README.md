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

