
# Exercise 4: Loan Prediction Practice Problem

## Position In Leaderboard: 962 Score: 0.777778

![leaderboard](https://github.com/zamiramos/ex4/blob/master/Submission2.PNG)

## Username: antomis

Based on this great tutorial: https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/

From the challange hosted at: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan.

The company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers. Here they have provided a partial data set.

## The Data

Variable | Description
----------|--------------
Loan_ID | Unique Loan ID
Gender | Male/ Female
Married | Applicant married (Y/N)
Dependents | Number of dependents
Education | Applicant Education (Graduate/ Under Graduate)
Self_Employed | Self employed (Y/N)
ApplicantIncome | Applicant income
CoapplicantIncome | Coapplicant income
LoanAmount | Loan amount in thousands
Loan_Amount_Term | Term of loan in months
Credit_History | credit history meets guidelines
Property_Area | Urban/ Semi Urban/ Rural
Loan_Status | Loan approved (Y/N)


Evaluation Metric is accuracy i.e. percentage of loan approval you correctly predict.

You may upload the solution in the format of "sample_submission.csv"

## Setups

To begin, start iPython interface in Inline Pylab mode by typing following on your terminal / windows command prompt:


```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib
    

    C:\Program Files\Anaconda2\lib\site-packages\IPython\core\magics\pylab.py:161: UserWarning: pylab import has clobbered these variables: ['table', 'plt']
    `%matplotlib` prevents importing * from pylab and numpy
      "\n`%matplotlib` prevents importing * from pylab and numpy"
    

Following are the libraries we will use during this task:
- numpy
- matplotlib
- pandas

Please note that you do not need to import matplotlib and numpy because of Pylab environment. I have still kept them in the code, in case you use the code in a different environment.


```python
import pandas as pd
import numpy as np
import matplotlib as plt
```

After importing the library, you read the dataset using function read_csv(). The file is assumed to be downloaded from the moodle to the data folder in your working directory.


```python
df = pd.read_csv("./data/train.csv") #Reading the dataset in a dataframe using Pandas
```

## Let’s begin with exploration

### Quick Data Exploration

Once you have read the dataset, you can have a look at few top rows by using the function head()


```python
df.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001002</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001005</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001006</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001008</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LP001011</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>5417</td>
      <td>4196.0</td>
      <td>267.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LP001013</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2333</td>
      <td>1516.0</td>
      <td>95.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LP001014</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3036</td>
      <td>2504.0</td>
      <td>158.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LP001018</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4006</td>
      <td>1526.0</td>
      <td>168.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LP001020</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>12841</td>
      <td>10968.0</td>
      <td>349.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>



This should print 10 rows. Alternately, you can also look at more rows by printing the dataset.

Next, you can look at summary of numerical fields by using describe() function


```python
df.describe() # get the summary of numerical variables
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>614.000000</td>
      <td>614.000000</td>
      <td>592.000000</td>
      <td>600.00000</td>
      <td>564.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5403.459283</td>
      <td>1621.245798</td>
      <td>146.412162</td>
      <td>342.00000</td>
      <td>0.842199</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6109.041673</td>
      <td>2926.248369</td>
      <td>85.587325</td>
      <td>65.12041</td>
      <td>0.364878</td>
    </tr>
    <tr>
      <th>min</th>
      <td>150.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>12.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2877.500000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3812.500000</td>
      <td>1188.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5795.000000</td>
      <td>2297.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>81000.000000</td>
      <td>41667.000000</td>
      <td>700.000000</td>
      <td>480.00000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



describe() function would provide count, mean, standard deviation (std), min, quartiles and max in its output

## Distribution analysis

Now that we are familiar with basic data characteristics, let us study distribution of various variables. Let us start with numeric variables – namely ApplicantIncome and LoanAmount

Lets start by plotting the histogram of ApplicantIncome using the following commands:


```python
df['ApplicantIncome'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10ea5f28>




![png](output_27_1.png)


Here we observe that there are few extreme values. This is also the reason why 50 bins are required to depict the distribution clearly.

Next, we look at box plots to understand the distributions. Box plot can be plotted by:


```python
df.boxplot(column='ApplicantIncome')
```

    C:\Program Files\Anaconda2\lib\site-packages\ipykernel\__main__.py:1: FutureWarning: 
    The default value for 'return_type' will change to 'axes' in a future release.
     To use the future behavior now, set return_type='axes'.
     To keep the previous behavior and silence this warning, set return_type='dict'.
      if __name__ == '__main__':
    




    {'boxes': [<matplotlib.lines.Line2D at 0x113d4e10>],
     'caps': [<matplotlib.lines.Line2D at 0x113e3c50>,
      <matplotlib.lines.Line2D at 0x113ee208>],
     'fliers': [<matplotlib.lines.Line2D at 0x113eecf8>],
     'means': [],
     'medians': [<matplotlib.lines.Line2D at 0x113ee780>],
     'whiskers': [<matplotlib.lines.Line2D at 0x10fe3160>,
      <matplotlib.lines.Line2D at 0x113e36d8>]}




![png](output_29_2.png)


This confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Part of this can be driven by the fact that we are looking at people with different education levels. Let us segregate them by Education:


```python
df.boxplot(column='ApplicantIncome', by = 'Education')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11582ac8>




![png](output_31_1.png)


We can see that there is no substantial different between the mean income of graduate and non-graduates. But there are a higher number of graduates with very high incomes, which are appearing to be the outliers.

# Task 2: Distribution Analysis

Plot the histogram and boxplot of LoanAmount

## Check yourself:


```python
df['LoanAmount'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x118e6080>




![png](output_36_1.png)



```python
df.boxplot(column='LoanAmount')
```

    C:\Program Files\Anaconda2\lib\site-packages\ipykernel\__main__.py:1: FutureWarning: 
    The default value for 'return_type' will change to 'axes' in a future release.
     To use the future behavior now, set return_type='axes'.
     To keep the previous behavior and silence this warning, set return_type='dict'.
      if __name__ == '__main__':
    




    {'boxes': [<matplotlib.lines.Line2D at 0x11d463c8>],
     'caps': [<matplotlib.lines.Line2D at 0x11d58278>,
      <matplotlib.lines.Line2D at 0x11d587f0>],
     'fliers': [<matplotlib.lines.Line2D at 0x11d64320>],
     'means': [],
     'medians': [<matplotlib.lines.Line2D at 0x11d58d68>],
     'whiskers': [<matplotlib.lines.Line2D at 0x118e60f0>,
      <matplotlib.lines.Line2D at 0x11d46cc0>]}




![png](output_37_2.png)


Again, there are some extreme values. Clearly, both ApplicantIncome and LoanAmount require some amount of data munging. LoanAmount has missing and well as extreme values values, while ApplicantIncome has a few extreme values, which demand deeper understanding. We will take this up in coming sections.

## Categorical variable analysis

Frequency Table for Credit History:


```python
temp1 = df['Credit_History'].value_counts(ascending=True)
temp1
```




    0.0     89
    1.0    475
    Name: Credit_History, dtype: int64



Probability of getting loan for each Credit History class:


```python
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
temp2
```




    Credit_History
    0.0    0.078652
    1.0    0.795789
    Name: Loan_Status, dtype: float64



This can be plotted as a bar chart using the “matplotlib” library with following code:


```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
```




    <matplotlib.text.Text at 0x12271ef0>




![png](output_45_1.png)


This shows that the chances of getting a loan are eight-fold if the applicant has a valid credit history. You can plot similar graphs by Married, Self-Employed, Property_Area, etc.

Alternately, these two plots can also be visualized by combining them in a stacked chart::


```python
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x122f9550>




![png](output_47_1.png)


We just saw how we can do exploratory analysis in Python using Pandas. I hope your love for pandas (the animal) would have increased by now – given the amount of help, the library can provide you in analyzing datasets.

Next let’s explore ApplicantIncome and LoanStatus variables further, perform data munging and create a dataset for applying various modeling techniques. I would strongly urge that you take another dataset and problem and go through an independent example before reading further.

## Function For Building Model


```python
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  #print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  #print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
  
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])
  return {'accuracy':accuracy ,'cv':np.mean(error)}
```

## How to fill missing values in LoanAmount?


How to fill missing values in LoanAmount?

There are numerous ways to fill the missing values of loan amount – the simplest being replacement by mean, which can be done by following code:



```python
# df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
```

The other extreme could be to build a supervised learning model to predict loan amount on the basis of other variables and then use age along with other variables to predict survival.

Since, the purpose now is to bring out the steps in data munging, I’ll rather take an approach, which lies some where in between these 2 extremes. A key hypothesis is that whether a person is educated or self-employed can combine to give a good estimate of loan amount.

But first, we have to ensure that each of Self_Employed and Education variables should not have a missing values.

As we say earlier, Self_Employed has some missing values. Let’s look at the frequency table:


```python
df['Self_Employed'].value_counts()
```




    No     500
    Yes     82
    Name: Self_Employed, dtype: int64



Since ~86% values are “No”, it is safe to impute the missing values as “No” as there is a high probability of success. This can be done using the following code:


```python
df['Self_Employed'].fillna('No',inplace=True)
```

Now, we will create a Pivot table, which provides us median values for all the groups of unique values of Self_Employed and Education features. Next, we define a function, which returns the values of these cells and apply it to fill the missing values of loan amount:


```python
tableLoanAmount = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
tableLoanAmount
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Education</th>
      <th>Graduate</th>
      <th>Not Graduate</th>
    </tr>
    <tr>
      <th>Self_Employed</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>No</th>
      <td>130.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>Yes</th>
      <td>157.5</td>
      <td>130.0</td>
    </tr>
  </tbody>
</table>
</div>



Define function to return value of this pivot_table:


```python
def fage(x):
 return tableLoanAmount.loc[x['Self_Employed'],x['Education']]
```

Replace missing values:


```python
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
```

This should provide you a good way to impute missing values of loan amount.

## How to treat for extreme values in distribution of LoanAmount and ApplicantIncome?

Let’s analyze LoanAmount first. Since the extreme values are practically possible, i.e. some people might apply for high value loans due to specific needs. So instead of treating them as outliers, let’s try a log transformation to nullify their effect:


```python
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x123dd828>




![png](output_67_1.png)


Now the distribution looks much closer to normal and effect of extreme values has been significantly subsided.

Coming to ApplicantIncome. One intuition can be that some applicants have lower income but strong support Co-applicants. So it might be a good idea to combine both incomes as total income and take a log transformation of the same.


```python
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1252beb8>




![png](output_69_1.png)


Now we see that the distribution is much better than before. 

# Submission 1

## Feature Engineering

### Fill missing Values

Let us look at missing values in all the variables.


```python
 df.apply(lambda x: sum(x.isnull()),axis=0) 
```




    Loan_ID               0
    Gender               13
    Married               3
    Dependents           15
    Education             0
    Self_Employed         0
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount            0
    Loan_Amount_Term     14
    Credit_History       50
    Property_Area         0
    Loan_Status           0
    LoanAmount_log        0
    TotalIncome           0
    TotalIncome_log       0
    dtype: int64



I build a generic function to predict categorical features. It will fill the NA values based on other records.


```python
from sklearn.preprocessing import LabelEncoder

def fill_na_by_predication(model, data,predictors,outcome):    
    #first convert the datatype
    data_correct_types = data.copy()
    var_mod = list(predictors)
    var_mod.append(outcome)
    le = LabelEncoder()
    for i in var_mod:
        data_correct_types[i] = le.fit_transform(data_correct_types[i].astype(str))
    
    #Fit the model:
    outcomeNansMask = data[outcome].isnull()
    
    #check if there are NA values
    if not (outcomeNansMask == True).any():
        return
    
    model.fit(data_correct_types[predictors][~outcomeNansMask],data_correct_types[outcome][~outcomeNansMask])
    
    #Make predictions on training set:
    predictions = model.predict(data_correct_types[predictors][~outcomeNansMask])
    
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data_correct_types[outcome][~outcomeNansMask])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    
    predictionsOutcomeNans = model.predict(data_correct_types[predictors][outcomeNansMask])
    
    predictionsOutcomeNans = le.inverse_transform(predictionsOutcomeNans)
    
    resultDFPredication = pd.DataFrame(predictionsOutcomeNans, columns=[outcome], index=data[outcomeNansMask].index)
    
    #print(pd.concat([resultDFPredication, data[outcome][~outcomeNansMask]], axis=1)[outcomeNansMask])
    #print(resultDFPredication.loc[104])
    
    #print(predictionsOutcomeNans[:,1])
    #print(pd.concat([data['Loan_ID'][outcomeNansMask], pd.DataFrame(predictionsOutcomeNans, columns=[outcome], index=outcomeNansMask)] ,axis=1))
    
    #data[data[outcome].isnull()].apply(lambda x: resultDFPredication.loc[x.name], axis=1)
    
    data[outcome].fillna(data[data[outcome].isnull()].apply(lambda x: resultDFPredication.loc[x.name][outcome], axis=1), inplace=True)
    
    #data[outcome][outcomeNansMask] = resultDFPredication
```

#### Fill Married

Let's fill na of Married Catagory by looking on Education, Dependents, ApplicantIncome, Gender, Property_Area.


```python
outcome_var = 'Married'
modelFillMarried = DecisionTreeClassifier()
predictor_var = ['Education','ApplicantIncome','Gender','Property_Area']

fill_na_by_predication(modelFillMarried, df,predictor_var,outcome_var)
```

    Accuracy : 98.200%
    

#### Fill Gender

Let's check the connection between Education and Gender.


```python
temp3 = pd.crosstab(df['Education'], df['Gender'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12c6bba8>




![png](output_83_1.png)



```python
df['Gender'].value_counts()
```




    Male      489
    Female    112
    Name: Gender, dtype: int64



We can see that the ratio is not significantly changed.

Let's check the connection between ApplicantIncome and Gender.


```python
table = df.pivot_table(values='LoanAmount', index='Gender', aggfunc=np.mean)
table
```




    Gender
    Female    126.879464
    Male      148.421268
    Name: LoanAmount, dtype: float64



We can see that LoanAmount can direct on the gender.

Let's check the connection between Married and Gender.


```python
temp3 = pd.crosstab(df['Married'], df['Gender'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12b27ba8>




![png](output_90_1.png)


When the applicant person is married, in most of the cases the gender will be male.

Let's check the connection between Property_Area and Gender. 


```python
temp3 = pd.crosstab(df['Property_Area'], df['Gender'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12e74400>




![png](output_93_1.png)


Here we see that there is a little tilt in each Property_Area categorial againt the gender ratio.

Let's check the connection between Dependents and Gender. 


```python
temp3 = pd.crosstab(df['Dependents'], df['Gender'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12f568d0>




![png](output_96_1.png)


Here we see that above 2 Dependents most likely to be Male. 

Now taking all those conclusion, we will predict all our NA values using Dependents, Property_Area, Married, LoanAmount.


```python
outcome_var = 'Gender'
modelFillGender = DecisionTreeClassifier()
predictor_var = ['Dependents','Married','LoanAmount','Property_Area']

fill_na_by_predication(modelFillGender, df,predictor_var,outcome_var)
```

    Accuracy : 95.840%
    

#### Fill Credit_History

We see from above that we have a lot of missing values in Credit_History catagory. Let's print again the Credit_History againt Loan Status.


```python
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1258e710>




![png](output_102_1.png)


We can see that if there is no credit_history the chances to get loan is very low, and the opposite if we have.

If we will fill the missing values to one of those values then we will get alot of false postive, or postive false. 
I think that the best option is to open a new catogary, let's say 2, which say that the history is unknown.


```python
df['Credit_History_Unknown'] = df['Credit_History'].fillna(2.0)
```


```python
temp3 = pd.crosstab(df['Credit_History_Unknown'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x135b46d8>




![png](output_105_1.png)


#### Dependents

We already saw that there is a connection between Dependents and Gender. Let's search other connection for the predection.

Lets explore other connection, Dependents and Property_Area.


```python
temp3 = pd.crosstab(df['Dependents'], df['Property_Area'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12ad2978>




![png](output_109_1.png)


We don't see here tilt to some categorial.


```python
table = df.pivot_table(values='TotalIncome', index='Dependents', aggfunc=np.mean)
table
```




    Dependents
    0      6541.119188
    1      7388.509804
    2      6614.027723
    3+    10605.529412
    Name: TotalIncome, dtype: float64



Here we see a good connection.

Lets check the connection between the Loan_Amount_Term and Dependents.


```python
table = df.pivot_table(values='Loan_Amount_Term', index='Dependents', aggfunc=np.mean)
table
```




    Dependents
    0     348.107784
    1     329.346535
    2     340.871287
    3+    325.200000
    Name: Loan_Amount_Term, dtype: float64



Also here we can see that mean is little diffrent releated to Dependents.

Now taking all those conclusion, we will predict all our NA values using Loan_Amount_Term, LoanAmount, Gender.


```python
outcome_var = 'Dependents'
modelFillDependents = DecisionTreeClassifier()
predictor_var = ['TotalIncome','Loan_Amount_Term','Gender']

fill_na_by_predication(modelFillDependents, df,predictor_var,outcome_var)
```

    Accuracy : 98.164%
    

#### Loan_Amount_Term

Lets fill Loan_Amount_Term with mean median value of 360


```python
df['Loan_Amount_Term'].fillna(360, inplace=True)
```

## Algorithm Description

VotingClassifier - Soft Voting/Majority Rule classifier built from:

1. An extra-trees classifier.
This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.

2. LogisticRegression classifier.
This class implements regularized logistic regression using the ‘liblinear’ library, ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers. It can handle both dense and sparse input. Use C-ordered arrays or CSR matrices containing 64-bit floats for optimal performance; any other input format will be converted (and copied).




## Algorithm Calibration

First use encoder to convert catagorial features to scalar.


```python
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
datasetLe = LabelEncoder()
for i in var_mod:
    df[i] = datasetLe.fit_transform(df[i].astype(str))
```

Check Types


```python
df.dtypes
```




    Loan_ID                    object
    Gender                      int64
    Married                     int64
    Dependents                  int64
    Education                   int64
    Self_Employed               int64
    ApplicantIncome             int64
    CoapplicantIncome         float64
    LoanAmount                float64
    Loan_Amount_Term          float64
    Credit_History            float64
    Property_Area               int64
    Loan_Status                 int64
    LoanAmount_log            float64
    TotalIncome               float64
    TotalIncome_log           float64
    Credit_History_Unknown    float64
    dtype: object


Great we can continue.
Define the range of the paramters we want to estimate


```python
#number of trees
n_estimators_vector = range(100, 1000, 200)

#Minmum split
min_samples_split_vector = range(20, 31, 5)

#Maximum Features
max_features_vector = range(2, 4, 1)
```


```python
from sklearn.ensemble import ExtraTreesClassifier

resultCalibrationDF = pd.DataFrame(columns = ['n_estimators', 'min_samples_split', 'max_features','Accuracy', 'Cross-Validation Score'])

outcome_var = 'Loan_Status'
predictor_var = ['Gender','Married','Dependents','Education', 'Self_Employed', 'TotalIncome_log', 'LoanAmount_log', 'Property_Area']

#train the model
for n_estimator in n_estimators_vector:
    for min_samples_split in min_samples_split_vector:
        for max_features in max_features_vector:
            model_etc = ExtraTreesClassifier(n_estimators=n_estimator, min_samples_split=min_samples_split, max_depth=None, max_features=max_features)
            result = classification_model(model_etc, df,predictor_var,outcome_var)
            #insert result to data frame
            resultCalibrationDF = resultCalibrationDF.append({'n_estimators':n_estimator,'min_samples_split':min_samples_split,'max_features':max_features, 'Accuracy':result['accuracy'], 'Cross-Validation Score':result['cv']}, ignore_index=True)

```


```python
resultCalibrationDF.plot(x=['n_estimators', 'min_samples_split', 'max_features'], y= ['Accuracy','Cross-Validation Score'], legend = True, subplots = True)
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x0000000013AC10F0>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x00000000141BA080>], dtype=object)




![png](output_132_1.png)


We can see that the best combination is n_estimators:500 min_samples_split:25 max_features:3
Lets train our model for this specific paramters:


```python
model_etc = ExtraTreesClassifier(n_estimators=500, min_samples_split=25, max_depth=None, max_features=3)
result = classification_model(model_etc, df,predictor_var,outcome_var)
print("Accuracy : %s" % "{0:.3%}".format(result['accuracy']))
print("Cross-Validation Score : %s" % "{0:.3%}".format(result['cv']))
```

    Accuracy : 75.081%
    Cross-Validation Score : 67.100%
    

I want now to create a simple model only for the dominant feature 'Credit_History'


```python
outcome_var = 'Loan_Status'
model_credit_history = LogisticRegression()
predictor_var = ['Credit_History_Unknown','LoanAmount_log']
result = classification_model(model_credit_history, df,predictor_var,outcome_var)
print("Accuracy : %s" % "{0:.3%}".format(result['accuracy']))
print("Cross-Validation Score : %s" % "{0:.3%}".format(result['cv']))
```

    Accuracy : 80.945%
    Cross-Validation Score : 80.946%
    

Ensemble between the model.


```python
from sklearn.ensemble import VotingClassifier

modelEnsemble = VotingClassifier(estimators=[('lr', model_credit_history), ('etc', model_etc)], voting='hard')
predictor_var = ['Gender','Married','Dependents','Education', 'Self_Employed', 'TotalIncome_log', 'LoanAmount_log', 'Property_Area','Credit_History_Unknown'] 
result = classification_model(modelEnsemble, df,predictor_var,outcome_var)
print("Accuracy : %s" % "{0:.3%}".format(result['accuracy']))
print("Cross-Validation Score : %s" % "{0:.3%}".format(result['cv']))
```

    Accuracy : 82.736%
    Cross-Validation Score : 80.296%
    

## Prepare Test Data


```python
testDF = pd.read_csv("./data/test.csv") #Reading the test dataset in a dataframe using Pandas
```

Let us look at missing values in all the variables.


```python
 testDF.apply(lambda x: sum(x.isnull()),axis=0) 
```




    Loan_ID               0
    Gender               11
    Married               0
    Dependents           10
    Education             0
    Self_Employed        23
    ApplicantIncome       0
    CoapplicantIncome     0
    LoanAmount            5
    Loan_Amount_Term      6
    Credit_History       29
    Property_Area         0
    dtype: int64



Fill the data with the same logic as we analyzed the training dataset

#### Gender


```python
from sklearn.preprocessing import LabelEncoder

def fill_na_by_predication_without_training(model, data,predictors,outcome):    
    #first convert the datatype
    data_correct_types = data.copy()
    var_mod = list(predictors)
    var_mod.append(outcome)
    le = LabelEncoder()
    for i in var_mod:
        data_correct_types[i] = le.fit_transform(data_correct_types[i].astype(str))
    
    #Fit the model:
    outcomeNansMask = data[outcome].isnull()
    
    #check if there are NA values
    if not (outcomeNansMask == True).any():
        print('NA values not found')
        return
    
    predictionsOutcomeNans = model.predict(data_correct_types[predictors][outcomeNansMask])
    
    predictionsOutcomeNans = le.inverse_transform(predictionsOutcomeNans)
    
    resultDFPredication = pd.DataFrame(predictionsOutcomeNans, columns=[outcome], index=data[outcomeNansMask].index)
    
    #print(pd.concat([resultDFPredication, data[outcome][~outcomeNansMask]], axis=1)[outcomeNansMask])
    #print(resultDFPredication.loc[104])
    
    #print(predictionsOutcomeNans[:,1])
    #print(pd.concat([data['Loan_ID'][outcomeNansMask], pd.DataFrame(predictionsOutcomeNans, columns=[outcome], index=outcomeNansMask)] ,axis=1))
    
    #data[data[outcome].isnull()].apply(lambda x: resultDFPredication.loc[x.name], axis=1)
    
    data[outcome].fillna(data[data[outcome].isnull()].apply(lambda x: resultDFPredication.loc[x.name][outcome], axis=1), inplace=True)
    
    print('NA values successfuly filled')
    
    #data[outcome][outcomeNansMask] = resultDFPredication
```


```python
outcome_var = 'Gender'
predictor_var = ['Dependents','Married','LoanAmount','Property_Area']

fill_na_by_predication_without_training(modelFillGender, testDF,predictor_var,outcome_var)
```

    NA values successfuly filled
    

#### Fill Credit_History


```python
testDF['Credit_History_Unknown'] = testDF['Credit_History'].fillna(2.0)
```

#### Fill Self_Employed


```python
testDF['Self_Employed'].fillna('No',inplace=True)
```

#### Fill LoanAmount / TotalIncome / LoanAmount_log / TotalIncome_log


```python
testDF['LoanAmount'].fillna(testDF[testDF['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
testDF['LoanAmount_log'] = np.log(testDF['LoanAmount'])
testDF['TotalIncome'] = testDF['ApplicantIncome'] + testDF['CoapplicantIncome']
testDF['TotalIncome_log'] = np.log(testDF['TotalIncome'])
```

#### Fill Dependents


```python
outcome_var = 'Dependents'
predictor_var = ['TotalIncome','Loan_Amount_Term','Gender']

fill_na_by_predication_without_training(modelFillDependents, testDF,predictor_var,outcome_var)
```

    NA values successfuly filled
    

## Predict Test Dataset

First use encoder to convert catagorial features to scalar.


```python
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    testDF[i] = le.fit_transform(testDF[i].astype(str))
```

Finally Predict:


```python
predictor_var = ['Gender','Married','Dependents','Education', 'Self_Employed', 'TotalIncome_log', 'LoanAmount_log', 'Property_Area','Credit_History_Unknown'] 
testVectorPrediction = modelEnsemble.predict(testDF[predictor_var])
testVectorPrediction = datasetLe.inverse_transform(testVectorPrediction)
testDFPrediction = pd.DataFrame(testVectorPrediction, columns=['Loan_Status'], index=testDF['Loan_ID'].index)
finalResultsDF = pd.concat([testDF['Loan_ID'], testDFPrediction], axis=1)
```


```python
#export to csv
finalResultsDF.to_csv(path_or_buf="./data/Submission1.csv", index=False)
```

## Link To Submission File
[Submission1-Link](https://github.com/zamiramos/ex4/blob/master/Submission1.csv)

## Leaderboard

![leaderboard](https://github.com/zamiramos/ex4/blob/master/Submission1.PNG)

# Submission 2

## Feature Engineering

Same as Submission 1 (above)

## Algorithm Description

An AdaBoost classifier.

An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

## Algorithm Calibration


```python
from sklearn.ensemble import AdaBoostClassifier

#number of trees
n_estimators_vector = range(50, 500, 50)

#Minmum split
learning_rate_vector = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

outcome_var = 'Loan_Status'
resultCalibrationDF = pd.DataFrame(columns = ['n_estimators', 'learning_rate','Accuracy', 'Cross-Validation Score'])

predictor_var = ['Gender','Married','Dependents','Education', 'Self_Employed', 'TotalIncome_log', 'LoanAmount_log', 'Property_Area','Credit_History_Unknown']

#train the model
for n_estimator in n_estimators_vector:
    for rate in learning_rate_vector:
        modelAdaBoost = AdaBoostClassifier(n_estimators=n_estimator, learning_rate=rate)
        result = classification_model(modelAdaBoost, df,predictor_var,outcome_var)
        #insert result to data frame
        resultCalibrationDF = resultCalibrationDF.append({'n_estimators':n_estimator,'learning_rate':rate, 'Accuracy':result['accuracy'], 'Cross-Validation Score':result['cv']}, ignore_index=True)

```


```python
resultCalibrationDF.plot(x=['n_estimators', 'learning_rate'], y= ['Accuracy','Cross-Validation Score'], legend = True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x143cdd30>




![png](output_171_1.png)


We can see that the more we increase the number of the estimators the more we get overfitting 

Lets choose the best combintion for the CV 50 estimators with learning_rate of 0.1


```python
modelAdaBoost = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
result = classification_model(modelAdaBoost, df,predictor_var,outcome_var)
print("Accuracy : %s" % "{0:.3%}".format(result['accuracy']))
print("Cross-Validation Score : %s" % "{0:.3%}".format(result['cv']))
```

    Accuracy : 81.433%
    Cross-Validation Score : 80.621%
    

## Prepare Test Data

(already done in submission 1) You must run submission 1 before.

## Predict Test Dataset


```python
predictor_var = ['Gender','Married','Dependents','Education', 'Self_Employed', 'TotalIncome_log', 'LoanAmount_log', 'Property_Area','Credit_History_Unknown'] 
testVectorPrediction = modelAdaBoost.predict(testDF[predictor_var])
testVectorPrediction = datasetLe.inverse_transform(testVectorPrediction)
testDFPrediction = pd.DataFrame(testVectorPrediction, columns=['Loan_Status'], index=testDF['Loan_ID'].index)
finalResultsDF = pd.concat([testDF['Loan_ID'], testDFPrediction], axis=1)
```


```python
#export to csv
finalResultsDF.to_csv(path_or_buf="./data/Submission2.csv", index=False)
```

## Link To Submission File
[Submission2-Link](https://github.com/zamiramos/ex4/blob/master/Submission2.csv)

## Leaderboard

![leaderboard](https://github.com/zamiramos/ex4/blob/master/Submission2.PNG)
