# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:55:40 2023

@author: Naveen
"""

# Importing Pandas Library 
import pandas as pd


# Loading Raw Data into Python Using Pandas Library
raw_data=pd.read_csv(r'C:/Users/Naveen/Desktop/pallet_Masked_fulldata.csv')

# Database Connectivity
## pip install sqlalchemy
from sqlalchemy import create_engine

## pip install pymysql
import pymysql

# Creating a connector(i.e. engine) between python sql
engine=create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                     .format(user="root",
                             pw="password",
                             db="inventory"))

# Storing the SQL Query int
sql='select * from pallet'

# Connecting The Database & Reading the data
raw_data=pd.read_sql_query(sql, engine)


# EDA BEFORE DATA PRE-PROCESSING
# First Moment Business Decision
    # Calculating Mode for City Column
city_mode=raw_data.City.mode()
print(city_mode)

# Calculating Mode for Region
region_mode=raw_data['Region'].mode()
print(region_mode)

# Calculating Mode for State
state_mode=raw_data['State'].mode()
print(state_mode)

# Calculating Mode for Product Code
product_code_mode=raw_data['Product_Code'].mode()
print(product_code_mode)

# Calculating Mode For Transaction Type
transaction_type_mode=raw_data['Transaction_Type'].mode()
print(transaction_type_mode)

# Calculating Mode For CustName 
custname_mode=raw_data['CustName'].mode()
print(custname_mode)

# Calculating Mean For QTY
qty_mean=raw_data['QTY'].mean()
print(qty_mean)

# Calculating Median For QTY
qty_median=raw_data['QTY'].median()
print(qty_median)

# Calculating Mode For QTY
qty_mode=raw_data['QTY'].mode()
print(qty_mode)

# Second Moment Business Dicision
# Calculating Standard Deviation For QTY
qty_std=raw_data['QTY'].std()
print(qty_std)

# Calculating Variance For QTY
qty_var=raw_data['QTY'].var()
print(qty_var)

# Calculating Range For QTY
qty_range=max(raw_data['QTY']-min(raw_data['QTY']))
print(qty_range)

# Third Moment Business Decision
# Calculating Skewness For QTY
qty_skew=raw_data['QTY'].skew()
print(qty_skew)

# Fourth Moment Business Decision
# Caluculating Kurtosis For QTY
qty_kurt=raw_data['QTY'].kurt()
print(qty_kurt)

# Calculating Mode For WHName
whname_mode=raw_data['WHName'].mode()
print(whname_mode)


# DATA PRE-PROCESSING

# Typecasting

raw_data['CustName']=raw_data['CustName'].astype(str)

raw_data['WHName']=raw_data['WHName'].astype(str)
# Changing the Integer column to Categorical Datatype


# Handling Duplicates
data_no_duplicates=raw_data.drop_duplicates(keep='last')
print(data_no_duplicates)
# We can see that there are no duplicates

# Outlier Treatment
Q1=data_no_duplicates['QTY'].quantile(0.25)
Q3=data_no_duplicates['QTY'].quantile(0.75)
IQR=Q3-Q1
upper_limit = Q3+(1.5*IQR)
lower_limit = Q1-(1.5*IQR)
outliers = data_no_duplicates[(data_no_duplicates['QTY'] <= lower_limit) & (data_no_duplicates['QTY'] >= upper_limit)]
print(outliers)
# Thier are no outliers in the Qty

# Zero or Near Zero variance
variance=data_no_duplicates[['QTY']].var()
near_zero_variance=variance[variance<0.01]
print(near_zero_variance)

# Missing Values
raw_data_no_missing=data_no_duplicates.dropna(inplace=True)
print(raw_data_no_missing)
# We can see that thier are no missing values

# Normalisation
from sklearn.preprocessing import StandardScaler
S_scaler=StandardScaler()
data_no_duplicates['Scaled_QTY']=S_scaler.fit_transform(data_no_duplicates[['QTY']])

# Transformation
from feature_engine import transformation
tf=transformation.YeoJohnsonTransformer()
data_no_duplicates['Transformed_QTY']=tf.fit_transform(data_no_duplicates[['Scaled_QTY']])

# Graphical Representation
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram For CustName
plt.hist(data_no_duplicates['CustName'])
plt.title('Histogram For CustName')
plt.xlabel('CustName')
plt.ylabel('Count')
plt.show()
# Customers From 0 to 1000 are more interactively participated in the Business


# Histogram For Region
plt.hist(data_no_duplicates['Region'])
plt.title('Histogram For Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.show()
# The performance in East Region is less When Compared to Nort,South & West

# Histogram For State
plt.figure(figsize=(12,4))
plt.hist(data_no_duplicates['State'])
plt.title('Histogram For State')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()
# Karnataka,Uttar Pradesh & West Bengal are the top states in Number Of Transaction
#Mizoram,Sikkim,Chandigarh are low in Business

# Histogram For QTY
sns.histplot(data_no_duplicates['QTY'],kde=True)
plt.title('Histogram for QTY')
plt.xlabel('QTY')
plt.ylabel('Frequency')
plt.show()
# we can see that the data is not normally distributed

# Histogram for Transformed Qty
sns.histplot(data_no_duplicates['Transformed_QTY'],kde=True)
plt.title('Histogram for Transformed QTY')
plt.xlabel('Transformed_QTY')
plt.ylabel('Frequency')
plt.show()
# After transforming the feature the data is now distributed normally.

# Histogram For CustName,Region,State
sns.histplot(data_no_duplicates['CustName'],kde=True)
# Customers From 0 to 1000 are more interactively participated in the Business

sns.histplot(data_no_duplicates['Region'],kde=True)
# North Region is slightly better that South and West

sns.histplot(data_no_duplicates['State'],kde=True)
plt.xticks(rotation=90)
# Maharastra has high number of Counts tha Uttar Pradesh & karnataka
# It is Negatively Skewed

# Boxplot For Qty 
sns.boxplot(data_no_duplicates['QTY'])

# Scatterplot between product_code and qty
sns.scatterplot(data_no_duplicates,x=data_no_duplicates['QTY'],y=data_no_duplicates['Product_Code'])
plt.title('Scatter Plot')
plt.xlabel('QTY')
plt.ylabel('Product_Code')
plt.show()


sns.scatterplot(data_no_duplicates,x=data_no_duplicates['date'],y=data_no_duplicates['QTY'])
plt.title('Scatter Plot')
plt.xlabel('Date')
plt.ylabel('QTY')
plt.show()
# 2019 to 2020 there were no returns
# From 2020 more number of returns were registered

# Probability plot 
import scipy.stats as stats
import pylab
stats.probplot(data_no_duplicates['Transformed_QTY'], dist= stats.norm, plot= pylab)



# AUTOEDA Libraries
# 1. Sweetviz
# pip install sweetviz
import sweetviz as sv
s=sv.analyze(data_no_duplicates)
s.show_html()

#2 Dtale
# pip install dtale
import dtale
d=dtale.show(data_no_duplicates)
d.open_browser()

# 3. Autoviz
#pip install autoviz

from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
a=AV.AutoViz("C:/Users/Naveen/Desktop/Clean data csv.csv",chart_format="html")

# 4. Mitosheet
# pip install Mitosheet
import mitosheet
mitosheet.sheet("data_no_duplicates")

# 5. Dataprep
# pip install dataprep
from dataprep.eda import create_report
report=create_report(data_no_duplicates,title='my report')
report.save('dataprep.html')

data_no_duplicates.to_csv("Clean data1",encoding='utf-8')

