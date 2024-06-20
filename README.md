# Supply Chain Optimization Project

![Supply Chain Optimization](https://media.licdn.com/dms/image/D4D12AQHUAzXJvlf-KA/article-cover_image-shrink_600_2000/0/1672033188187?e=2147483647&v=beta&t=rNMOICJQ0GG6OL4UoGYCthPvJhNaNl64DG1QSXQNYY0)

## Project Overview

This project focuses on optimizing the supply chain to minimize volatility in inventory stock of pallets, reduce costs, and enhance overall efficiency. By leveraging data analytics and visualization tools, this project aims to provide actionable insights for better decision-making.

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Business Objective](#business-objective)
- [Business Constraint](#business-constraint)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Usage](#usage)
- [Project Structure](#project-structure)


## Business Problem

The number of pallets stored in inventory for shipping to different customers is highly volatile, leading to understocking or overstocking. Understocking results in unmet client requirements, while overstocking incurs higher inventory costs.

## Business Objective

Minimize the volatility in inventory stock to ensure a balanced and cost-effective supply chain.

## Business Constraint

Minimize human intervention in the supply chain optimization process.

## Tech Stack

- **Python**
- **MySQL**
- **Matplotlib**
- **Seaborn**
- **Power BI**
- **Looker Studio**
- **Excel**

## Features

- **Data Cleaning and Preprocessing**: Identification and removal of outliers, handling of missing values, and ensuring data consistency.
- **Data Exploration and Analysis**: In-depth analysis using Matplotlib and Seaborn to uncover patterns and insights.
- **Interactive Dashboards**: Dynamic dashboards created with Power BI, Looker Studio, and Excel for effective data visualization and reporting.

    ```

## Usage

1. Load the dataset:
    ```python
    import pandas as pd
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
    ```
2. Run the data cleaning and preprocessing scripts:
    ```python
      # Typecasting
      raw_data['CustName']=raw_data['CustName'].astype(str)
      raw_data['WHName']=raw_data['WHName'].astype(str)
      # Changing the Integer column to Categorical Datatype

     # Handling Duplicates
      data_no_duplicates=raw_data.drop_duplicates(keep='last')
      print(data_no_duplicates)
      # We can see that there are 16962 duplicates
    
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
    ```
3. Perform data exploration and visualization:
    ```python
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

    ```
4. Generate interactive dashboards:
    Open the Power BI / Looker Studio / Excel files in the `dashboards` directory to view the interactive visualizations.

## Project Structure
supply-chain-optimization/
- data/
  - raw_data.csv
- scripts/
  - data_cleaning.py
  - data_analysis.py
- dashboards/
  - dashboard.pbix
  - dashboard.looker
  - dashboard.xlsx
- images/
  - charts.png
  - [README.md](https://github.com/NaveenM-10/optimization-supply-chain-management/new/main?filename=README.md)
  - requirements.txt
