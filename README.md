

# StoreItemDemandForecasting
Predict 3 months of item sales at different stores

Steps
Data Analysis

Data Cleansing
Feature Engineering
Model Identification
Train and validate the model
Predict the price for test.csv data using the trained model

Data Analysis
A.	Numerical features
•	Sales – predict 3 months of item-level sales data at different store locations 

Data Cleansing
Data was in a good state and we did not have to clean.

B.	Categorical features
•date - Date of the sale data. There are no holiday effects or store closures.
•store - Store ID
•item - Item ID
•sales - Number of items sold at a particular store on a particular date.

Feature Engineering
We have used a maximum of 16 features.
Date: We categorised  this field into weekofyear,dayofweek,day,weekday,weekday_name,dayofyear,quarter,is_month_start,is_month_end,is_quarter_start,is_quarter_end,is_year_start,is_year_end,freq,daysinmonth etc 

Model Identification

Identified 2-3 algorithms to fulfil the requirement (Linear Regression,Random Forest)
We applied linear regression and Random Forest and found the Best Fit 
Compared (Linear and Random Forest) based on performance - and decided to go ahead with Random Forest.

Train and Validate the model
We have used Random Forest model from scikit learn library to train the model.

Potential shortcomings
We could have identified more number of Features

Possible improvements
We can try other algorithms like XGBoost, ARIMA(Autoregressive Integrated Moving Average ) to compare error rate
