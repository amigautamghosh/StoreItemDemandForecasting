

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

We have used a maximum of  50 features like 

store 

item

year 

month

day 

weekofyear 

dayofweek 

dayofyear

quarter 

weekend 

weekday_name_Friday 

weekday_name_Monday

weekday_name_Saturday 

weekday_name_Sunday 

weekday_name_Thursday 
 
weekday_name_Tuesday 

weekday_name_Wednesday 
 
mean-store-item 
 
median-store-item
 
mean-month-item 

median-month-item 

sum-month-item

median-month-store

mean-month-store

sum-month-store

mean-item 

median-item

mean-store 
 
median-store

mean-month-item-store

mean-weekofyear-item-store

median-weekofyear-item-store

sum-weekofyear-item-store

mean-quarter-store

median-quarter-store

sum-item-store

mean-quarter-item

median-quarter-item

sum-item-quarter

mean-weekend-item-store
 
median-weekend-item-store


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
