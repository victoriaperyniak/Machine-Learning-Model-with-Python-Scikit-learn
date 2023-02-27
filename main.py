# project: p7
# submitter: peryniak
# partner: none
# hours: 6
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
#References: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
#https://www.geeksforgeeks.org/replace-nan-values-with-zeros-in-pandas-dataframe/
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.count.html
#https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

class UserPredictor:
    def __init__(self):
        self.xcols = ["age", "past_purchase_amt", "badge", "url", "seconds"]
        
        #TODO:
        #build custom transformer to feed into pipeline:
        #Polynomial: numeric data, OneHotEncoder: categorical data
        self.custom_trans = make_column_transformer(
             (PolynomialFeatures(), ["age", "past_purchase_amt", "url", "seconds"]),
             (OneHotEncoder(), ["badge"]),
         )
        #create pipeline:
        #1. transform data 2.scale data (do outside of custom_trans) 3.create logistic reg. model
        self.pipeline = Pipeline([
            ("both", self.custom_trans),
            ("scalr", StandardScaler()),
            ("logr", LogisticRegression()),
        ])
        
    def fit(self, train_users, train_logs, train_y):
        #create a new df that contains the user info and their browsing history (# webpages visited and total time spent on the websites). This must be done for the train and test datasets, separately
        train_idandurl = train_logs[["user_id","url"]]
        train_idandsec = train_logs[["user_id", "seconds"]]
        new_df1 = pd.merge(train_users, train_idandurl.groupby("user_id").count(), how = "left", on="user_id").fillna(0) #url column is the number of webpages the user visited on the website
        new_df2 = pd.merge(new_df1, train_idandsec.groupby("user_id").sum(), how = "left", on="user_id").fillna(0) #seconds column is the total number of seconds the user spent on the website
        
        #fit new data
        self.pipeline.fit(new_df2[self.xcols], train_y["y"])
        
    def predict(self, test_users, test_logs):
        test_idandurl = test_logs[["user_id","url"]]
        test_idandsec = test_logs[["user_id", "seconds"]]
        new_df1 = pd.merge(test_users, test_idandurl.groupby("user_id").count(), how = "left", on="user_id").fillna(0) #url column is the number of webpages the user visited on the website
        new_df2 = pd.merge(new_df1, test_idandsec.groupby("user_id").sum(), how = "left", on="user_id").fillna(0) #seconds column is the total number of seconds the user spent on the website
        
        #predict new data
        return self.pipeline.predict(new_df2[self.xcols])