import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

import argparse


def get_data():
    location = 'E:\\8. INEURON\\2. September_batch\\2. Machine Learning\\3. ML datasets'
    file_name='54. winequality-red.csv'
    df = pd.read_csv(f'{location}\\{file_name}')
    
    return df


def evaluate_fun(y_pred,y_test):
    try:
        MAE=mean_absolute_error(y_pred,y_test)
        MSE=mean_squared_error(y_pred,y_test)
        RMSE=np.sqrt(mean_squared_error(y_pred,y_test))
        R2=r2_score(y_pred,y_test)
        
        return MAE,MSE,RMSE,R2
    except Exception as e:
        raise e
    
    
def evaluate_randomforest(y_test,y_pred,y_pred_prob):
    try:
        ac=accuracy_score(y_test,y_pred)
        auc_roc = roc_auc_score(y_test,y_pred_prob,multi_class='ovr')
        return ac, auc_roc
    except Exception as e:
        raise e
    
    
    
def main(n_estimators, max_depth):
    try:
        df=get_data()
        train ,test = train_test_split(df,test_size=0.33,random_state=42)
        X_train=train.drop(['quality'],axis=1)
        X_test=test.drop(['quality'],axis=1)
        y_train = train[['quality']]
        y_test = test[['quality']]
        
        """model = ElasticNet()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)"""
        with mlflow.start_run():
            
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)
            ac,roc=evaluate_randomforest(y_test,y_pred,y_pred_prob) 
        
            """MAE,MSE,RMSE,R2=evaluate_fun(y_pred,y_test) 
            print(f"mae:- {MAE*100}")
            print(f"mse;- {MSE*100}")
            print(f"Rmse:- {RMSE*100}")
            print(f"R2:- {R2}") """   
            mlflow.log_param("n_estimators",n_estimators)
            mlflow.log_param("max_deopth",max_depth)
            mlflow.log_metric("aur_roc_score",roc)
            mlflow.log_metric("accuracy",ac)
            
             
            print(f"accuracy :- {ac*100}")
            print(f"auc_roc :- {roc*100}")
        
    except Exception as e:
        raise e
    
    
    
if __name__ == '__main__':
    try:
        args=argparse.ArgumentParser()
        args.add_argument("--n_estimators",'-n',default=50,type=int)
        args.add_argument("--max_depth",'-a',default=5,type=int)
        parse_arg=args.parse_args()
        print(parse_arg.n_estimators,parse_arg.max_depth)
    
        m = main(n_estimators=parse_arg.n_estimators, max_depth=parse_arg.max_depth)
        print(m)
    except Exception as e:
        raise e