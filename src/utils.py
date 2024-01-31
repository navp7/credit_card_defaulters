import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score,average_precision_score


# Defining a Function to save pickle files at provided directory:
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("Exception occured while saving object file")
        raise CustomException(e,sys)



def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info('Exception occured in load_object function utlis')
        raise CustomException(e,sys)


def evaluate_model(true, predicted):
    cl_report = classification_report(true, predicted)
    con_mat = confusion_matrix(true, predicted)
    roc_score = roc_auc_score(true,predicted)*100
    acc_score = accuracy_score(true, predicted)*100
    
    return cl_report, con_mat,roc_score, acc_score

def train_models(X_train,y_train,X_test,y_test,models,params):
    '''
    evalaute_models is used to train the data across a list of models by performing 
    Hyperparameter tunning and finding the best parameter within it. It returns the 
    accuracy report as well to figure out the best model to be deployed.
    '''
    try:
        report={}
            
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs = GridSearchCV(estimator=model,param_grid=param, cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #Make Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred=model.predict(X_test)

            # print('Model Training Performance')
            cl_report, con_mat,roc_score, test_acc_score = evaluate_model(y_test, y_test_pred)
            train_acc_score = accuracy_score(y_train,y_train_pred)*100
            
            # Evaluating Precision-Recall Curve:
            y_pred_proba1 = model.predict_proba(X_test)[::,1]
            pr_auc_score = average_precision_score(y_test, y_pred_proba1)
            

            report[list(models.keys())[i]] = train_acc_score,test_acc_score,roc_score,pr_auc_score,cl_report,con_mat

        return report

    except Exception as e:
        logging.info("Error occured while model training and hyperparameter tunning")
        raise CustomException (e,sys)


def evaluate_best_model(report):
    '''
    best_model evaluates the best model based on model report and returns sorted dict of models

    '''
    try:
        dict_final ={}
        for key,value in report.items():
            
            if report[key][0]>report[key][1]:
                if abs(report[key][0]-report[key][1]) < 15:
                    dict_final[key] = list(value)
        
        b_model = sorted(dict_final.items(),key = lambda x:x[1][3], reverse = True)

        return b_model

    except Exception as e:
        logging.info("Error occured while evaluating best model")
        raise CustomException (e,sys)
