import os, sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.utils import save_object , train_models, evaluate_best_model
from typing import List


@dataclass
class Model_Trainer_config():
    '''
    creating a class to config path for the model object.
    '''
    model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer():
    '''
    creating a main class of Modeltrainer.
    '''
    # Initializing model training path:
    def __init__(self):
        self.model_trainer_config = Model_Trainer_config()
    

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting independent and dependent variables from train and test array")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],train_arr[:,-1],
                test_arr[:,:-1],test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVC": SVC(),
                "Logistic Regression": LogisticRegression(),
            }
            
            # list of parameters for every model:
            params={
                "Random Forest": {
                    'criterion':['gini', 'entropy'],
                     'n_estimators':[32,70,100,120],
                     'max_features':['sqrt','log2'],
                },
                "Decision Tree":{
                    'criterion':['gini', 'entropy'],
                    'max_features':['sqrt','log2', None],
                    'splitter': ['best','random']
                },
                "Gradient Boosting":{
                    'loss':['log_loss', 'exponential'],
                    'learning_rate':[0.001,0.01,0.1,0.05],
                    'n_estimators':[32,70,100,120],
                    'criterion':['friedman_mse', 'squared_error']
                },
                "K-Neighbors Classifier":{
                    'n_neighbors':[3,5,8,12],
                    'weights':['uniform', 'distance']
                },

                "XGBClassifier":{
                    'booster':['gbtree', 'gblinear','dart']
                },
                "AdaBoost Classifier":{
                    'algorithm':['SAMME', 'SAMME.R'],
                    'learning_rate':[0.001,0.01,0.1,0.05],
                    'n_estimators':[20,32,70,100]
                },
                "SVC":{
                    'C':[0.01,0.05,0.1,0.5,1],
                    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma':['scale','auto'],
                    'probability':[True]
                },
                "Logistic Regression":{
                    'penalty':['l2'],
                    'solver':['lbfgs', 'newton-cg', 'newton-cholesky'],
                    'C':[0.01,0.05,0.1,0.5,1]
                }}


            logging.info("Training With Different Models...")
            model_report:dict=train_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            logging.info("Model Training Completed")
            logging.info(f"Model Report:{model_report}")
            print(model_report)

            best_sorted_dict = evaluate_best_model(model_report)
            

            ## To get best model name from dict
            best_model_name = best_sorted_dict[0][0]
            best_acc_score = best_sorted_dict[0][1][1]
            best_pr_score = best_sorted_dict[0][1][3]

            best_model = models[best_model_name]

            logging.info(f'Best Model Found, Model_Name: {best_model_name},Acc_Score:{best_acc_score}, PR-AUC Score:{best_pr_score}')
            print('\n============================================================================\n')
            print(f'Best Model Found, Model_Name: {best_model_name},Acc_Score:{best_acc_score}, PR-AUC Score:{best_pr_score}')
            print('\n============================================================================\n') 
            
            logging.info("Saving Model File")
            save_object(
                file_path=self.model_trainer_config.model_file_path,
                obj=best_model)

            return best_model_name,best_acc_score

        except Exception as e:
            logging.info("The error is occured in model training process")
            raise CustomException (e,sys)
