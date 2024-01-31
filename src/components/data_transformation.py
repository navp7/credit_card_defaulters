import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from src.utils import save_object
from src.constant import *



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path =  os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_DataTransformation_obj(self):
        try:
            preprocessor = Pipeline(
                            steps=[('imputer',SimpleImputer(strategy='mean')),
                                    ('scaler',StandardScaler())]   
                            )   
            return preprocessor   

        except Exception as e:
            logging.info("Error is occured in creating data transformation object")
            raise CustomException (e,sys)


    def initiate_DataTransformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            train_df.rename(columns={"default payment next month": TARGET_COLUMN}, inplace=True)
            test_df.rename(columns={"default payment next month": TARGET_COLUMN}, inplace=True)

            logging.info('Train and test data read sucessful')
            logging.info(f'Train DataFrame Head: /n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: /n{test_df.head().to_string()}')

            #train dataframe
            input_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_train_df = train_df[TARGET_COLUMN]

            #test dataframe
            input_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_test_df = test_df[TARGET_COLUMN]

            
            logging.info("Obtaining Preprocessor File/Object")
            preprocessing_obj = self.get_DataTransformation_obj()

            ## Data Transformation:
            input_train_trans_df = preprocessing_obj.fit_transform(input_train_df)
            input_test_trans_df = preprocessing_obj.transform(input_test_df)

            logging.info('Completing preprocessing on train and test data')

            train_arr = np.c_[input_train_trans_df,np.array(target_train_df)]
            test_arr = np.c_[input_test_trans_df,np.array(target_test_df)]

            # Calling save_object from utlis.py 
            # Saving preprocessing_obj at the artifacts destination
            logging.info("Saving Preprocessor File")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
        except Exception as e:
            logging.info('Error occures at Data Transformation Stage')
            raise CustomException (e,sys)