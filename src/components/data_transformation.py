import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object


@dataclass
class DataTransformationConfi:
    preprocessorpath = os.path.join('artifacts','preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfi()

    def get_data_transformer_object(self):
        try:
            numerical_columns=["writing score", "reading score"]
            categorical_columns =["gender",
                                  "race/ethnicity",
                                  "parental level of education",
                                  "lunch",
                                  "test preparation course"]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder",OneHotEncoder())
                ]
            )

            logging.info("Column Transformation completed")

            preprocessor = ColumnTransformer(
                [
                    ('Num_Pipeline',num_pipeline,numerical_columns),
                    ('Cat_Pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            preprocessor_obj = self.get_data_transformer_object()

            target_colum_name = "math score"

            input_feature_train_df = train_df.drop(columns=[target_colum_name],axis=1)
            target_feature_train_df = train_df[target_colum_name]

            input_feature_test_df = test_df.drop(columns=[target_colum_name],axis=1)
            target_feature_test_df = test_df[target_colum_name]

            input_feature_train_array =  preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_array,np.array(target_feature_test_df)
            ]

            save_object (
                file_path = self.data_transformation_config.preprocessorpath,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessorpath
            )
        except Exception as e:
            raise CustomException(e,sys)
        



