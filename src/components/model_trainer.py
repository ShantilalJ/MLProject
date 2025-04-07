import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainigConfig:
    trained_model_file_path = os.path.join("artifcats","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainigConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "RandomForestRegressor" : RandomForestRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "LinearRegression" : LinearRegression(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "KNeighborsRegressor" : KNeighborsRegressor()
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best Model Found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test,predicted)
            return r2score

        except Exception as e:
            raise CustomException(e,sys)
