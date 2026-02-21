import os,sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from src.components.model_params import model_params


@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__ (self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(**model_params["Random Forest"]),
                "Decision Tree": DecisionTreeRegressor(**model_params["Decision Tree"]),
                "Gradient Boosting": GradientBoostingRegressor(**model_params["Gradient Boosting"]),
                "Linear Regression": LinearRegression(**model_params["Linear Regression"]),
                "K-Neighbors Regressor": KNeighborsRegressor(**model_params["K-Neighbors Regressor"]),
                "XGBRegressor": XGBRegressor(**model_params["XGBRegressor"]),
                "CatBoosting Regressor": CatBoostRegressor(**model_params["CatBoosting Regressor"]),
                "AdaBoost Regressor": AdaBoostRegressor(**model_params["AdaBoost Regressor"])
            }

            models_report : dict=evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            # to get the best model score from dict
            best_model_score = max(models_report.values())

            #to get the best model name from dict
            best_model_name = max(models_report, key=models_report.get)

            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                logging.info("No best model found")
                raise CustomException("No best model found", sys)
            
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square, best_model_name
        except Exception as e:
            logging.info("Error occured in model training")
            raise CustomException(e, sys)