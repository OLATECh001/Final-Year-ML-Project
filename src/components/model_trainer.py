import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        report = {}
        trained_models = {}

        for name, model in models.items():
            logging.info(f"Training model: {name}")
            
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            report[name] = {
                "Accuracy": acc,
                "F1 Score": f1
            }

            trained_models[name] = model

            logging.info(f"{name} -> Accuracy: {acc}, F1 Score: {f1}")

        return report, trained_models

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # ✅ Your tuned models (use your best hyperparameters here)
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
                "Gradient Boost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
                "SVM": SVC(kernel='rbf', C=1, probability=True),
                "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss'),
                "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
            }

            model_report, trained_models = self.evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            # ✅ Select best model based on F1-score
            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["F1 Score"]
            )

            best_model = trained_models[best_model_name]
            best_f1 = model_report[best_model_name]["F1 Score"]

            logging.info(f"Best Model: {best_model_name} with F1 Score: {best_f1}")

            if best_f1 < 0.6:
                raise CustomException("No good model found (F1 < 0.6)", sys)

            # ✅ Save best trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)