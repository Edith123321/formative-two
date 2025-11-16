import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

class ProductRecommender:
    def __init__(self, model_type='xgb', class_weight=None, random_state=42):
        """Initialize the recommender with specified model type and class weights."""
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None
        self.best_params_ = None
        self.scaler = None

    def train(self, X_train, y_train, cv=5):
        """Train the model with cross-validation and hyperparameter tuning."""
        
        # Add data validation
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Need at least 2 classes, got {len(np.unique(y_train))}")
        
        # --------- MODEL SELECTION --------- #
        if self.model_type == 'xgb':
            self.model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=self.random_state
            )

            # NOTE: Cannot use class_weight directly for XGB multi-class
            # We skip it here

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                random_state=42,
                class_weight=self.class_weight
            )

            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        # --------- GRID SEARCH --------- #
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation on the entire dataset."""
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        return {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'all_scores': cv_scores.tolist()
        }
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk."""
        return joblib.load(filepath)
