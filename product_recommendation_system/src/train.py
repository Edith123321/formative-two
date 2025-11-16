import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from src.data_processing import DataProcessor
from src.model import ProductRecommender
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def analyze_data(X, y, name):
    """Print data analysis information."""
    print(f"\n=== {name} Data Analysis ===")
    print(f"Samples: {len(X)}")
    
    # Class distribution
    class_counts = pd.Series(y).value_counts()
    print("\nClass distribution:")
    print(class_counts)
    print("\nClass proportions:")
    print(class_counts / len(y))
    
    # Feature statistics
    if hasattr(X, 'shape') and len(X.shape) > 1:
        print(f"\nNumber of features: {X.shape[1]}")
        
        # Print basic statistics for the first few features
        if X.shape[1] > 0:
            print("\nFeature statistics (first 5 features):")
            for i in range(min(5, X.shape[1])):
                print(f"Feature {i+1}: min={X[:, i].min():.2f}, max={X[:, i].max():.2f}, mean={X[:, i].mean():.2f}, std={X[:, i].std():.2f}")
    print("=" * 30)

def train_model(data_dir='data/raw', model_type='xgb', test_size=0.2, random_state=42):
    """Complete training pipeline with enhanced diagnostics."""
    
    try:
        # 1. Load and preprocess data
        print("Loading and preprocessing data...")
        processor = DataProcessor(data_dir)
        
        # 2. Perform EDA
        print("\nPerforming Exploratory Data Analysis...")
        from src.eda import perform_eda
        perform_eda(processor)
        
        # Get processed data
        X_train, X_test, y_train, y_test, preprocessor = processor.prepare_data(
            test_size=test_size,
            random_state=random_state
        )
        
        print("\nTraining baseline model...")
        baseline_models = []

        # Try HistGradientBoosting first
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            baseline_models.append(('HistGradientBoosting', HistGradientBoostingClassifier(random_state=random_state)))
        except ImportError:
            print("HistGradientBoosting not available, trying XGBoost...")

        # Then try XGBoost
        try:
            from xgboost import XGBClassifier
            baseline_models.append(('XGBoost', XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=random_state
            )))
        except ImportError:
            print("XGBoost not available, trying RandomForest...")

        # Finally try RandomForest if others fail
        if not baseline_models:
            from sklearn.ensemble import RandomForestClassifier
            baseline_models.append(('RandomForest', RandomForestClassifier(random_state=random_state)))

        # Train and evaluate each baseline model
        for name, model in baseline_models:
            try:
                print(f"\nTraining {name} baseline model...")
                model.fit(X_train, y_train)
                baseline_acc = model.score(X_test, y_test)
                print(f"{name} baseline accuracy: {baseline_acc:.4f}")
                break  # Stop at the first successful model
            except Exception as e:
                print(f"Warning: Could not train {name} model: {str(e)}")
                if model == baseline_models[-1][1]:  # If this was the last model
                    print("Warning: Could not train any baseline models, continuing with main model...")
        
        # 3. Train main model
        print("\nTraining main model...")
        try:
            # Calculate class weights
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = dict(zip(classes, weights))
            print("Using class weights:", class_weight)
        except Exception as e:
            print(f"Could not compute class weights: {e}")
            class_weight = None
            
        recommender = ProductRecommender(model_type=model_type, class_weight=class_weight)
        recommender.train(X_train, y_train)
        
        # 4. Evaluate on test set
        print("\nEvaluating model...")
        test_metrics = recommender.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # 5. Save everything
        print("\nSaving model and artifacts...")
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'product_recommender.joblib')
        processor_path = os.path.join('models', 'preprocessor.joblib')
        
        # Save model
        recommender.save_model(model_path)
        
        # Save preprocessor and label encoder
        processor.save_processor(processor_path)
        
        print(f"\nTraining complete!")
        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {processor_path}")
        
        return recommender, test_metrics
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()