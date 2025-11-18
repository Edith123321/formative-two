import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Tuple, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    A machine learning model trainer for facial recognition, voice verification, 
    and product recommendation systems.
    """
    
    # Model configuration constants
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    RF_ESTIMATORS = 100
    
    # Column names for data preparation
    FACE_DROP_COLS = ['member_id', 'image_file', 'augmentation', 'expression']
    AUDIO_DROP_COLS = ['member_id', 'audio_file', 'augmentation', 'phrase']
    PRODUCT_DROP_COLS = [
        'product_category', 'customer_id_new', 'customer_id_legacy', 
        'transaction_id', 'purchase_date'
    ]
    
    def __init__(self):
        """Initialize the ModelTrainer with empty models and scalers."""
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
    
    def prepare_facial_data(self, image_features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for facial recognition model.
        
        Args:
            image_features_df: DataFrame containing image features and labels
            
        Returns:
            Tuple of features (X) and labels (y)
        """
        try:
            X = image_features_df.drop(columns=self.FACE_DROP_COLS, errors='ignore')
            y = image_features_df['member_id']
            
            # Handle missing values
            X = X.fillna(0)
            
            logger.info(f"Facial data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except KeyError as e:
            logger.error(f"Missing required columns in facial data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error preparing facial data: {e}")
            raise
    
    def prepare_audio_data(self, audio_features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for voice verification model.
        
        Args:
            audio_features_df: DataFrame containing audio features and labels
            
        Returns:
            Tuple of features (X) and labels (y)
        """
        try:
            X = audio_features_df.drop(columns=self.AUDIO_DROP_COLS, errors='ignore')
            y = audio_features_df['member_id']
            
            # Handle missing values
            X = X.fillna(0)
            
            logger.info(f"Audio data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except KeyError as e:
            logger.error(f"Missing required columns in audio data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error preparing audio data: {e}")
            raise
    
    def prepare_product_data(self, merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for product recommendation model.
        
        Args:
            merged_df: DataFrame containing customer and transaction data
            
        Returns:
            Tuple of features (X) and labels (y)
        """
        try:
            X = merged_df.drop(columns=self.PRODUCT_DROP_COLS, errors='ignore')
            y = merged_df['product_category']
            
            # Handle categorical variables and missing values
            X = pd.get_dummies(X)
            X = X.fillna(0)
            
            logger.info(f"Product data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except KeyError as e:
            logger.error(f"Missing required columns in product data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error preparing product data: {e}")
            raise
    
    def _initialize_model(self, model_type: str) -> Any:
        """
        Initialize the appropriate model based on type.
        
        Args:
            model_type: Type of model to initialize ('random_forest', 'logistic_regression')
            
        Returns:
            Initialized model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        model_registry = {
            'random_forest': RandomForestClassifier(
                n_estimators=self.RF_ESTIMATORS, 
                random_state=self.DEFAULT_RANDOM_STATE
            ),
            'logistic_regression': LogisticRegression(random_state=self.DEFAULT_RANDOM_STATE),
        }
        
        if model_type not in model_registry:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(model_registry.keys())}")
        
        return model_registry[model_type]
    
    def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: pd.Series, 
                       model_name: str, model_type: str) -> Dict[str, float]:
        """
        Evaluate model performance and log results.
        
        Args:
            model: Trained model instance
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            model_type: Type of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log results
        logger.info(f"{model_name} - {model_type} Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        print(f"\n{model_name} - {model_type} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        return {'accuracy': accuracy, 'f1_score': f1}
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_type: str = 'random_forest', 
                   model_name: str = 'default') -> Tuple[Any, Dict[str, float]]:
        """
        Train a model with given parameters.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            model_name: Name for the trained model
            
        Returns:
            Tuple of (trained_model, evaluation_metrics)
            
        Raises:
            ValueError: If input data is invalid
        """
        # Validate inputs
        if X.empty or y.empty:
            raise ValueError("Input features and labels cannot be empty")
        
        if len(X) != len(y):
            raise ValueError("Features and labels must have the same length")
        
        logger.info(f"Training {model_name} with {model_type} on {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.DEFAULT_TEST_SIZE, 
            random_state=self.DEFAULT_RANDOM_STATE,
            stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[model_name] = scaler
        
        # Initialize and train model
        model = self._initialize_model(model_type)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_test_scaled, y_test, model_name, model_type)
        
        # Store model
        self.models[model_name] = model
        
        logger.info(f"Successfully trained {model_name} with accuracy: {metrics['accuracy']:.4f}")
        return model, metrics
    
    def save_models(self, models_dir: str) -> None:
        """
        Save trained models and scalers to disk.
        
        Args:
            models_dir: Directory to save models and scalers
            
        Raises:
            OSError: If directory cannot be created or files cannot be written
        """
        import os
        os.makedirs(models_dir, exist_ok=True)
        
        try:
            # Save models
            for name, model in self.models.items():
                model_path = f"{models_dir}/{name}_model.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved model: {model_path}")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                scaler_path = f"{models_dir}/{name}_scaler.pkl"
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved scaler: {scaler_path}")
                
            logger.info(f"All models and scalers saved to {models_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models to {models_dir}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about trained models.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'trained_models': list(self.models.keys()),
            'available_scalers': list(self.scalers.keys()),
            'model_counts': {
                'models': len(self.models),
                'scalers': len(self.scalers)
            }
        }
        return info
    
    def clear_models(self) -> None:
        """Clear all stored models and scalers."""
        self.models.clear()
        self.scalers.clear()
        logger.info("Cleared all models and scalers")

# Example usage
if __name__ == "__main__":
    # Example of how to use the ModelTrainer
    trainer = ModelTrainer()
    
    # Example with dummy data (replace with actual data)
    try:
        # This is just an example - replace with your actual data loading
        dummy_features = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'member_id': np.random.choice(['member1', 'member2'], 100)
        })
        
        X, y = trainer.prepare_facial_data(dummy_features)
        model, metrics = trainer.train_model(
            X, y, 
            model_type='random_forest', 
            model_name='facial_recognition'
        )
        
        # Save models
        trainer.save_models('trained_models')
        
        # Print model info
        print("\nModel Information:")
        print(trainer.get_model_info())
        
    except Exception as e:
        logger.error(f"Example failed: {e}")