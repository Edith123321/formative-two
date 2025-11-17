import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
from typing import Tuple, Dict, Union, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        """Initialize the ModelTrainer with a random state for reproducibility."""
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.feature_importances: Dict[str, Dict[str, float]] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.expression_encoder: Optional[LabelEncoder] = None
        self.member_encoder: Optional[LabelEncoder] = None
    
    def prepare_facial_data(self, image_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare image features for facial recognition model training.
        
        Args:
            image_features: DataFrame containing image features
            
        Returns:
            Tuple of (features, target) for model training
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = image_features.copy()
        
        # Encode the target variable (expression) as numerical values
        self.expression_encoder = LabelEncoder()
        df['expression_encoded'] = self.expression_encoder.fit_transform(df['expression'])
        
        # Get features and target
        X = df.select_dtypes(include=[np.number]).drop(['expression_encoded'], axis=1)
        y = df['expression_encoded']
        
        return X, y
    
    def prepare_audio_data(self, audio_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare audio features for voice verification model training.
        
        Args:
            audio_features: DataFrame containing audio features
            
        Returns:
            Tuple of (features, target) for model training
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = audio_features.copy()
        
        # Encode the member_id as numerical values
        self.member_encoder = LabelEncoder()
        df['member_encoded'] = self.member_encoder.fit_transform(df['member_id'])
        
        # Get features and target
        X = df.select_dtypes(include=[np.number]).drop(['member_encoded'], axis=1)
        y = df['member_encoded']
        
        return X, y
    
    def prepare_product_data(self, merged_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare product recommendation data for model training with detailed debugging.
        
        Args:
            merged_data: DataFrame containing merged customer and transaction data
            
        Returns:
            Tuple of (features, target) for model training
            
        Raises:
            ValueError: If required columns are missing or no valid samples remain
        """
        # Make a copy to avoid modifying the original data
        df = merged_data.copy()
        
        # Debug: Print available columns and first few rows
        print("\n🔍 Available columns in merged data:")
        print(df.columns.tolist())
        print("\n📊 First few rows of data:")
        print(df.head())
        
        # Ensure we have the required columns
        if 'purchase_amount' not in df.columns:
            raise ValueError("'purchase_amount' column is required but not found in the data")
        
        # Check for missing values in the target
        missing_target = df['purchase_amount'].isna().sum()
        if missing_target > 0:
            print(f"\n⚠️  Warning: Found {missing_target} missing values in 'purchase_amount'")
        
        # Get numeric columns (excluding the target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'purchase_amount' in numeric_cols:
            numeric_cols.remove('purchase_amount')
        
        print(f"\n🔢 Numeric features available: {numeric_cols}")
        
        # If no numeric features, try to convert some common columns
        if not numeric_cols:
            print("\n⚠️  No numeric features found. Attempting to convert potential numeric columns...")
            potential_numeric = ['customer_rating', 'engagement_score', 'purchase_interest_score']
            for col in potential_numeric:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        numeric_cols.append(col)
                        print(f"✅ Converted '{col}' to numeric")
                    except Exception as e:
                        print(f"⚠️  Could not convert '{col}' to numeric: {e}")
        
        # If still no numeric features, try to use one-hot encoded columns
        if not numeric_cols:
            print("\n⚠️  Still no numeric features. Trying to use one-hot encoded columns...")
            potential_binary = [col for col in df.columns if col.startswith(('is_', 'has_', 'social_media_platform_'))]
            if potential_binary:
                print(f"Using binary columns as features: {potential_binary}")
                numeric_cols = potential_binary
        
        # If we have the target but no features, create some basic features
        if not numeric_cols and 'purchase_amount' in df.columns:
            print("\n⚠️  No suitable features found. Creating basic features...")
            # Add a constant feature (intercept)
            df['constant'] = 1
            numeric_cols = ['constant']
        
        if not numeric_cols:
            raise ValueError("No valid features available for model training")
        
        # Prepare features and target
        X = df[numeric_cols].copy()
        y = df['purchase_amount']
        
        # Drop rows where target is missing
        valid_idx = y.notna()
        X = X[valid_idx].copy()
        y = y[valid_idx]
        
        if len(X) == 0:
            print("\n❌ Error: No valid samples remaining after removing missing values")
            print("Missing values by column:")
            print(df.isnull().sum())
            print("\nData types:")
            print(df.dtypes)
            print("\nTarget value counts:")
            print(df['purchase_amount'].value_counts(dropna=False))
            raise ValueError("No valid samples remaining after removing missing values")
        
        print(f"\n✅ Successfully prepared {len(X)} samples with {len(numeric_cols)} features")
        print(f"Features: {numeric_cols}")
        print(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest', 
                   task: str = 'classification') -> Dict[str, float]:
        """
        Train a machine learning model.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to train ('random_forest' or 'xgboost')
            task: Type of task ('classification' or 'regression')
            
        Returns:
            Dictionary of model metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Initialize model
        if model_type == 'random_forest':
            if task == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        elif model_type == 'xgboost':
            if task == 'classification':
                # For XGBoost classification, we need to specify the number of classes
                num_class = len(np.unique(y))
                model = xgb.XGBClassifier(
                    objective='multi:softmax',
                    num_class=num_class,
                    random_state=self.random_state
                )
            else:
                model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        if task == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            # Convert numerical predictions back to original labels if we have an encoder
            if self.expression_encoder is not None:
                y_test_labels = self.expression_encoder.inverse_transform(y_test)
                y_pred_labels = self.expression_encoder.inverse_transform(y_pred.astype(int))
                metrics['classification_report'] = classification_report(
                    y_test_labels, y_pred_labels, output_dict=True
                )
            elif self.member_encoder is not None:
                y_test_labels = self.member_encoder.inverse_transform(y_test)
                y_pred_labels = self.member_encoder.inverse_transform(y_pred.astype(int))
                metrics['classification_report'] = classification_report(
                    y_test_labels, y_pred_labels, output_dict=True
                )
            else:
                metrics['classification_report'] = classification_report(
                    y_test, y_pred, output_dict=True
                )
        else:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Store model and metrics
        self.models[task] = model
        self.metrics[task] = metrics
        
        # Store feature importances if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importances[task] = dict(zip(X.columns, model.feature_importances_))
        
        return metrics
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save the models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for task, model in self.models.items():
            model_path = os.path.join(output_dir, f'{task}_model.joblib')
            joblib.dump(model, model_path)
            
            # Save feature importances if available
            if task in self.feature_importances:
                importances = pd.DataFrame(
                    self.feature_importances[task].items(),
                    columns=['feature', 'importance']
                ).sort_values('importance', ascending=False)
                
                importances.to_csv(
                    os.path.join(output_dir, f'{task}_feature_importances.csv'),
                    index=False
                )
            
            # Save metrics
            if task in self.metrics:
                metrics_df = pd.DataFrame([self.metrics[task]])
                metrics_df.to_csv(
                    os.path.join(output_dir, f'{task}_metrics.csv'),
                    index=False
                )
    
    def load_model(self, model_path: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)