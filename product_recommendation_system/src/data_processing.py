import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor with default settings."""
        pass
    
    def load_data(self, social_path: str, transactions_path: str) -> pd.DataFrame:
        """
        Load and preprocess the social profiles and transactions data.
        
        Args:
            social_path: Path to the social profiles CSV file
            transactions_path: Path to the transactions CSV file
            
        Returns:
            Merged and preprocessed DataFrame
        """
        # Load the data
        social_df = pd.read_csv(social_path)
        transactions_df = pd.read_csv(transactions_path)
        
        print("\n🔍 Social profiles data shape:", social_df.shape)
        print("📊 First few rows of social profiles:")
        print(social_df.head())
        
        print("\n🔍 Transactions data shape:", transactions_df.shape)
        print("📊 First few rows of transactions:")
        print(transactions_df.head())
        
        # Create a mapping from the letter-prefixed IDs to the numeric IDs
        social_df['numeric_id'] = social_df['customer_id_new'].str.extract('A(\d+)').astype(float)
        
        # Convert customer_id_legacy to float for merging
        transactions_df['customer_id_legacy'] = transactions_df['customer_id_legacy'].astype(float)
        
        # Print unique customer IDs for debugging
        print("\n🔢 Numeric IDs in social_df (first 5):", social_df['numeric_id'].head().tolist())
        print("🔢 Customer IDs in transactions_df (first 5):", transactions_df['customer_id_legacy'].head().tolist())
        
        # Merge the dataframes using the numeric ID
        merged_df = pd.merge(
            social_df,
            transactions_df,
            left_on='numeric_id',
            right_on='customer_id_legacy',
            how='left'
        )
        
        # Drop the temporary numeric_id column
        if 'numeric_id' in merged_df.columns:
            merged_df = merged_df.drop('numeric_id', axis=1)
        
        print("\n🔍 Merged data shape:", merged_df.shape)
        print("📊 First few rows of merged data:")
        print(merged_df.head())
        
        # Check for missing values in the merged data
        print("\n🔍 Missing values in merged data:")
        print(merged_df.isnull().sum())
        
        return merged_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features from the raw data.
        
        Args:
            df: Merged DataFrame from load_data
            
        Returns:
            DataFrame with additional engineered features
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Convert purchase_date to datetime if it exists
        if 'purchase_date' in df.columns:
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            
            # Extract time-based features
            df['purchase_year'] = df['purchase_date'].dt.year
            df['purchase_month'] = df['purchase_date'].dt.month
            df['purchase_day'] = df['purchase_date'].dt.day
            df['purchase_dayofweek'] = df['purchase_date'].dt.dayofweek
            df['is_weekend'] = df['purchase_dayofweek'].isin([5, 6]).astype(int)
        
        # Create interaction features
        if 'engagement_score' in df.columns and 'purchase_amount' in df.columns:
            df['engagement_purchase_ratio'] = df['engagement_score'] / (df['purchase_amount'] + 1)  # +1 to avoid division by zero
        
        # Create sentiment score (convert text to numerical)
        if 'review_sentiment' in df.columns:
            sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
            df['sentiment_score'] = df['review_sentiment'].map(sentiment_map)
        
        # One-hot encode categorical columns
        categorical_cols = ['social_media_platform', 'product_category']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare data for modeling by handling missing values and feature selection.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Tuple of (features, target) where features is a DataFrame and target is a Series
        """
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Define numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Handle missing values
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Select features - exclude IDs and dates
        feature_cols = [col for col in df.columns 
                       if col not in ['customer_id_new', 'customer_id_legacy', 'transaction_id', 'purchase_date']
                       and not col.startswith('social_media_platform_')  # Exclude one-hot encoded columns
                       and not col.startswith('product_category_')]
        
        # If we have a target variable, separate it
        target = None
        if 'purchase_amount' in df.columns:
            target = df['purchase_amount']
            feature_cols.remove('purchase_amount')
        
        features = df[feature_cols]
        
        return features, target