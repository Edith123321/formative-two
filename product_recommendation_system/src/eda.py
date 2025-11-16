import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import DataProcessor

def perform_eda(data_processor, output_dir='reports/figures'):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        data_processor: Initialized DataProcessor instance
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the processed data
    df = data_processor.data.copy()
    
    # 1. Basic Info
    print("=== Dataset Info ===")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    # 2. Missing Values
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/missing_values.png')
    plt.close()
    
    # 3. Distribution of Numerical Features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        # Plot distributions
        fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(15, 5*len(numeric_cols)))
        for i, col in enumerate(numeric_cols):
            # Distribution plot
            sns.histplot(data=df, x=col, kde=True, ax=axes[i,0])
            axes[i,0].set_title(f'Distribution of {col}')
            
            # Boxplot for outlier detection
            sns.boxplot(data=df, x=col, ax=axes[i,1])
            axes[i,1].set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/numerical_distributions.png')
        plt.close()
    
    # 4. Categorical Features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        plt.figure(figsize=(15, 5*len(cat_cols)))
        for i, col in enumerate(cat_cols, 1):
            plt.subplot(len(cat_cols), 1, i)
            sns.countplot(data=df, x=col)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/categorical_distributions.png')
        plt.close()
    
    # 5. Correlation Heatmap (for numerical features)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
        plt.close()
    
    print(f"\nEDA plots saved to {output_dir}/")

if __name__ == "__main__":
    # Initialize data processor
    processor = DataProcessor(data_dir='data/raw')
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = processor.prepare_data()
    
    # Perform EDA on the training data
    perform_eda(processor)