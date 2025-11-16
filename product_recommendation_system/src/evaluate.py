import os
import json  # Add this import
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.close()

def generate_report(model, X_test, y_test, label_encoder, output_dir='reports'):
    """Generate evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X_test)
    class_names = label_encoder.classes_
    
    # Generate plots
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Generate metrics
    metrics = model.evaluate(X_test, y_test)
    
    # Save report
    report = {
        'model_type': model.model_type,
        'best_params': model.best_params_,
        'test_metrics': metrics,
        'feature_importance': model.model.feature_importances_.tolist() if hasattr(model.model, 'feature_importances_') else None
    }
    
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    return report