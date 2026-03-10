
"""
Model Evaluation and Analysis Module
Provides detailed performance analysis and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import joblib
from tensorflow import keras


class ModelEvaluator:
    """Advanced model evaluation and visualization"""
    
    def __init__(self, model_path, scaler_path, model_type='random_forest'):
        self.model_type = model_type
        
        # Load model
        if model_type == 'neural_network':
            self.model = keras.models.load_model(model_path)
        else:
            self.model = joblib.load(model_path)
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        print(f"✓ Loaded {model_type} model and scaler")
    
    def predict_proba(self, X):
        """Get probability predictions"""
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'neural_network':
            proba = self.model.predict(X_scaled).flatten()
            # Convert to 2-class format
            proba = np.vstack([1-proba, proba]).T
        else:
            proba = self.model.predict_proba(X_scaled)
        
        return proba
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path='models/roc_curve.png'):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✓ ROC curve saved to {save_path}")
        
        return fpr, tpr, thresholds, roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, 
                                    save_path='models/pr_curve.png'):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✓ Precision-Recall curve saved to {save_path}")
        
        return precision, recall, thresholds
    
    def find_optimal_threshold(self, y_true, y_pred_proba, metric='f1'):
        """
        Find optimal classification threshold
        
        Args:
            metric: 'f1', 'youden' (sensitivity+specificity-1), or 'cost'
        """
        from sklearn.metrics import f1_score
        
        thresholds = np.linspace(0, 1, 100)
        scores = []
        
        for thresh in thresholds:
            y_pred = (y_pred_proba[:, 1] >= thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'youden':
                cm = confusion_matrix(y_true, y_pred)
                sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
                specificity = cm[0,0] / (cm[0,0] + cm[0,1])
                score = sensitivity + specificity - 1
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"🎯 Optimal Threshold Analysis ({metric}):")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Score at optimal: {scores[optimal_idx]:.3f}")
        
        # Plot threshold analysis
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, scores, 'b-', linewidth=2)
        plt.axvline(optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal = {optimal_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel(f'{metric.upper()} Score')
        plt.title(f'Threshold Optimization ({metric.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'models/threshold_optimization_{metric}.png', dpi=300)
        print(f"✓ Threshold plot saved")
        
        return optimal_threshold
    
    def analyze_errors(self, X_test, y_test, feature_names=None):
        """Analyze misclassified samples"""
        y_pred_proba = self.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Find errors
        errors = y_pred != y_test
        
        false_positives = (y_pred == 1) & (y_test == 0)
        false_negatives = (y_pred == 0) & (y_test == 1)
        
        print(f"❌ Error Analysis:")
        print(f"  Total errors: {np.sum(errors)}")
        print(f"  False Positives: {np.sum(false_positives)} (Normal classified as Crash)")
        print(f"  False Negatives: {np.sum(false_negatives)} (Crash classified as Normal)")
        
        # Analyze confidence of errors
        fp_confidence = y_pred_proba[false_positives, 1]
        fn_confidence = y_pred_proba[false_negatives, 0]
        
        if len(fp_confidence) > 0:
            print(f"False Positive Confidence: {fp_confidence.mean():.3f} ± {fp_confidence.std():.3f}")
        if len(fn_confidence) > 0:
            print(f"  False Negative Confidence: {fn_confidence.mean():.3f} ± {fn_confidence.std():.3f}")
        
        return {
            'false_positives': np.where(false_positives)[0],
            'false_negatives': np.where(false_negatives)[0],
            'fp_confidence': fp_confidence,
            'fn_confidence': fn_confidence
        }
    
    def test_realtime_latency(self, X_sample, n_iterations=1000):
        """Test inference latency for edge deployment"""
        import time
        
        X_scaled = self.scaler.transform(X_sample.reshape(1, -1))
        
        # Warmup
        for _ in range(10):
            if self.model_type == 'neural_network':
                _ = self.model.predict(X_scaled, verbose=0)
            else:
                _ = self.model.predict(X_scaled)
        
        # Measure latency
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            
            if self.model_type == 'neural_network':
                _ = self.model.predict(X_scaled, verbose=0)
            else:
                _ = self.model.predict(X_scaled)
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        print(f"⚡ Inference Latency Test ({n_iterations} iterations):")
        print(f"  Mean:   {latencies.mean():.2f} ms")
        print(f"  Median: {np.median(latencies):.2f} ms")
        print(f"  Std:    {latencies.std():.2f} ms")
        print(f"  Min:    {latencies.min():.2f} ms")
        print(f"  Max:    {latencies.max():.2f} ms")
        print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
        
        # Check if suitable for real-time (< 100ms target)
        if latencies.mean() < 100:
            print(f"  ✅ Suitable for real-time inference!")
        else:
            print(f"  ⚠️  May need optimization for real-time use")
        
        return latencies
    
    def generate_evaluation_report(self, X_test, y_test, output_dir='models/'):
        """Generate comprehensive evaluation report"""
        print("" + "="*60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        # Predictions
        y_pred_proba = self.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # ROC Curve
        self.plot_roc_curve(y_test, y_pred_proba, f'{output_dir}roc_curve.png')
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve(y_test, y_pred_proba, f'{output_dir}pr_curve.png')
        
        # Optimal threshold
        optimal_thresh = self.find_optimal_threshold(y_test, y_pred_proba, metric='f1')
        
        # Error analysis
        errors = self.analyze_errors(X_test, y_test)
        
        # Latency test
        latencies = self.test_realtime_latency(X_test[0])
        
        print("✅ Evaluation report generated successfully!")
        
        return {
            'optimal_threshold': optimal_thresh,
            'errors': errors,
            'latencies': latencies
        }


def main():
    """Example usage"""
    
    # Load test data
    data = np.load("data/processed/features.npz")
    X = data['X']
    y = data['y']
    feature_names = data['feature_names'].tolist() if 'feature_names' in data else None
    
    # Use last 20% as test set
    test_size = int(0.2 * len(X))
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='models/saved_models/crash_detector_random_forest.pkl',
        scaler_path='models/saved_models/scaler.pkl',
        model_type='random_forest'
    )
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(X_test, y_test)
    
    print("📊 Evaluation Summary:")
    print(f"  Optimal Threshold: {report['optimal_threshold']:.3f}")
    print(f"  Mean Latency: {report['latencies'].mean():.2f} ms")


if __name__ == "__main__":
    main()
