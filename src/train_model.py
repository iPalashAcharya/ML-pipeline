
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)

import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
class CrashDetectionTrainer:
    """Trains and evaluates crash detection models"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_type = self.config['model']['type']
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
        """Load preprocessed feature data"""
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        feature_names = data['feature_names'].tolist() if 'feature_names' in data else None

        print(f"✓ Loaded data: X={X.shape}, y={y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")

        return X, y, feature_names

    def stratified_train_val_test_split(self, X, y, val_size=0.2, test_size=0.2, random_state=42):
      X_train_val, X_test, y_train_val, y_test = train_test_split(
          X, y, test_size=test_size, stratify=y, random_state=random_state
      )
      val_relative_size = val_size / (1 - test_size)
      X_train, X_val, y_train, y_val = train_test_split(
          X_train_val, y_train_val, test_size=val_relative_size,
          stratify=y_train_val, random_state=random_state
      )
      return X_train, X_val, X_test, y_train, y_val, y_test



    def normalize_features(self, X_train, X_val, X_test):
        """Normalize features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def build_random_forest(self) -> RandomForestClassifier:
        """Build Random Forest model"""
        params = self.config['model']['random_forest']

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            class_weight=params['class_weight'],
            random_state=self.config['training']['random_state'],
            n_jobs=-1,
            verbose=1
        )

        return model

    def build_xgboost(self) -> xgb.XGBClassifier:
        """Build XGBoost model"""
        params = self.config['model']['xgboost']

        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            random_state=self.config['training']['random_state'],
            use_label_encoder=False,
            eval_metric='logloss'
        )

        return model

    def build_neural_network(self, input_dim: int) -> keras.Model:
        """Build Neural Network model"""
        params = self.config['model']['neural_network']

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization()
        ])

        # Add hidden layers
        for units in params['layers']:
            model.add(layers.Dense(units, activation=params['activation']))
            model.add(layers.Dropout(params['dropout']))

        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the selected model"""

        print(f"{'='*60}")
        print(f"Training {self.model_type.upper()} model...")
        print(f"{'='*60}")

        if self.model_type == "random_forest":
            self.model = self.build_random_forest()
            self.model.fit(X_train, y_train)

        elif self.model_type == "xgboost":
            self.model = self.build_xgboost()

            eval_set = [(X_train, y_train)]
            if X_val is not None:
                eval_set.append((X_val, y_val))

            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=True
            )

        elif self.model_type == "neural_network":
            self.model = self.build_neural_network(X_train.shape[1])

            params = self.config['model']['neural_network']

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )

            return history

        print("✓ Training completed!")

    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluate model on test set"""

        print(f"{'='*60}")
        print(f"Evaluating model on test set...")
        print(f"{'='*60}")

        # Predictions
        if self.model_type == "neural_network":
            y_pred_proba = self.model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba_all = self.model.predict_proba(X_test)
            if y_pred_proba_all.shape[1] == 1:
                # Only one class present, use first column
                y_pred_proba = y_pred_proba_all[:, 0]
            else:
                y_pred_proba = y_pred_proba_all[:, 1]

        unique_classes = np.unique(y_test)

        if len(unique_classes) == 1:
            labels = unique_classes.tolist()
            target_names = [str(x) for x in labels]
            print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))
        else:
          print(classification_report(y_test, y_pred, target_names=['Normal', 'Crash'], zero_division=0))

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Print results
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Crash']))

        print("📈 Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC AUC:   {roc_auc:.4f}")

        print("🎯 Confusion Matrix:")
        print(cm)
        print(f"  True Negatives:  {cm[0,0]}")
        print(f"  False Positives: {cm[0,1]}")
        print(f"  False Negatives: {cm[1,0]}")
        print(f"  True Positives:  {cm[1,1]}")

        # Calculate specific metrics for crash detection
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # Recall for crash class
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # Recall for normal class

        print(f"🚨 Crash Detection Specific:")
        print(f"  Sensitivity (Crash Detection Rate): {sensitivity:.4f}")
        print(f"  Specificity (Normal Ride Accuracy):  {specificity:.4f}")
        print(f"  False Alarm Rate: {1-specificity:.4f}")

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

        return metrics

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"{'='*60}")
        print(f"Performing {cv}-fold cross-validation...")
        print(f"{'='*60}")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        if self.model_type == "random_forest":
            model = self.build_random_forest()
        elif self.model_type == "xgboost":
            model = self.build_xgboost()
        else:
            print("Cross-validation not supported for neural networks")
            return None

        scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

        print(f"Cross-validation F1 scores: {scores}")
        print(f"Mean F1: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores

    def get_feature_importance(self, feature_names=None):
        """Get feature importance for tree-based models"""

        if self.model_type in ["random_forest", "xgboost"]:
            importances = self.model.feature_importances_

            if feature_names:
                feature_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                print("🔍 Top 15 Most Important Features:")
                print(feature_imp.head(15).to_string(index=False))

                return feature_imp

            return importances
        else:
            print("Feature importance only available for tree-based models")
            return None

    def plot_training_history(self, history):
        """Plot training history for neural networks"""
        if history is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300)
        print("✓ Training history plot saved to models/training_history.png")

    def plot_confusion_matrix(self, cm, save_path='models/confusion_matrix.png'):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Crash'],
            yticklabels=['Normal', 'Crash']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✓ Confusion matrix plot saved to {save_path}")

    def save_model(self, save_dir='models/saved_models/'):
        """Save trained model and scaler"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.model_type == "neural_network":
            model_path = f"{save_dir}crash_detector_nn.h5"
            self.model.save(model_path)
        else:
            model_path = f"{save_dir}crash_detector_{self.model_type}.pkl"
            joblib.dump(self.model, model_path)

        # Save scaler
        scaler_path = f"{save_dir}scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        print(f"✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")

    def load_model(self, model_path, scaler_path):
        """Load trained model and scaler"""
        if self.model_type == "neural_network":
            self.model = keras.models.load_model(model_path)
        else:
            self.model = joblib.load(model_path)

        self.scaler = joblib.load(scaler_path)
        print(f"✓ Model loaded from {model_path}")


def main():
    """Complete training pipeline"""

    # Initialize trainer
    trainer = CrashDetectionTrainer()

    # Load data
    X, y, feature_names = trainer.load_data("data/processed/features.npz")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.stratified_train_val_test_split(X, y)


    # Normalize features
    X_train, X_val, X_test = trainer.normalize_features(X_train, X_val, X_test)

    # Cross-validation (optional, comment out if not needed)
    # trainer.cross_validate(X_train, y_train, cv=5)

    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Plot training history (for neural networks)
    if history:
        trainer.plot_training_history(history)

    # Evaluate on test set
    metrics = trainer.evaluate(X_test, y_test)

    # Plot confusion matrix
    trainer.plot_confusion_matrix(metrics['confusion_matrix'])

    # Feature importance (for tree-based models)
    if trainer.model_type in ["random_forest", "xgboost"]:
        trainer.get_feature_importance(feature_names)

    # Save model
    trainer.save_model()

    print("" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
