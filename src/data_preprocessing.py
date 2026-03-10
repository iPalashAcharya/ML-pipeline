
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import yaml
from tqdm import tqdm


class SensorDataPreprocessor:
    """Preprocesses raw MPU6050 sensor data for ML training"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.window_size = self.config['data']['window_size']
        self.overlap = self.config['data']['overlap']
        self.sample_rate = self.config['data']['sample_rate']
        
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw CSV data from Arduino/ESP32
        Expected columns: timestamp, ax, ay, az, gx, gy, gz, label
        """
        try:
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label']
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            print(f"✓ Loaded {len(df)} samples from {filepath}")
            return df
            
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers and handle missing values"""
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Handle missing values
        df = df.dropna()
        
        # Remove extreme outliers (beyond physical sensor limits)
        # MPU6050: ±16g for accelerometer, ±2000°/s for gyro
        accel_cols = ['ax', 'ay', 'az']
        gyro_cols = ['gx', 'gy', 'gz']
        
        for col in accel_cols:
            df = df[(df[col] >= -20) & (df[col] <= 20)]  # Allow some margin
        
        for col in gyro_cols:
            df = df[(df[col] >= -2500) & (df[col] <= 2500)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Cleaned data: {len(df)} samples remaining")
        return df
    
    def create_sliding_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows from time-series data
        
        Returns:
            X: windowed sensor data (n_windows, window_size, n_features)
            y: labels for each window
        """
        sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        data = df[sensor_cols].values
        labels = df['label'].values
        
        stride = self.window_size - self.overlap
        n_windows = (len(data) - self.window_size) // stride + 1
        
        X_windows = []
        y_windows = []
        
        for i in tqdm(range(n_windows), desc="Creating windows"):
            start_idx = i * stride
            end_idx = start_idx + self.window_size
            
            window = data[start_idx:end_idx]
            
            # Label is the most common label in the window
            window_labels = labels[start_idx:end_idx]
            window_label = np.bincount(window_labels).argmax()
            
            X_windows.append(window)
            y_windows.append(window_label)
        
        X = np.array(X_windows)
        y = np.array(y_windows)
        
        print(f"✓ Created {len(X)} windows of shape {X.shape}")
        return X, y
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance crash and normal samples using SMOTE or undersampling"""
        from imblearn.over_sampling import SMOTE
        
        # Flatten X for SMOTE (it requires 2D input)
        n_samples, window_size, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Original distribution: {dict(zip(unique, counts))}")
        
        # Apply SMOTE only if imbalance is significant
        ratio = counts.min() / counts.max()
        if ratio < 0.5:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X_flat, y)
            
            # Reshape back
            X_balanced = X_balanced.reshape(-1, window_size, n_features)
            
            unique, counts = np.unique(y_balanced, return_counts=True)
            print(f"Balanced distribution: {dict(zip(unique, counts))}")
            
            return X_balanced, y_balanced
        
        return X, y
    
    def process_pipeline(self, input_dir: str, output_path: str):
        """Complete preprocessing pipeline"""
        
        input_path = Path(input_dir)
        all_dfs = []
        
        # Load all CSV files
        for csv_file in input_path.glob("*.csv"):
            df = self.load_raw_data(csv_file)
            if df is not None:
                df_clean = self.clean_data(df)
                all_dfs.append(df_clean)
        
        # Combine all data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"✓ Combined dataset: {len(combined_df)} samples")
        
        # Create windows
        X, y = self.create_sliding_windows(combined_df)
        
        # Balance dataset
        X_balanced, y_balanced = self.balance_dataset(X, y)
        
        # Save processed data
        np.savez_compressed(
            output_path,
            X=X_balanced,
            y=y_balanced,
            window_size=self.window_size,
            sample_rate=self.sample_rate
        )
        
        print(f"✓ Saved processed data to {output_path}")
        return X_balanced, y_balanced


def main():
    """Example usage"""
    preprocessor = SensorDataPreprocessor()
    
    X, y = preprocessor.process_pipeline(
        input_dir="data/raw/",
        output_path="data/processed/windowed_data.npz"
    )
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()