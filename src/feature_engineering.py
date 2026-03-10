
class FeatureExtractor:
    """Extracts features from windowed sensor data"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['data']['sample_rate']
        self.feature_config = self.config['features']
    
    def time_domain_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain statistical features
        
        Args:
            window: (window_size, n_channels) array
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        for i, axis in enumerate(['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
            signal_data = window[:, i]
            
            # Basic statistics
            features[f'{axis}_mean'] = np.mean(signal_data)
            features[f'{axis}_std'] = np.std(signal_data)
            features[f'{axis}_min'] = np.min(signal_data)
            features[f'{axis}_max'] = np.max(signal_data)
            
            # RMS (Root Mean Square)
            features[f'{axis}_rms'] = np.sqrt(np.mean(signal_data**2))
            
            # Peak-to-peak
            features[f'{axis}_ptp'] = np.ptp(signal_data)
            
            # Zero crossing rate
            features[f'{axis}_zcr'] = np.sum(np.diff(np.sign(signal_data)) != 0) / len(signal_data)
            
            # Skewness and Kurtosis (shape of distribution)
            features[f'{axis}_skewness'] = stats.skew(signal_data)
            features[f'{axis}_kurtosis'] = stats.kurtosis(signal_data)
            
            # Energy
            features[f'{axis}_energy'] = np.sum(signal_data**2)
            
            # Percentiles
            features[f'{axis}_p25'] = np.percentile(signal_data, 25)
            features[f'{axis}_p75'] = np.percentile(signal_data, 75)
            features[f'{axis}_iqr'] = features[f'{axis}_p75'] - features[f'{axis}_p25']
        
        return features
    
    def frequency_domain_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT
        
        Args:
            window: (window_size, n_channels) array
        
        Returns:
            Dictionary of frequency features
        """
        features = {}
        n_fft = len(window)
        freq = np.fft.fftfreq(n_fft, 1/self.sample_rate)
        
        for i, axis in enumerate(['ax', 'ay', 'az', 'gx', 'gy', 'gz']):
            signal_data = window[:, i]
            
            # Compute FFT
            fft_vals = fft(signal_data)
            fft_magnitude = np.abs(fft_vals[:n_fft//2])
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(fft_magnitude)
            features[f'{axis}_dominant_freq'] = freq[dominant_freq_idx]
            
            # Spectral energy
            features[f'{axis}_spectral_energy'] = np.sum(fft_magnitude**2)
            
            # Spectral centroid
            features[f'{axis}_spectral_centroid'] = np.sum(freq[:n_fft//2] * fft_magnitude) / np.sum(fft_magnitude)
            
            # Spectral entropy
            psd = fft_magnitude / np.sum(fft_magnitude)
            psd = psd[psd > 0]  # Remove zeros for log
            features[f'{axis}_spectral_entropy'] = -np.sum(psd * np.log2(psd))
        
        return features
    
    def derived_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Extract derived/composite features
        
        Args:
            window: (window_size, 6) array [ax, ay, az, gx, gy, gz]
        
        Returns:
            Dictionary of derived features
        """
        features = {}
        
        # Acceleration magnitude (resultant vector)
        acc_mag = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
        features['acc_magnitude_mean'] = np.mean(acc_mag)
        features['acc_magnitude_std'] = np.std(acc_mag)
        features['acc_magnitude_max'] = np.max(acc_mag)
        features['acc_magnitude_min'] = np.min(acc_mag)
        
        # Gyroscope magnitude
        gyro_mag = np.sqrt(window[:, 3]**2 + window[:, 4]**2 + window[:, 5]**2)
        features['gyro_magnitude_mean'] = np.mean(gyro_mag)
        features['gyro_magnitude_std'] = np.std(gyro_mag)
        features['gyro_magnitude_max'] = np.max(gyro_mag)
        
        # Jerk (rate of change of acceleration) - critical for crash detection
        jerk_x = np.diff(window[:, 0])
        jerk_y = np.diff(window[:, 1])
        jerk_z = np.diff(window[:, 2])
        jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
        
        features['jerk_magnitude_mean'] = np.mean(jerk_mag)
        features['jerk_magnitude_max'] = np.max(jerk_mag)
        features['jerk_magnitude_std'] = np.std(jerk_mag)
        
        # Signal Magnitude Area (SMA) - total movement intensity
        sma = (np.sum(np.abs(window[:, 0])) + 
               np.sum(np.abs(window[:, 1])) + 
               np.sum(np.abs(window[:, 2]))) / len(window)
        features['sma'] = sma
        
        # Correlation between axes (crash events often show high correlation)
        features['corr_ax_ay'] = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
        features['corr_ax_az'] = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
        features['corr_ay_az'] = np.corrcoef(window[:, 1], window[:, 2])[0, 1]
        
        return features
    
    def extract_all_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract all configured features from a single window
        
        Args:
            window: (window_size, 6) array
        
        Returns:
            Feature vector as numpy array
        """
        all_features = {}
        
        # Time-domain
        all_features.update(self.time_domain_features(window))
        
        # Frequency-domain (if enabled)
        if self.feature_config['frequency_domain']['enabled']:
            all_features.update(self.frequency_domain_features(window))
        
        # Derived features
        all_features.update(self.derived_features(window))
        
        # Convert to array (maintain consistent order)
        feature_vector = np.array(list(all_features.values()))
        return feature_vector
    
    def extract_features_batch(self, X_windows: np.ndarray) -> np.ndarray:
        """
        Extract features from multiple windows
        
        Args:
            X_windows: (n_samples, window_size, n_channels) array
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        n_samples = X_windows.shape[0]
        
        # Extract features from first window to get feature count
        first_features = self.extract_all_features(X_windows[0])
        n_features = len(first_features)
        
        # Initialize feature matrix
        X_features = np.zeros((n_samples, n_features))
        
        # Extract features for all windows
        for i in tqdm(range(n_samples), desc="Extracting features"):
            X_features[i] = self.extract_all_features(X_windows[i])
        
        print(f"✓ Extracted {n_features} features from {n_samples} windows")
        return X_features
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        dummy_window = np.random.randn(100, 6)
        
        all_features = {}
        all_features.update(self.time_domain_features(dummy_window))
        
        if self.feature_config['frequency_domain']['enabled']:
            all_features.update(self.frequency_domain_features(dummy_window))
        
        all_features.update(self.derived_features(dummy_window))
        
        return list(all_features.keys())


def main():
    """Example usage"""
    # Load processed windows
    data = np.load("data/processed/windowed_data.npz")
    X_windows = data['X']
    y = data['y']
    
    # Extract features
    extractor = FeatureExtractor()
    X_features = extractor.extract_features_batch(X_windows)
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"Feature names ({len(feature_names)}):")
    for i, name in enumerate(feature_names[:10]):  # Show first 10
        print(f"  {i+1}. {name}")
    print("  ...")
    
    # Save features
    np.savez_compressed(
        "data/processed/features.npz",
        X=X_features,
        y=y,
        feature_names=feature_names
    )
    print(f"✓ Saved features to data/processed/features.npz")


if __name__ == "__main__":
    main()
