import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, data_path='data/delhi_aqi.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        if self.df is None:
            print("Data not loaded. Call load_data() first.")
            return False
        
        # Convert date to datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Extract time features
        self.df['hour'] = self.df['date'].dt.hour
        self.df['day'] = self.df['date'].dt.day
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        # Feature engineering
        self.df['total_pollutants'] = self.df[['co', 'no', 'no2', 'o3', 'so2', 'nh3']].sum(axis=1)
        self.df['pm_ratio'] = self.df['pm2_5'] / (self.df['pm10'] + 1e-5)  # Avoid division by zero
        
        # Handle missing values (if any)
        self.df = self.df.fillna(self.df.median())
        
        print("Data preprocessing completed.")
        return True
    
    def prepare_features(self, target='pm2_5'):
        """Prepare features for modeling"""
        if self.df is None:
            print("Data not processed. Call preprocess_data() first.")
            return False
        
        # Define features
        features = [
            'co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3',
            'hour', 'day', 'month', 'day_of_week', 'is_weekend',
            'total_pollutants', 'pm_ratio'
        ]
        
        # Ensure all features exist
        features = [f for f in features if f in self.df.columns]
        
        X = self.df[features]
        y = self.df[target]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Features prepared. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        return True
    
    def get_processed_data(self):
        """Get processed data for modeling"""
        return {
            'X_train': self.X_train_scaled,
            'X_test': self.X_test_scaled,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'features': self.X_train.columns.tolist(),
            'scaler': self.scaler
        }
    
    def get_data_summary(self):
        """Get summary statistics of the data"""
        if self.df is None:
            return None
        
        summary = {
            'total_samples': len(self.df),
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            },
            'pollutants': {
                'pm2_5': {
                    'mean': self.df['pm2_5'].mean(),
                    'std': self.df['pm2_5'].std(),
                    'min': self.df['pm2_5'].min(),
                    'max': self.df['pm2_5'].max()
                },
                'pm10': {
                    'mean': self.df['pm10'].mean(),
                    'std': self.df['pm10'].std(),
                    'min': self.df['pm10'].min(),
                    'max': self.df['pm10'].max()
                },
                'co': {
                    'mean': self.df['co'].mean(),
                    'std': self.df['co'].std(),
                    'min': self.df['co'].min(),
                    'max': self.df['co'].max()
                }
            },
            'missing_values': self.df.isnull().sum().to_dict(),
            'correlations': self.df[['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 'nh3']].corr().to_dict()
        }
        
        return summary
    
    def save_processed_data(self, path='data/processed_data.pkl'):
        """Save processed data"""
        import pickle
        data_to_save = {
            'df': self.df,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'X_train_scaled': self.X_train_scaled,
            'X_test_scaled': self.X_test_scaled
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Processed data saved to {path}")

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    if processor.load_data():
        processor.preprocess_data()
        processor.prepare_features()
        
        summary = processor.get_data_summary()
        print(f"Data Summary: {summary['total_samples']} samples")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        processor.save_processed_data()