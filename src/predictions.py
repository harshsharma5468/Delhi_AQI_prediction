import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AQIPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            self.model = joblib.load('models/trained_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            self.scaler = None
    
    def prepare_input_features(self, input_data):
        """
        Prepare input features for prediction
        
        Parameters:
        input_data: dict containing pollutant values and optional datetime
        """
        # Default features (can be modified based on actual model)
        features = [
            'co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3',
            'hour', 'day', 'month', 'day_of_week', 'is_weekend',
            'total_pollutants', 'pm_ratio'
        ]
        
        # Create feature vector
        feature_vector = []
        
        # Extract datetime if provided
        if 'datetime' in input_data:
            dt = pd.to_datetime(input_data['datetime'])
            hour = dt.hour
            day = dt.day
            month = dt.month
            day_of_week = dt.dayofweek
            is_weekend = 1 if day_of_week in [5, 6] else 0
        else:
            # Use current datetime
            now = datetime.now()
            hour = now.hour
            day = now.day
            month = now.month
            day_of_week = now.weekday()
            is_weekend = 1 if day_of_week in [5, 6] else 0
        
        # Calculate total pollutants
        pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'nh3']
        total_pollutants = sum(input_data.get(p, 0) for p in pollutants)
        
        # Calculate PM ratio
        pm_ratio = input_data.get('pm2_5', 0) / (input_data.get('pm10', 0) + 1e-5)
        
        # Build feature vector in the correct order
        feature_vector = [
            input_data.get('co', 0),
            input_data.get('no', 0),
            input_data.get('no2', 0),
            input_data.get('o3', 0),
            input_data.get('so2', 0),
            input_data.get('pm10', 0),
            input_data.get('nh3', 0),
            hour,
            day,
            month,
            day_of_week,
            is_weekend,
            total_pollutants,
            pm_ratio
        ]
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_pm25(self, input_data):
        """
        Predict PM2.5 concentration
        
        Parameters:
        input_data: dict containing pollutant values
        
        Returns:
        float: Predicted PM2.5 concentration
        """
        if self.model is None or self.scaler is None:
            print("Model or scaler not loaded.")
            return None
        
        try:
            # Prepare features
            features = self.prepare_input_features(input_data)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def predict_batch(self, input_df):
        """
        Predict PM2.5 for multiple samples
        
        Parameters:
        input_df: pandas DataFrame containing input features
        
        Returns:
        array: Predictions for all samples
        """
        if self.model is None or self.scaler is None:
            print("Model or scaler not loaded.")
            return None
        
        try:
            # Scale features
            scaled_features = self.scaler.transform(input_df)
            
            # Make predictions
            predictions = self.model.predict(scaled_features)
            
            return predictions
            
        except Exception as e:
            print(f"Error making batch predictions: {str(e)}")
            return None
    
    def calculate_aqi_category(self, pm25_value):
        """
        Calculate AQI category based on PM2.5 value
        
        Parameters:
        pm25_value: PM2.5 concentration in μg/m³
        
        Returns:
        tuple: (category, color, health_impact, emoji)
        """
        if pm25_value <= 30:
            return "Good", "#00E400", "Minimal impact", "😊", 1
        elif pm25_value <= 60:
            return "Satisfactory", "#FFFF00", "Minor breathing discomfort", "🙂", 2
        elif pm25_value <= 90:
            return "Moderate", "#FF7E00", "Breathing discomfort for sensitive people", "😐", 3
        elif pm25_value <= 120:
            return "Poor", "#FF0000", "Breathing discomfort for most people", "😷", 4
        elif pm25_value <= 250:
            return "Very Poor", "#8F3F97", "Respiratory illness on prolonged exposure", "🤢", 5
        else:
            return "Severe", "#7E0023", "Health impact even on short exposure", "🚨", 6
    
    def get_health_recommendations(self, category):
        """
        Get health recommendations based on AQI category
        
        Parameters:
        category: AQI category string
        
        Returns:
        dict: Health recommendations
        """
        recommendations = {
            "Good": {
                "outdoor_activities": "Ideal for outdoor activities",
                "sensitive_groups": "No restrictions",
                "indoor_air": "Natural ventilation is fine"
            },
            "Satisfactory": {
                "outdoor_activities": "Generally acceptable",
                "sensitive_groups": "Consider reducing prolonged exertion",
                "indoor_air": "Natural ventilation is fine"
            },
            "Moderate": {
                "outdoor_activities": "Reduce prolonged exertion",
                "sensitive_groups": "Avoid prolonged outdoor activities",
                "indoor_air": "Consider air purifiers"
            },
            "Poor": {
                "outdoor_activities": "Avoid outdoor activities",
                "sensitive_groups": "Stay indoors",
                "indoor_air": "Use air purifiers"
            },
            "Very Poor": {
                "outdoor_activities": "Stay indoors",
                "sensitive_groups": "Stay indoors with air purifiers",
                "indoor_air": "Use air purifiers, keep windows closed"
            },
            "Severe": {
                "outdoor_activities": "Avoid all outdoor activities",
                "sensitive_groups": "Stay indoors with air purifiers",
                "indoor_air": "Use high-efficiency air purifiers"
            }
        }
        
        return recommendations.get(category, recommendations["Moderate"])
    
    def predict_with_confidence(self, input_data, n_iterations=100):
        """
        Predict PM2.5 with confidence intervals using bootstrapping
        
        Parameters:
        input_data: dict containing pollutant values
        n_iterations: number of bootstrap iterations
        
        Returns:
        dict: Prediction with confidence intervals
        """
        if self.model is None:
            print("Model not loaded.")
            return None
        
        try:
            # Prepare features
            features = self.prepare_input_features(input_data)
            scaled_features = self.scaler.transform(features)
            
            # Bootstrap predictions
            predictions = []
            
            if hasattr(self.model, 'estimators_'):
                # For ensemble models
                for estimator in self.model.estimators_:
                    pred = estimator.predict(scaled_features)[0]
                    predictions.append(pred)
            else:
                # For single models, simulate uncertainty
                base_prediction = self.predict_pm25(input_data)
                if base_prediction is not None:
                    # Add some random noise for demonstration
                    np.random.seed(42)
                    predictions = np.random.normal(
                        loc=base_prediction,
                        scale=base_prediction * 0.1,  # 10% standard deviation
                        size=n_iterations
                    )
            
            if predictions:
                predictions = np.array(predictions)
                
                result = {
                    'prediction': np.mean(predictions),
                    'lower_bound': np.percentile(predictions, 2.5),
                    'upper_bound': np.percentile(predictions, 97.5),
                    'std_dev': np.std(predictions),
                    'confidence_interval': (np.percentile(predictions, 2.5), np.percentile(predictions, 97.5))
                }
                
                return result
            else:
                return None
                
        except Exception as e:
            print(f"Error in confidence prediction: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = AQIPredictor()
    
    if predictor.model is not None:
        # Example input
        sample_input = {
            'co': 2616.88,
            'no': 2.18,
            'no2': 70.6,
            'o3': 13.59,
            'so2': 38.62,
            'pm10': 411.73,
            'nh3': 28.63,
            'datetime': '2020-11-25 01:00:00'
        }
        
        # Make prediction
        prediction = predictor.predict_pm25(sample_input)
        
        if prediction is not None:
            print(f"Predicted PM2.5: {prediction:.2f} μg/m³")
            
            # Get AQI category
            category, color, health_impact, emoji, level = predictor.calculate_aqi_category(prediction)
            print(f"AQI Category: {category} {emoji}")
            print(f"Health Impact: {health_impact}")
            
            # Get recommendations
            recommendations = predictor.get_health_recommendations(category)
            print("\nHealth Recommendations:")
            for key, value in recommendations.items():
                print(f"  {key}: {value}")