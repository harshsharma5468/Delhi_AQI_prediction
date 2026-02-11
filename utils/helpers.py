import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os

class AQIHelper:
    """Helper functions for AQI prediction project"""
    
    @staticmethod
    def calculate_aqi(pm25):
        """
        Calculate AQI from PM2.5 concentration
        
        Parameters:
        pm25: PM2.5 concentration in μg/m³
        
        Returns:
        dict: AQI information
        """
        # AQI breakpoints for PM2.5 (24-hour)
        breakpoints = [
            (0, 12, 0, 50, 'Good', '#00E400'),
            (12.1, 35.4, 51, 100, 'Moderate', '#FFFF00'),
            (35.5, 55.4, 101, 150, 'Unhealthy for Sensitive Groups', '#FF7E00'),
            (55.5, 150.4, 151, 200, 'Unhealthy', '#FF0000'),
            (150.5, 250.4, 201, 300, 'Very Unhealthy', '#8F3F97'),
            (250.5, 500.4, 301, 500, 'Hazardous', '#7E0023')
        ]
        
        for bp_low, bp_high, aqi_low, aqi_high, category, color in breakpoints:
            if bp_low <= pm25 <= bp_high:
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
                aqi = round(aqi)
                return {
                    'aqi': aqi,
                    'category': category,
                    'color': color,
                    'health_concern': AQIHelper.get_health_concern(category)
                }
        
        # If above 500
        return {
            'aqi': 500,
            'category': 'Hazardous+',
            'color': '#7E0023',
            'health_concern': 'Health warnings of emergency conditions'
        }
    
    @staticmethod
    def get_health_concern(category):
        """Get health concern based on AQI category"""
        concerns = {
            'Good': 'Air quality is satisfactory',
            'Moderate': 'Acceptable air quality',
            'Unhealthy for Sensitive Groups': 'Members of sensitive groups may experience health effects',
            'Unhealthy': 'Everyone may begin to experience health effects',
            'Very Unhealthy': 'Health alert: everyone may experience more serious health effects',
            'Hazardous': 'Health warnings of emergency conditions'
        }
        return concerns.get(category, 'Unknown category')
    
    @staticmethod
    def get_recommendations(aqi_category):
        """Get recommendations based on AQI category"""
        recommendations = {
            'Good': {
                'general': 'Perfect day to be active outside.',
                'sensitive': 'Ideal air quality for everyone.'
            },
            'Moderate': {
                'general': 'Unusually sensitive people should consider reducing prolonged outdoor exertion.',
                'sensitive': 'People with respiratory disease should watch for symptoms.'
            },
            'Unhealthy for Sensitive Groups': {
                'general': 'Sensitive groups should reduce prolonged outdoor exertion.',
                'sensitive': 'People with heart or lung disease, older adults, and children should limit outdoor exertion.'
            },
            'Unhealthy': {
                'general': 'Everyone should reduce prolonged outdoor exertion.',
                'sensitive': 'People with heart or lung disease, older adults, and children should avoid outdoor exertion.'
            },
            'Very Unhealthy': {
                'general': 'Everyone should avoid prolonged outdoor exertion.',
                'sensitive': 'People with heart or lung disease, older adults, and children should remain indoors.'
            },
            'Hazardous': {
                'general': 'Everyone should avoid all outdoor activities.',
                'sensitive': 'People with heart or lung disease, older adults, and children should remain indoors and keep activity low.'
            }
        }
        return recommendations.get(aqi_category, recommendations['Moderate'])
    
    @staticmethod
    def generate_sample_data(n_samples=100):
        """Generate sample data for testing"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        data = {
            'date': dates,
            'co': np.random.uniform(1000, 5000, n_samples),
            'no': np.random.uniform(1, 50, n_samples),
            'no2': np.random.uniform(20, 150, n_samples),
            'o3': np.random.uniform(5, 100, n_samples),
            'so2': np.random.uniform(10, 100, n_samples),
            'pm10': np.random.uniform(100, 500, n_samples),
            'nh3': np.random.uniform(5, 50, n_samples),
            'pm2_5': np.random.uniform(50, 400, n_samples)
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def save_results(results, filename='results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4, default=str)
    
    @staticmethod
    def load_results(filename='results.json'):
        """Load results from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def create_directory_structure():
        """Create necessary directories"""
        directories = [
            'data',
            'models',
            'visualizations',
            'results',
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
    
    @staticmethod
    def validate_input_data(input_data):
        """Validate input data for prediction"""
        required_fields = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Check data types
        for field in required_fields:
            if not isinstance(input_data[field], (int, float)):
                return False, f"Field {field} must be numeric"
            
            # Check reasonable ranges
            ranges = {
                'co': (0, 10000),
                'no': (0, 500),
                'no2': (0, 500),
                'o3': (0, 500),
                'so2': (0, 500),
                'pm10': (0, 1000),
                'nh3': (0, 200)
            }
            
            min_val, max_val = ranges.get(field, (0, 1000))
            if not (min_val <= input_data[field] <= max_val):
                return False, f"Field {field} value {input_data[field]} outside reasonable range [{min_val}, {max_val}]"
        
        return True, "Input data validated successfully"
    
    @staticmethod
    def format_prediction_result(prediction, input_data):
        """Format prediction result for display"""
        aqi_info = AQIHelper.calculate_aqi(prediction)
        recommendations = AQIHelper.get_recommendations(aqi_info['category'])
        
        result = {
            'prediction': {
                'pm2_5': round(prediction, 2),
                'unit': 'μg/m³'
            },
            'aqi': {
                'value': aqi_info['aqi'],
                'category': aqi_info['category'],
                'color': aqi_info['color'],
                'health_concern': aqi_info['health_concern']
            },
            'recommendations': recommendations,
            'input_data': input_data,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    @staticmethod
    def get_season(month):
        """Get season based on month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    @staticmethod
    def calculate_statistics(df, column):
        """Calculate statistics for a column"""
        stats = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'q25': df[column].quantile(0.25),
            'q75': df[column].quantile(0.75),
            'count': df[column].count(),
            'missing': df[column].isnull().sum()
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    helper = AQIHelper()
    
    # Create directory structure
    helper.create_directory_structure()
    
    # Generate sample data
    sample_df = helper.generate_sample_data(50)
    print(f"Generated sample data with {len(sample_df)} rows")
    
    # Calculate AQI for a sample PM2.5 value
    pm25_sample = 150
    aqi_info = helper.calculate_aqi(pm25_sample)
    print(f"\nPM2.5: {pm25_sample} μg/m³")
    print(f"AQI: {aqi_info['aqi']}")
    print(f"Category: {aqi_info['category']}")
    
    # Get recommendations
    recommendations = helper.get_recommendations(aqi_info['category'])
    print(f"\nRecommendations: {recommendations['general']}")