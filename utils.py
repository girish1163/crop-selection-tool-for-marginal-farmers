"""
Utility functions for crop recommendation system
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Configuration class for the crop recommendation system"""
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.trained_models_dir = os.path.join(self.models_dir, 'trained_models')
        
        # Model parameters
        self.random_state = 42
        self.test_size = 0.2
        self.cv_folds = 5
        
        # Feature columns
        self.soil_features = ['soil_pH', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium']
        self.climate_features = ['temperature', 'rainfall', 'humidity']
        self.market_features = ['market_demand', 'price_per_ton']
        self.yield_features = ['yield_tons_per_hectare']
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.data_dir, self.models_dir, self.trained_models_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_all_features(self):
        """Get all feature columns"""
        return (self.soil_features + self.climate_features + 
                self.market_features + self.yield_features)

def save_results(results, file_path):
    """Save results to JSON file"""
    try:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_results = convert_numpy(results)
        
        with open(file_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def load_results(file_path):
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        print(f"Results loaded from {file_path}")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def log_experiment(experiment_name, parameters, results, metrics):
    """Log experiment details"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
        'parameters': parameters,
        'results': results,
        'metrics': metrics
    }
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Save to log file
    log_file = os.path.join(logs_dir, f'experiment_log_{datetime.now().strftime("%Y%m%d")}.json')
    
    try:
        # Load existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new entry
        logs.append(log_entry)
        
        # Save logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Experiment logged to {log_file}")
        
    except Exception as e:
        print(f"Error logging experiment: {e}")

def validate_input_data(data, required_columns):
    """Validate input data against required columns"""
    missing_columns = set(required_columns) - set(data.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty values
    empty_columns = data.columns[data.isnull().any()].tolist()
    if empty_columns:
        print(f"Warning: Columns with empty values: {empty_columns}")
    
    # Check data types
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found in the data")
    
    print("Data validation passed!")
    return True

def calculate_crop_suitability_score(data, crop_requirements):
    """Calculate suitability score for different crops based on requirements"""
    suitability_scores = {}
    
    for crop, requirements in crop_requirements.items():
        score = 0
        total_weight = 0
        
        for feature, ideal_range in requirements.items():
            if feature in data.columns:
                actual_value = data[feature].mean()
                
                # Calculate how close the actual value is to the ideal range
                if isinstance(ideal_range, tuple) and len(ideal_range) == 2:
                    min_val, max_val = ideal_range
                    if min_val <= actual_value <= max_val:
                        feature_score = 1.0
                    else:
                        # Calculate distance from range
                        if actual_value < min_val:
                            feature_score = max(0, 1 - (min_val - actual_value) / min_val)
                        else:
                            feature_score = max(0, 1 - (actual_value - max_val) / max_val)
                else:
                    # Single ideal value
                    ideal_value = ideal_range
                    feature_score = max(0, 1 - abs(actual_value - ideal_value) / ideal_value)
                
                score += feature_score
                total_weight += 1
        
        if total_weight > 0:
            suitability_scores[crop] = score / total_weight
    
    return suitability_scores

def generate_crop_recommendations(predictions, probabilities, top_n=3):
    """Generate detailed crop recommendations"""
    if predictions is None or probabilities is None:
        return None
    
    recommendations = []
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        # Get top N predictions with probabilities
        top_indices = np.argsort(prob)[-top_n:][::-1]
        
        crop_recommendations = []
        for idx in top_indices:
            crop_recommendations.append({
                'crop': idx,
                'confidence': prob[idx],
                'rank': len(crop_recommendations) + 1
            })
        
        recommendations.append({
            'sample_index': i,
            'predicted_crop': pred,
            'recommendations': crop_recommendations
        })
    
    return recommendations

def create_crop_requirements_dict():
    """Create a dictionary of ideal growing conditions for different crops"""
    return {
        'Wheat': {
            'soil_pH': (6.0, 7.0),
            'soil_nitrogen': (80, 120),
            'soil_phosphorus': (40, 60),
            'soil_potassium': (100, 150),
            'temperature': (15, 25),
            'rainfall': (500, 1000),
            'humidity': (50, 70)
        },
        'Rice': {
            'soil_pH': (5.5, 6.5),
            'soil_nitrogen': (90, 130),
            'soil_phosphorus': (35, 55),
            'soil_potassium': (100, 140),
            'temperature': (20, 30),
            'rainfall': (1000, 2000),
            'humidity': (65, 85)
        },
        'Corn': {
            'soil_pH': (6.0, 7.5),
            'soil_nitrogen': (70, 110),
            'soil_phosphorus': (50, 70),
            'soil_potassium': (130, 170),
            'temperature': (18, 28),
            'rainfall': (600, 1200),
            'humidity': (55, 75)
        },
        'Soybeans': {
            'soil_pH': (6.0, 7.0),
            'soil_nitrogen': (60, 100),
            'soil_phosphorus': (45, 65),
            'soil_potassium': (120, 160),
            'temperature': (20, 30),
            'rainfall': (500, 900),
            'humidity': (60, 80)
        },
        'Cotton': {
            'soil_pH': (5.8, 7.0),
            'soil_nitrogen': (75, 115),
            'soil_phosphorus': (40, 60),
            'soil_potassium': (110, 150),
            'temperature': (22, 32),
            'rainfall': (700, 1300),
            'humidity': (50, 70)
        }
    }

def format_prediction_output(prediction_result, label_encoder):
    """Format prediction result for better readability"""
    if prediction_result is None:
        return "No prediction available"
    
    predicted_crop = prediction_result['predicted_crop'][0]
    confidence = prediction_result['confidence_scores'][0]
    
    # Convert numeric prediction to crop name
    if hasattr(predicted_crop, '__iter__') and not isinstance(predicted_crop, str):
        predicted_crop_name = label_encoder.inverse_transform([predicted_crop])[0]
    else:
        predicted_crop_name = predicted_crop
    
    output = {
        'recommended_crop': predicted_crop_name,
        'confidence_score': float(confidence),
        'recommendation_level': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'
    }
    
    # Add top 3 recommendations if probabilities are available
    if prediction_result['probabilities'] is not None:
        probabilities = prediction_result['probabilities'][0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        top_crops = []
        for idx in top_indices:
            crop_name = label_encoder.inverse_transform([idx])[0]
            prob = probabilities[idx]
            top_crops.append({
                'crop': crop_name,
                'probability': float(prob)
            })
        
        output['top_3_recommendations'] = top_crops
    
    return output

def create_sample_input():
    """Create sample input for testing"""
    return pd.DataFrame({
        'soil_pH': [6.5],
        'soil_nitrogen': [85],
        'soil_phosphorus': [45],
        'soil_potassium': [120],
        'temperature': [25.5],
        'rainfall': [1200],
        'humidity': [65],
        'market_demand': [85],
        'price_per_ton': [2500],
        'yield_tons_per_hectare': [3.2]
    })

def benchmark_models(models_dict, X_test, y_test, label_encoder):
    """Benchmark multiple models and return comprehensive results"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import time
    
    benchmark_results = {}
    
    for model_name, model in models_dict.items():
        print(f"Benchmarking {model_name}...")
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        benchmark_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': prediction_time,
            'predictions': y_pred,
            'model_size': model.__sizeof__()
        }
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(benchmark_results).T
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    print("\n=== Model Benchmark Results ===")
    print(comparison_df.round(4))
    
    return benchmark_results, comparison_df

if __name__ == "__main__":
    # Test utility functions
    config = Config()
    print(f"Data directory: {config.data_dir}")
    print(f"All features: {config.get_all_features()}")
    
    # Test sample input creation
    sample_input = create_sample_input()
    print(f"\nSample input shape: {sample_input.shape}")
    print(f"Sample input columns: {list(sample_input.columns)}")
