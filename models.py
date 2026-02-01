"""
Machine Learning Models for Crop Recommendation System
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# For complete presentation system
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class CropRecommendationModels:
    """Class containing various ML models for crop recommendation"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_scores = {}
        
    def initialize_models(self):
        """Initialize all models with default parameters"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'SVM': SVC(
                random_state=42,
                probability=True
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=1000
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_single_model(self, model_name, X_train, y_train):
        """Train a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all models and evaluate their performance"""
        print("=== Training All Models ===")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                self.model_scores[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"{model_name}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Find best model
        if self.model_scores:
            best_model_name = max(self.model_scores, key=lambda x: self.model_scores[x]['cv_mean'])
            self.best_model = trained_models[best_model_name]
            print(f"\nBest model: {best_model_name} with CV accuracy: {self.model_scores[best_model_name]['cv_mean']:.4f}")
        
        return trained_models
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate a single model"""
        print(f"\n=== Evaluating {model_name} ===")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def evaluate_all_models(self, X_test, y_test, label_encoder):
        """Evaluate all trained models"""
        print("\n=== Evaluating All Models ===")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                try:
                    result = self.evaluate_model(model, X_test, y_test, model_name)
                    evaluation_results[model_name] = result
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
        
        return evaluation_results
    
    def create_ensemble_model(self, X_train, y_train, top_n=3):
        """Create an ensemble model from top performing models"""
        print(f"\n=== Creating Ensemble Model (Top {top_n}) ===")
        
        if not self.model_scores:
            print("No model scores available. Train models first.")
            return None
        
        # Get top N models
        top_models = sorted(self.model_scores.items(), 
                          key=lambda x: x[1]['cv_mean'], 
                          reverse=True)[:top_n]
        
        print(f"Top models for ensemble: {[model[0] for model in top_models]}")
        
        # Create voting classifier
        estimators = []
        for model_name, _ in top_models:
            if model_name in self.models:
                estimators.append((model_name.lower().replace(' ', '_'), self.models[model_name]))
        
        if len(estimators) < 2:
            print("Not enough models for ensemble. Using single best model.")
            return self.best_model
        
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Ensemble CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return ensemble
    
    def hyperparameter_tuning(self, model_name, X_train, y_train):
        """Perform hyperparameter tuning for a specific model"""
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def predict_crop(self, model, X_new, label_encoder):
        """Make predictions for new data"""
        try:
            # Make prediction
            prediction = model.predict(X_new)
            prediction_proba = None
            
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(X_new)
            
            # Convert back to original labels
            predicted_crop = label_encoder.inverse_transform(prediction)
            
            # Get confidence scores
            confidence_scores = None
            if prediction_proba is not None:
                confidence_scores = np.max(prediction_proba, axis=1)
            
            return {
                'predicted_crop': predicted_crop,
                'predicted_class': prediction,
                'confidence_scores': confidence_scores,
                'probabilities': prediction_proba
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            return None
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_model(self, model, file_path):
        """Save trained model to file"""
        try:
            joblib.dump(model, file_path)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, file_path):
        """Load trained model from file"""
        try:
            model = joblib.load(file_path)
            print(f"Model loaded from {file_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

class ModelComparator:
    """Class for comparing different models"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_models(self, models_dict, X_test, y_test, label_encoder):
        """Compare multiple models"""
        print("\n=== Model Comparison ===")
        
        comparison_data = []
        
        for model_name, model in models_dict.items():
            try:
                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Get cross-validation scores if available
                cv_score = cross_val_score(model, X_test, y_test, cv=3, scoring='accuracy').mean()
                
                comparison_data.append({
                    'Model': model_name,
                    'Test Accuracy': accuracy,
                    'CV Score': cv_score,
                    'Difference': abs(accuracy - cv_score)
                })
                
            except Exception as e:
                print(f"Error comparing {model_name}: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        print("\nModel Comparison Results:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        self.comparison_results = comparison_df
        return comparison_df

# Utility functions
def create_sample_prediction():
    """Create a sample prediction for testing"""
    from .data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess_pipeline('../data/sample_data.csv')
    
    if result:
        # Initialize and train models
        models = CropRecommendationModels()
        models.initialize_models()
        trained_models = models.train_all_models(result['X_train'], result['y_train'])
        
        # Create sample data for prediction
        sample_data = pd.DataFrame({
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
        
        # Make prediction
        if models.best_model:
            prediction = models.predict_crop(
                models.best_model, 
                sample_data, 
                result['label_encoder']
            )
            
            print(f"\nSample Prediction:")
            print(f"Predicted Crop: {prediction['predicted_crop'][0]}")
            print(f"Confidence: {prediction['confidence_scores'][0]:.4f}")
            
            return prediction
    
    return None

def run_presentation_system():
    """
    🌾 COMPLETE PRESENTATION SYSTEM - Run this function to show everything to jury/faculty
    Your friend can simply run: python models.py
    """
    print("🌾 === CROP RECOMMENDATION SYSTEM FOR PRESENTATION ===")
    print("This will show ML models, visualizations, and recommendations!")
    print("=" * 70)
    
    try:
        # Import the complete system
        from models_complete import CompleteCropRecommendationSystem
        
        # Create and run the complete system
        system = CompleteCropRecommendationSystem()
        system.run_complete_analysis()
        
    except ImportError:
        print("⚠️  Please run models_complete.py for the full presentation system")
        print("   Or install required packages: pip install matplotlib seaborn")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Running basic demonstration...")
        
        # Fallback to basic demonstration
        create_sample_prediction()

if __name__ == "__main__":
    # Run the complete presentation system
    run_presentation_system()
