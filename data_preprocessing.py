"""
Data preprocessing module for crop recommendation system
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Class for preprocessing crop recommendation data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self, df):
        """Explore and display basic information about the dataset"""
        print("\n=== Data Exploration ===")
        print(f"Dataset Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{df.describe()}")
        
        if 'crop' in df.columns:
            print(f"\nCrop Distribution:\n{df['crop'].value_counts()}")
        
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        print("\n=== Data Cleaning ===")
        
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = self.imputer.fit_transform(df_clean[numeric_columns])
        
        # Remove duplicates
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_shape - df_clean.shape[0]} duplicate rows")
        
        # Handle outliers using IQR method for numeric columns
        for col in numeric_columns:
            if col != 'crop':  # Don't process target column
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers > 0:
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    print(f"Removed {outliers} outliers from {col}")
        
        print(f"Final cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def feature_engineering(self, df):
        """Create additional features from existing data"""
        print("\n=== Feature Engineering ===")
        
        df_fe = df.copy()
        
        # Create soil quality index
        if all(col in df_fe.columns for col in ['soil_nitrogen', 'soil_phosphorus', 'soil_potassium']):
            df_fe['soil_quality_index'] = (
                df_fe['soil_nitrogen'] * 0.4 + 
                df_fe['soil_phosphorus'] * 0.3 + 
                df_fe['soil_potassium'] * 0.3
            )
        
        # Create climate suitability score
        if all(col in df_fe.columns for col in ['temperature', 'rainfall', 'humidity']):
            df_fe['climate_suitability'] = (
                np.where(df_fe['temperature'].between(20, 30), 1, 0.5) *
                np.where(df_fe['rainfall'].between(800, 1500), 1, 0.5) *
                np.where(df_fe['humidity'].between(50, 80), 1, 0.5)
            )
        
        # Create market performance score
        if all(col in df_fe.columns for col in ['market_demand', 'price_per_ton']):
            df_fe['market_score'] = df_fe['market_demand'] * df_fe['price_per_ton'] / 10000
        
        # Create efficiency ratio
        if all(col in df_fe.columns for col in ['yield_tons_per_hectare', 'price_per_ton']):
            df_fe['revenue_per_hectare'] = df_fe['yield_tons_per_hectare'] * df_fe['price_per_ton']
        
        print(f"Added engineered features. New shape: {df_fe.shape}")
        return df_fe
    
    def prepare_features(self, df, target_column='crop'):
        """Prepare features for machine learning"""
        print("\n=== Feature Preparation ===")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        self.target_column = target_column
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        print(f"Features prepared: {len(self.feature_columns)} columns")
        print(f"Target classes: {len(self.label_encoder.classes_)}")
        print(f"Target classes: {list(self.label_encoder.classes_)}")
        
        return X_scaled, y_encoded
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\n=== Data Splitting ===")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path, target_column='crop'):
        """Complete preprocessing pipeline"""
        print("=== Starting Data Preprocessing Pipeline ===")
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        # Explore data
        self.explore_data(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_fe = self.feature_engineering(df_clean)
        
        # Prepare features
        X, y = self.prepare_features(df_fe, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\n=== Preprocessing Complete ===")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'target_classes': self.label_encoder.classes_,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'processed_data': df_fe
        }
    
    def transform_new_data(self, new_data):
        """Transform new data using fitted preprocessing"""
        if self.feature_columns is None or self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call preprocess_pipeline first.")
        
        # Ensure new_data has all required columns
        missing_cols = set(self.feature_columns) - set(new_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in new data: {missing_cols}")
        
        # Select and scale features
        X_new = new_data[self.feature_columns]
        X_new_scaled = self.scaler.transform(X_new)
        
        return pd.DataFrame(X_new_scaled, columns=self.feature_columns)

# Utility functions
def load_sample_data():
    """Load the sample dataset"""
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'data', 'sample_data.csv')
    
    preprocessor = DataPreprocessor()
    return preprocessor.load_data(data_path)

if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess_pipeline('../data/sample_data.csv')
    
    if result:
        print("Preprocessing pipeline completed successfully!")
    else:
        print("Preprocessing pipeline failed!")
