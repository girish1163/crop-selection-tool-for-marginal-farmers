"""
Data loader module for handling different data formats
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Class for loading and handling different data formats"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.txt']
    
    def load_data(self, file_path):
        """Load data from various file formats"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                return self._load_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return self._load_excel(file_path)
            elif file_extension == '.json':
                return self._load_json(file_path)
            elif file_extension == '.txt':
                return self._load_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    
    def _load_csv(self, file_path):
        """Load CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with errors='ignore'
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            print("Loaded CSV with UTF-8 encoding (ignoring errors)")
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load CSV: {e}")
    
    def _load_excel(self, file_path):
        """Load Excel file"""
        try:
            # Try to read the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            print(f"Successfully loaded Excel file")
            return df
        except Exception as e:
            raise Exception(f"Failed to load Excel file: {e}")
    
    def _load_json(self, file_path):
        """Load JSON file"""
        try:
            df = pd.read_json(file_path)
            print(f"Successfully loaded JSON file")
            return df
        except Exception as e:
            raise Exception(f"Failed to load JSON file: {e}")
    
    def _load_txt(self, file_path):
        """Load text file (assuming tab or space delimited)"""
        try:
            # Try different delimiters
            delimiters = ['\t', ',', ';', ' ', '|']
            
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    if df.shape[1] > 1:  # If we got multiple columns
                        print(f"Successfully loaded text file with '{delimiter}' delimiter")
                        return df
                except:
                    continue
            
            # If no delimiter works, try to read as single column
            df = pd.read_csv(file_path, delimiter=None, header=None)
            print("Loaded text file as single column")
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load text file: {e}")
    
    def auto_detect_columns(self, df):
        """Automatically detect and rename columns based on data patterns"""
        if df is None or df.empty:
            return df
        
        # If columns are already named properly, return as is
        if self._has_proper_columns(df):
            return df
        
        # Try to detect column types from data patterns
        new_columns = []
        
        for i, col in enumerate(df.columns):
            if isinstance(col, str) and not col.startswith('Unnamed'):
                new_columns.append(col)
                continue
            
            # Sample data from this column
            sample_data = df[col].dropna().head(10)
            
            if len(sample_data) == 0:
                new_columns.append(f'column_{i}')
                continue
            
            # Try to detect the type of data
            if self._is_ph_data(sample_data):
                new_columns.append('soil_pH')
            elif self._is_temperature_data(sample_data):
                new_columns.append('temperature')
            elif self._is_rainfall_data(sample_data):
                new_columns.append('rainfall')
            elif self._is_humidity_data(sample_data):
                new_columns.append('humidity')
            elif self._is_nitrogen_data(sample_data):
                new_columns.append('soil_nitrogen')
            elif self._is_phosphorus_data(sample_data):
                new_columns.append('soil_phosphorus')
            elif self._is_potassium_data(sample_data):
                new_columns.append('soil_potassium')
            elif self._is_market_demand_data(sample_data):
                new_columns.append('market_demand')
            elif self._is_price_data(sample_data):
                new_columns.append('price_per_ton')
            elif self._is_yield_data(sample_data):
                new_columns.append('yield_tons_per_hectare')
            elif self._is_crop_data(sample_data):
                new_columns.append('crop')
            else:
                new_columns.append(f'feature_{i}')
        
        df.columns = new_columns
        return df
    
    def _has_proper_columns(self, df):
        """Check if DataFrame has properly named columns"""
        expected_keywords = [
            'ph', 'nitrogen', 'phosphorus', 'potassium', 
            'temperature', 'rainfall', 'humidity',
            'market', 'price', 'yield', 'crop'
        ]
        
        for col in df.columns:
            if isinstance(col, str) and any(keyword in col.lower() for keyword in expected_keywords):
                return True
        
        return False
    
    def _is_ph_data(self, series):
        """Check if series represents pH data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return (numeric_data.between(0, 14)).all() and numeric_data.mean() > 4 and numeric_data.mean() < 9
        except:
            return False
    
    def _is_temperature_data(self, series):
        """Check if series represents temperature data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return (numeric_data.between(-50, 60)).all() and numeric_data.mean() > 0 and numeric_data.mean() < 50
        except:
            return False
    
    def _is_rainfall_data(self, series):
        """Check if series represents rainfall data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return numeric_data.min() >= 0 and numeric_data.max() <= 5000 and numeric_data.mean() > 100
        except:
            return False
    
    def _is_humidity_data(self, series):
        """Check if series represents humidity data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return (numeric_data.between(0, 100)).all() and numeric_data.mean() > 20 and numeric_data.mean() < 90
        except:
            return False
    
    def _is_nitrogen_data(self, series):
        """Check if series represents nitrogen data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return numeric_data.min() >= 0 and numeric_data.max() <= 300 and numeric_data.mean() > 20
        except:
            return False
    
    def _is_phosphorus_data(self, series):
        """Check if series represents phosphorus data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return numeric_data.min() >= 0 and numeric_data.max() <= 200 and numeric_data.mean() > 10
        except:
            return False
    
    def _is_potassium_data(self, series):
        """Check if series represents potassium data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return numeric_data.min() >= 0 and numeric_data.max() <= 400 and numeric_data.mean() > 50
        except:
            return False
    
    def _is_market_demand_data(self, series):
        """Check if series represents market demand data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return (numeric_data.between(0, 100)).all() and numeric_data.mean() > 30
        except:
            return False
    
    def _is_price_data(self, series):
        """Check if series represents price data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return numeric_data.min() > 0 and numeric_data.max() <= 10000 and numeric_data.mean() > 500
        except:
            return False
    
    def _is_yield_data(self, series):
        """Check if series represents yield data"""
        try:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) == 0:
                return False
            return numeric_data.min() > 0 and numeric_data.max() <= 20 and numeric_data.mean() < 10
        except:
            return False
    
    def _is_crop_data(self, series):
        """Check if series represents crop data"""
        try:
            # Check if data contains common crop names
            crop_keywords = ['wheat', 'rice', 'corn', 'maize', 'soybean', 'cotton', 
                           'sugarcane', 'barley', 'oats', 'millet', 'sorghum']
            
            string_data = series.astype(str).str.lower()
            
            for keyword in crop_keywords:
                if string_data.str.contains(keyword).any():
                    return True
            
            # Check if it's categorical with few unique values
            unique_ratio = series.nunique() / len(series)
            return unique_ratio < 0.1 and series.nunique() <= 20
            
        except:
            return False
    
    def clean_and_standardize_data(self, df):
        """Clean and standardize the loaded data"""
        if df is None:
            return None
        
        # Make a copy
        df_clean = df.copy()
        
        # Remove completely empty rows and columns
        df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
        
        # Auto-detect and rename columns
        df_clean = self.auto_detect_columns(df_clean)
        
        # Convert numeric columns
        for col in df_clean.columns:
            if col != 'crop':  # Don't convert crop column
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with too many missing values
        threshold = len(df_clean.columns) * 0.5  # Remove rows with >50% missing values
        df_clean = df_clean.dropna(thresh=threshold)
        
        # Fill remaining missing values with median for numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill missing crop values with 'Unknown'
        if 'crop' in df_clean.columns:
            df_clean['crop'] = df_clean['crop'].fillna('Unknown')
        
        print(f"Data cleaned and standardized. Final shape: {df_clean.shape}")
        return df_clean
    
    def load_and_process_data(self, file_path):
        """Complete data loading and processing pipeline"""
        print(f"Loading data from: {file_path}")
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Clean and standardize
        df_clean = self.clean_and_standardize_data(df)
        
        # Display info
        print(f"Final processed data shape: {df_clean.shape}")
        print(f"Columns: {list(df_clean.columns)}")
        
        if 'crop' in df_clean.columns:
            print(f"Crop types: {df_clean['crop'].nunique()}")
            print(f"Crop distribution:\n{df_clean['crop'].value_counts()}")
        
        return df_clean

# Utility function to handle Numbers files
def handle_numbers_file(numbers_file_path):
    """Handle Apple Numbers files by providing instructions"""
    print("=" * 60)
    print("APPLE NUMBERS FILE DETECTED")
    print("=" * 60)
    print("Your file appears to be an Apple Numbers file.")
    print("To use this data with the crop recommendation system:")
    print("")
    print("Option 1: Export as CSV")
    print("1. Open the file in Apple Numbers")
    print("2. Go to File > Export To > CSV...")
    print("3. Save the exported CSV file in the data/ folder")
    print("4. Use the CSV file with the system")
    print("")
    print("Option 2: Export as Excel")
    print("1. Open the file in Apple Numbers")
    print("2. Go to File > Export To > Excel...")
    print("3. Save the exported Excel file in the data/ folder")
    print("4. Use the Excel file with the system")
    print("")
    print("Expected data columns:")
    print("- soil_pH (pH level)")
    print("- soil_nitrogen, soil_phosphorus, soil_potassium (nutrients)")
    print("- temperature, rainfall, humidity (climate)")
    print("- market_demand, price_per_ton (market)")
    print("- yield_tons_per_hectare (yield)")
    print("- crop (crop name)")
    print("=" * 60)

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Test with sample data
    try:
        df = loader.load_and_process_data('../data/sample_data.csv')
        if df is not None:
            print("Data loader test successful!")
            print(df.head())
    except Exception as e:
        print(f"Data loader test failed: {e}")
