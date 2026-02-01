"""
🌾 COMPLETE CROP RECOMMENDATION SYSTEM - ALL IN ONE FILE
Your friend can run this file and show everything to jury/faculty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Machine Learning Imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import joblib

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class CompleteCropRecommendationSystem:
    """Complete system that does everything from data loading to visualization"""
    
    def __init__(self):
        print("🌾 === CROP RECOMMENDATION SYSTEM INITIALIZED ===")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        self.models = {}
        self.best_model = None
        self.model_scores = {}
        self.feature_columns = None
        self.target_column = None
        
        # Create results directory
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"📁 Results will be saved to: {self.results_dir}")
        
    def create_comprehensive_dataset(self):
        """Create a comprehensive dataset for demonstration"""
        print("\n📊 === CREATING COMPREHENSIVE DATASET ===")
        
        np.random.seed(42)
        n_samples = 500
        
        # Define crop-specific characteristics
        crop_data = []
        
        for crop in ['Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton']:
            n_crop = n_samples // 5
            
            if crop == 'Wheat':
                pH = np.random.normal(6.8, 0.5, n_crop)
                nitrogen = np.random.normal(85, 15, n_crop)
                phosphorus = np.random.normal(45, 10, n_crop)
                potassium = np.random.normal(125, 15, n_crop)
                temperature = np.random.normal(20, 3, n_crop)
                rainfall = np.random.normal(600, 150, n_crop)
                humidity = np.random.normal(60, 8, n_crop)
                market_demand = np.random.normal(80, 10, n_crop)
                price = np.random.normal(2400, 300, n_crop)
                yield_tons = np.random.normal(3.0, 0.5, n_crop)
                
            elif crop == 'Rice':
                pH = np.random.normal(6.2, 0.4, n_crop)
                nitrogen = np.random.normal(95, 15, n_crop)
                phosphorus = np.random.normal(40, 8, n_crop)
                potassium = np.random.normal(115, 12, n_crop)
                temperature = np.random.normal(25, 2, n_crop)
                rainfall = np.random.normal(1200, 300, n_crop)
                humidity = np.random.normal(75, 8, n_crop)
                market_demand = np.random.normal(75, 8, n_crop)
                price = np.random.normal(2200, 250, n_crop)
                yield_tons = np.random.normal(2.8, 0.4, n_crop)
                
            elif crop == 'Corn':
                pH = np.random.normal(6.5, 0.5, n_crop)
                nitrogen = np.random.normal(75, 12, n_crop)
                phosphorus = np.random.normal(55, 10, n_crop)
                potassium = np.random.normal(135, 15, n_crop)
                temperature = np.random.normal(23, 3, n_crop)
                rainfall = np.random.normal(800, 200, n_crop)
                humidity = np.random.normal(65, 8, n_crop)
                market_demand = np.random.normal(90, 8, n_crop)
                price = np.random.normal(2800, 350, n_crop)
                yield_tons = np.random.normal(3.8, 0.6, n_crop)
                
            elif crop == 'Soybeans':
                pH = np.random.normal(6.7, 0.4, n_crop)
                nitrogen = np.random.normal(60, 10, n_crop)
                phosphorus = np.random.normal(50, 8, n_crop)
                potassium = np.random.normal(130, 12, n_crop)
                temperature = np.random.normal(24, 3, n_crop)
                rainfall = np.random.normal(700, 150, n_crop)
                humidity = np.random.normal(60, 7, n_crop)
                market_demand = np.random.normal(85, 10, n_crop)
                price = np.random.normal(3200, 400, n_crop)
                yield_tons = np.random.normal(2.5, 0.4, n_crop)
                
            else:  # Cotton
                pH = np.random.normal(6.3, 0.5, n_crop)
                nitrogen = np.random.normal(70, 12, n_crop)
                phosphorus = np.random.normal(48, 8, n_crop)
                potassium = np.random.normal(120, 15, n_crop)
                temperature = np.random.normal(26, 3, n_crop)
                rainfall = np.random.normal(900, 200, n_crop)
                humidity = np.random.normal(55, 8, n_crop)
                market_demand = np.random.normal(70, 8, n_crop)
                price = np.random.normal(2600, 300, n_crop)
                yield_tons = np.random.normal(2.2, 0.4, n_crop)
            
            # Ensure values are in realistic ranges
            pH = np.clip(pH, 4.0, 9.0)
            nitrogen = np.clip(nitrogen, 20, 150)
            phosphorus = np.clip(phosphorus, 10, 100)
            potassium = np.clip(potassium, 50, 200)
            temperature = np.clip(temperature, 10, 40)
            rainfall = np.clip(rainfall, 200, 2500)
            humidity = np.clip(humidity, 20, 100)
            market_demand = np.clip(market_demand, 50, 100)
            price = np.clip(price, 1500, 5000)
            yield_tons = np.clip(yield_tons, 1.0, 8.0)
            
            for i in range(n_crop):
                crop_data.append({
                    'soil_pH': pH[i],
                    'soil_nitrogen': nitrogen[i],
                    'soil_phosphorus': phosphorus[i],
                    'soil_potassium': potassium[i],
                    'temperature': temperature[i],
                    'rainfall': rainfall[i],
                    'humidity': humidity[i],
                    'market_demand': market_demand[i],
                    'price_per_ton': price[i],
                    'yield_tons_per_hectare': yield_tons[i],
                    'crop': crop
                })
        
        df = pd.DataFrame(crop_data)
        print(f"✅ Dataset created with {len(df)} samples")
        print(f"   Crop distribution: {df['crop'].value_counts().to_dict()}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data"""
        print("\n🧹 === DATA PREPROCESSING ===")
        
        # Make a copy
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_columns] = self.imputer.fit_transform(df_clean[numeric_columns])
        
        # Feature engineering
        df_clean['soil_quality_index'] = (
            df_clean['soil_nitrogen'] * 0.4 + 
            df_clean['soil_phosphorus'] * 0.3 + 
            df_clean['soil_potassium'] * 0.3
        )
        
        df_clean['climate_suitability'] = (
            np.where(df_clean['temperature'].between(20, 30), 1, 0.5) *
            np.where(df_clean['rainfall'].between(800, 1500), 1, 0.5) *
            np.where(df_clean['humidity'].between(50, 80), 1, 0.5)
        )
        
        df_clean['market_score'] = df_clean['market_demand'] * df_clean['price_per_ton'] / 10000
        df_clean['revenue_per_hectare'] = df_clean['yield_tons_per_hectare'] * df_clean['price_per_ton']
        
        # Separate features and target
        X = df_clean.drop(columns=['crop'])
        y = df_clean['crop']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        self.target_column = 'crop'
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"✅ Data preprocessed successfully!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Features: {len(self.feature_columns)}")
        
        return X_train, X_test, y_train, y_test, df_clean
    
    def initialize_models(self):
        """Initialize all machine learning models"""
        print("\n🤖 === INITIALIZING ML MODELS ===")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, random_state=42, max_depth=6
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100, random_state=42, max_depth=6
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42
            )
        }
        
        print(f"✅ {len(self.models)} models initialized:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("\n🏋️ === TRAINING MODELS ===")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"   Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            self.model_scores[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   ✅ {model_name}: CV Accuracy = {cv_scores.mean():.4f}")
        
        # Find best model
        best_model_name = max(self.model_scores, key=lambda x: self.model_scores[x]['cv_mean'])
        self.best_model = trained_models[best_model_name]
        
        print(f"\n🏆 Best model: {best_model_name} with CV accuracy: {self.model_scores[best_model_name]['cv_mean']:.4f}")
        
        return trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\n📊 === MODEL EVALUATION ===")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"   {model_name}: Test Accuracy = {accuracy:.4f}")
        
        return evaluation_results
    
    def create_visualizations(self, df, evaluation_results):
        """Create comprehensive visualizations"""
        print("\n📈 === CREATING VISUALIZATIONS ===")
        
        # Graph 1: Data Distribution
        plt.figure(figsize=(15, 10))
        plt.suptitle('🌾 Crop Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:9]
        for i, col in enumerate(numeric_cols):
            plt.subplot(3, 3, i+1)
            sns.histplot(data=df, x=col, kde=True)
            plt.title(col.replace('_', ' ').title())
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/1_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 1_data_distribution.png")
        
        # Graph 2: Crop Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🌱 Crop Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Count plot
        df['crop'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Crop Count Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        axes[0,1].pie(df['crop'].value_counts(), labels=df['crop'].value_counts().index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Crop Percentage Distribution')
        
        # Yield by crop
        sns.boxplot(data=df, x='crop', y='yield_tons_per_hectare', ax=axes[1,0])
        axes[1,0].set_title('Yield Distribution by Crop')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Market demand by crop
        sns.barplot(data=df, x='crop', y='market_demand', ax=axes[1,1])
        axes[1,1].set_title('Market Demand by Crop')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/2_crop_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 2_crop_distribution.png")
        
        # Graph 3: Correlation Matrix
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f')
        plt.title('🔗 Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.savefig(f'{self.results_dir}/3_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 3_correlation_matrix.png")
        
        # Graph 4: Soil Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🌍 Soil Parameters Analysis', fontsize=16, fontweight='bold')
        
        soil_cols = ['soil_pH', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium']
        for i, col in enumerate(soil_cols):
            row, col_idx = i // 2, i % 2
            sns.boxplot(data=df, x='crop', y=col, ax=axes[row, col_idx])
            axes[row, col_idx].set_title(col.replace('_', ' ').title())
            axes[row, col_idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/4_soil_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 4_soil_analysis.png")
        
        # Graph 5: Climate Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('🌤️ Climate Parameters Analysis', fontsize=16, fontweight='bold')
        
        climate_cols = ['temperature', 'rainfall', 'humidity']
        for i, col in enumerate(climate_cols):
            sns.boxplot(data=df, x='crop', y=col, ax=axes[i])
            axes[i].set_title(col.title())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/5_climate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 5_climate_analysis.png")
        
        # Graph 6: Market Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('💰 Market Analysis', fontsize=16, fontweight='bold')
        
        # Market demand
        sns.barplot(data=df, x='crop', y='market_demand', ax=axes[0,0])
        axes[0,0].set_title('Market Demand by Crop')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Price distribution
        sns.boxplot(data=df, x='crop', y='price_per_ton', ax=axes[0,1])
        axes[0,1].set_title('Price Distribution by Crop')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Revenue analysis
        sns.barplot(data=df, x='crop', y='revenue_per_hectare', ax=axes[1,0])
        axes[1,0].set_title('Revenue per Hectare by Crop')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Market performance
        sns.barplot(data=df, x='crop', y='market_score', ax=axes[1,1])
        axes[1,1].set_title('Market Performance Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/6_market_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 6_market_analysis.png")
        
        # Graph 7: Model Performance Comparison
        if evaluation_results:
            models = list(evaluation_results.keys())
            accuracies = [result['accuracy'] for result in evaluation_results.values()]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(models, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
            plt.title('🏆 Model Performance Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Models')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/7_model_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"💾 Saved: 7_model_performance.png")
        
        # Graph 8: Feature Importance (if available)
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('🎯 Top 10 Feature Importance', fontsize=16, fontweight='bold')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/8_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"💾 Saved: 8_feature_importance.png")
        
        # Graph 9: Interactive 3D Plot (using matplotlib)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {'Wheat': 'red', 'Rice': 'blue', 'Corn': 'green', 
                 'Soybeans': 'orange', 'Cotton': 'purple'}
        
        for crop in df['crop'].unique():
            crop_data = df[df['crop'] == crop]
            ax.scatter(crop_data['soil_pH'], crop_data['temperature'], 
                      crop_data['yield_tons_per_hectare'], 
                      c=colors[crop], label=crop, alpha=0.6)
        
        ax.set_xlabel('Soil pH')
        ax.set_ylabel('Temperature (°C)')
        ax.set_zlabel('Yield (tons/hectare)')
        ax.set_title('🌐 3D Analysis: Soil pH vs Temperature vs Yield', fontsize=14, fontweight='bold')
        ax.legend()
        plt.savefig(f'{self.results_dir}/9_3d_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 9_3d_analysis.png")
        
        # Graph 10: Weather Conditions Impact
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🌦️ Weather Conditions Impact Analysis', fontsize=16, fontweight='bold')
        
        # Temperature vs Yield
        sns.scatterplot(data=df, x='temperature', y='yield_tons_per_hectare', 
                       hue='crop', ax=axes[0,0])
        axes[0,0].set_title('Temperature vs Yield')
        
        # Rainfall vs Yield
        sns.scatterplot(data=df, x='rainfall', y='yield_tons_per_hectare', 
                       hue='crop', ax=axes[0,1])
        axes[0,1].set_title('Rainfall vs Yield')
        
        # Humidity vs Yield
        sns.scatterplot(data=df, x='humidity', y='yield_tons_per_hectare', 
                       hue='crop', ax=axes[1,0])
        axes[1,0].set_title('Humidity vs Yield')
        
        # Climate suitability vs Yield
        sns.scatterplot(data=df, x='climate_suitability', y='yield_tons_per_hectare', 
                       hue='crop', ax=axes[1,1])
        axes[1,1].set_title('Climate Suitability vs Yield')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/10_weather_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 10_weather_impact.png")
        
        # Graph 11: Recommendations Dashboard
        self.create_recommendation_dashboard(df)
        
        # Graph 12: Summary Statistics
        self.create_summary_statistics(df)
    
    def create_recommendation_dashboard(self, df):
        """Create a recommendations dashboard"""
        print("\n🎯 === CREATING RECOMMENDATIONS DASHBOARD ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🌾 Crop Recommendations Dashboard', fontsize=16, fontweight='bold')
        
        # Best crops for different conditions
        # High temperature
        high_temp = df[df['temperature'] > df['temperature'].quantile(0.75)]
        best_high_temp = high_temp.groupby('crop')['yield_tons_per_hectare'].mean().idxmax()
        
        # Low rainfall
        low_rain = df[df['rainfall'] < df['rainfall'].quantile(0.25)]
        best_low_rain = low_rain.groupby('crop')['yield_tons_per_hectare'].mean().idxmax()
        
        # High market demand
        high_demand = df[df['market_demand'] > df['market_demand'].quantile(0.75)]
        best_high_demand = high_demand.groupby('crop')['revenue_per_hectare'].mean().idxmax()
        
        # Best soil quality
        high_soil = df[df['soil_quality_index'] > df['soil_quality_index'].quantile(0.75)]
        best_soil = high_soil.groupby('crop')['yield_tons_per_hectare'].mean().idxmax()
        
        # Most profitable
        most_profitable = df.groupby('crop')['revenue_per_hectare'].mean().idxmax()
        
        # Highest yielding
        highest_yielding = df.groupby('crop')['yield_tons_per_hectare'].mean().idxmax()
        
        recommendations = [
            ('High Temperature (>75%)', best_high_temp),
            ('Low Rainfall (<25%)', best_low_rain),
            ('High Market Demand', best_high_demand),
            ('Best Soil Quality', best_soil),
            ('Most Profitable', most_profitable),
            ('Highest Yielding', highest_yielding)
        ]
        
        for i, (condition, crop) in enumerate(recommendations):
            row, col = i // 3, i % 3
            
            # Create a simple visualization
            crop_data = df[df['crop'] == crop]
            
            # Show the crop name and condition
            axes[row, col].text(0.5, 0.7, crop, fontsize=20, fontweight='bold', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].text(0.5, 0.3, condition, fontsize=12, 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            
            # Add some visual indicator
            colors = {'Wheat': '#F4A460', 'Rice': '#87CEEB', 'Corn': '#FFD700', 
                     'Soybeans': '#90EE90', 'Cotton': '#DDA0DD'}
            axes[row, col].add_patch(plt.Circle((0.5, 0.1), 0.08, 
                                               color=colors.get(crop, 'gray'), 
                                               transform=axes[row, col].transAxes))
            
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)
            axes[row, col].axis('off')
            axes[row, col].set_title(condition, fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/11_recommendations_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 11_recommendations_dashboard.png")
    
    def create_summary_statistics(self, df):
        """Create summary statistics visualization"""
        print("\n📋 === CREATING SUMMARY STATISTICS ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('📊 Summary Statistics & Insights', fontsize=16, fontweight='bold')
        
        # Overall statistics
        stats_text = f"""
        Dataset Overview:
        • Total Samples: {len(df):,}
        • Crop Types: {df['crop'].nunique()}
        • Features: {len(df.columns)-1}
        
        Average Values:
        • Soil pH: {df['soil_pH'].mean():.2f}
        • Temperature: {df['temperature'].mean():.1f}°C
        • Rainfall: {df['rainfall'].mean():.0f}mm
        • Yield: {df['yield_tons_per_hectare'].mean():.2f} tons/ha
        • Market Demand: {df['market_demand'].mean():.0f}
        • Price: ${df['price_per_ton'].mean():.0f}/ton
        """
        
        axes[0,0].text(0.05, 0.95, stats_text, fontsize=11, va='top', 
                      transform=axes[0,0].transAxes, family='monospace')
        axes[0,0].axis('off')
        axes[0,0].set_title('Dataset Statistics', fontweight='bold')
        
        # Crop-wise performance
        crop_performance = df.groupby('crop').agg({
            'yield_tons_per_hectare': 'mean',
            'revenue_per_hectare': 'mean',
            'market_demand': 'mean'
        }).round(2)
        
        axes[0,1].axis('off')
        axes[0,1].text(0.05, 0.95, 'Crop Performance:', fontsize=12, fontweight='bold',
                      transform=axes[0,1].transAxes)
        
        y_pos = 0.85
        for crop, row in crop_performance.iterrows():
            text = f"{crop}: Yield={row['yield_tons_per_hectare']}t/ha, "
            text += f"Revenue=${row['revenue_per_hectare']:.0f}, Demand={row['market_demand']:.0f}"
            axes[0,1].text(0.05, y_pos, text, fontsize=10, 
                          transform=axes[0,1].transAxes)
            y_pos -= 0.15
        
        axes[0,1].set_title('Crop Performance Summary', fontweight='bold')
        
        # Best conditions for each crop
        axes[1,0].axis('off')
        axes[1,0].text(0.05, 0.95, 'Optimal Conditions by Crop:', fontsize=12, fontweight='bold',
                      transform=axes[1,0].transAxes)
        
        y_pos = 0.85
        for crop in df['crop'].unique():
            crop_data = df[df['crop'] == crop]
            optimal_temp = crop_data['temperature'].mean()
            optimal_rain = crop_data['rainfall'].mean()
            optimal_ph = crop_data['soil_pH'].mean()
            
            text = f"{crop}: Temp={optimal_temp:.1f}°C, Rain={optimal_rain:.0f}mm, pH={optimal_ph:.1f}"
            axes[1,0].text(0.05, y_pos, text, fontsize=10, 
                          transform=axes[1,0].transAxes)
            y_pos -= 0.15
        
        axes[1,0].set_title('Optimal Growing Conditions', fontweight='bold')
        
        # Model performance summary
        if self.model_scores:
            axes[1,1].axis('off')
            axes[1,1].text(0.05, 0.95, 'Model Performance:', fontsize=12, fontweight='bold',
                          transform=axes[1,1].transAxes)
            
            y_pos = 0.85
            sorted_models = sorted(self.model_scores.items(), 
                                 key=lambda x: x[1]['cv_mean'], reverse=True)
            
            for model_name, scores in sorted_models[:5]:
                text = f"{model_name}: {scores['cv_mean']:.4f} ± {scores['cv_std']:.4f}"
                axes[1,1].text(0.05, y_pos, text, fontsize=10, 
                              transform=axes[1,1].transAxes)
                y_pos -= 0.15
            
            axes[1,1].set_title('Top 5 Models', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/12_summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Saved: 12_summary_statistics.png")
    
    def make_sample_predictions(self):
        """Make sample predictions to demonstrate the system"""
        print("\n🔮 === MAKING SAMPLE PREDICTIONS ===")
        
        # Create sample scenarios
        scenarios = [
            {
                'name': 'Wheat Farm in Moderate Climate',
                'data': {
                    'soil_pH': [6.8], 'soil_nitrogen': [85], 'soil_phosphorus': [45], 
                    'soil_potassium': [125], 'temperature': [20], 'rainfall': [600], 
                    'humidity': [60], 'market_demand': [80], 'price_per_ton': [2400], 
                    'yield_tons_per_hectare': [3.0]
                }
            },
            {
                'name': 'Rice Paddy in Tropical Climate',
                'data': {
                    'soil_pH': [6.2], 'soil_nitrogen': [95], 'soil_phosphorus': [40], 
                    'soil_potassium': [115], 'temperature': [25], 'rainfall': [1200], 
                    'humidity': [75], 'market_demand': [75], 'price_per_ton': [2200], 
                    'yield_tons_per_hectare': [2.8]
                }
            },
            {
                'name': 'Corn Farm in Warm Climate',
                'data': {
                    'soil_pH': [6.5], 'soil_nitrogen': [75], 'soil_phosphorus': [55], 
                    'soil_potassium': [135], 'temperature': [23], 'rainfall': [800], 
                    'humidity': [65], 'market_demand': [90], 'price_per_ton': [2800], 
                    'yield_tons_per_hectare': [3.8]
                }
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            
            # Create DataFrame
            input_df = pd.DataFrame(scenario['data'])
            
            # Add engineered features
            input_df['soil_quality_index'] = (
                input_df['soil_nitrogen'] * 0.4 + 
                input_df['soil_phosphorus'] * 0.3 + 
                input_df['soil_potassium'] * 0.3
            )
            
            input_df['climate_suitability'] = (
                np.where(input_df['temperature'].between(20, 30), 1, 0.5) *
                np.where(input_df['rainfall'].between(800, 1500), 1, 0.5) *
                np.where(input_df['humidity'].between(50, 80), 1, 0.5)
            )
            
            input_df['market_score'] = input_df['market_demand'] * input_df['price_per_ton'] / 10000
            input_df['revenue_per_hectare'] = input_df['yield_tons_per_hectare'] * input_df['price_per_ton']
            
            # Scale features
            input_scaled = self.scaler.transform(input_df[self.feature_columns])
            input_scaled = pd.DataFrame(input_scaled, columns=self.feature_columns)
            
            # Make prediction
            if self.best_model:
                prediction = self.best_model.predict(input_scaled)
                prediction_proba = self.best_model.predict_proba(input_scaled)
                
                # Convert back to crop names
                predicted_crop = self.label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(prediction_proba)
                
                print(f"   🌱 Recommended Crop: {predicted_crop}")
                print(f"   🎯 Confidence: {confidence:.2%}")
                
                # Show top 3 recommendations
                top_indices = np.argsort(prediction_proba[0])[-3:][::-1]
                print("   🏆 Top 3 Recommendations:")
                for i, idx in enumerate(top_indices, 1):
                    crop_name = self.label_encoder.inverse_transform([idx])[0]
                    prob = prediction_proba[0][idx]
                    print(f"      {i}. {crop_name} - {prob:.2%}")
    
    def run_complete_analysis(self):
        """Run the complete analysis from start to finish"""
        print("🚀 === STARTING COMPLETE CROP RECOMMENDATION ANALYSIS ===")
        
        # Step 1: Create and load data
        df = self.create_comprehensive_dataset()
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, df_processed = self.preprocess_data(df)
        
        # Step 3: Initialize models
        self.initialize_models()
        
        # Step 4: Train models
        trained_models = self.train_models(X_train, y_train)
        
        # Step 5: Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # Step 6: Create visualizations
        self.create_visualizations(df_processed, evaluation_results)
        
        # Step 7: Make sample predictions
        self.make_sample_predictions()
        
        # Step 8: Final summary
        print("\n🎉 === ANALYSIS COMPLETE ===")
        print("✅ Successfully demonstrated:")
        print("   • Data preprocessing and feature engineering")
        print("   • Multiple machine learning models")
        print("   • Model evaluation and comparison")
        print("   • Comprehensive visualizations (12 graphs)")
        print("   • Crop recommendations with confidence scores")
        print("   • Weather conditions impact analysis")
        print("   • Market analysis and profitability insights")
        
        # Save summary report
        self.save_summary_report(df, evaluation_results)
        
        print("\n🌾 This system is ready for presentation to jury/faculty!")
        print(f"📁 All images saved to: {self.results_dir}")
    
    def save_summary_report(self, df, evaluation_results):
        """Save a comprehensive summary report"""
        report_path = os.path.join(self.results_dir, 'ANALYSIS_REPORT.txt')
        
        with open(report_path, 'w') as f:
            f.write("🌾 CROP RECOMMENDATION SYSTEM - ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"• Total Samples: {len(df):,}\n")
            f.write(f"• Crop Types: {df['crop'].nunique()}\n")
            f.write(f"• Features: {len(df.columns)-1}\n")
            f.write(f"• Training Samples: {len(df) * 0.8:.0f}\n")
            f.write(f"• Test Samples: {len(df) * 0.2:.0f}\n\n")
            
            f.write("CROP DISTRIBUTION:\n")
            for crop, count in df['crop'].value_counts().items():
                percentage = (count / len(df)) * 100
                f.write(f"• {crop}: {count} samples ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("AVERAGE VALUES:\n")
            f.write(f"• Soil pH: {df['soil_pH'].mean():.2f}\n")
            f.write(f"• Temperature: {df['temperature'].mean():.1f}°C\n")
            f.write(f"• Rainfall: {df['rainfall'].mean():.0f}mm\n")
            f.write(f"• Humidity: {df['humidity'].mean():.0f}%\n")
            f.write(f"• Yield: {df['yield_tons_per_hectare'].mean():.2f} tons/ha\n")
            f.write(f"• Market Demand: {df['market_demand'].mean():.0f}\n")
            f.write(f"• Price: ${df['price_per_ton'].mean():.0f}/ton\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            if evaluation_results:
                sorted_models = sorted(evaluation_results.items(), 
                                     key=lambda x: x[1]['accuracy'], reverse=True)
                for model_name, result in sorted_models:
                    f.write(f"• {model_name}: {result['accuracy']:.4f} accuracy\n")
            f.write("\n")
            
            f.write("BEST MODEL:\n")
            if self.best_model:
                best_name = type(self.best_model).__name__
                # Find the corresponding model name in evaluation_results
                best_eval_name = None
                for eval_name in evaluation_results.keys():
                    if best_name.lower() in eval_name.lower():
                        best_eval_name = eval_name
                        break
                
                if best_eval_name:
                    f.write(f"• Model: {best_name}\n")
                    f.write(f"• Accuracy: {evaluation_results[best_eval_name]['accuracy']:.4f}\n")
                else:
                    f.write(f"• Model: {best_name}\n")
                    f.write(f"• Accuracy: Not available in evaluation results\n")
            f.write("\n")
            
            f.write("FEATURE IMPORTANCE (Top 5):\n")
            if hasattr(self.best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for i, row in importance_df.head(5).iterrows():
                    f.write(f"• {row['feature']}: {row['importance']:.4f}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("• Use " + type(self.best_model).__name__ + " for best accuracy\n")
            f.write("• Focus on soil quality and climate suitability\n")
            f.write("• Consider market demand for profitability\n")
            f.write("• Monitor weather conditions for optimal yield\n")
        
        print(f"📄 Analysis report saved to: {report_path}")

# Main execution
if __name__ == "__main__":
    print("🌾 === CROP RECOMMENDATION SYSTEM FOR PRESENTATION ===")
    print("Your friend can run this file to show everything to jury/faculty!")
    print("This includes: ML models, 12+ visualizations, recommendations, and more!")
    print("=" * 70)
    
    # Create and run the complete system
    system = CompleteCropRecommendationSystem()
    system.run_complete_analysis()
    
    print("\n✨ Presentation ready! All graphs and results displayed! ✨")
