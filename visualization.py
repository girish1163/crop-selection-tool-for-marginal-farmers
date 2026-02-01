"""
Visualization module for crop recommendation system with deep graphical representations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import altair as alt
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CropVisualization:
    """Class for creating comprehensive visualizations"""
    
    def __init__(self):
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4A261',
            'info': '#264653'
        }
    
    def plot_data_distribution(self, df, save_path=None):
        """Create comprehensive data distribution plots"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Crop Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for i, col in enumerate(numeric_columns[:9]):
            row, col_idx = i // 3, i % 3
            ax = axes[row, col_idx]
            
            # Histogram with KDE
            sns.histplot(data=df, x=col, kde=True, ax=ax, color=self.colors['primary'])
            ax.set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        # Remove empty subplots
        for i in range(len(numeric_columns), 9):
            row, col_idx = i // 3, i % 3
            fig.delaxes(axes[row, col_idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, df, save_path=None):
        """Create correlation matrix heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_crop_distribution(self, df, save_path=None):
        """Create crop distribution visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Crop Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Count plot
        sns.countplot(data=df, x='crop', ax=axes[0, 0], palette='husl')
        axes[0, 0].set_title('Crop Count Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Crop Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        crop_counts = df['crop'].value_counts()
        axes[0, 1].pie(crop_counts.values, labels=crop_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=sns.color_palette('husl', len(crop_counts)))
        axes[0, 1].set_title('Crop Percentage Distribution', fontweight='bold')
        
        # Box plot of yield by crop
        if 'yield_tons_per_hectare' in df.columns:
            sns.boxplot(data=df, x='crop', y='yield_tons_per_hectare', ax=axes[1, 0], palette='husl')
            axes[1, 0].set_title('Yield Distribution by Crop', fontweight='bold')
            axes[1, 0].set_xlabel('Crop Type')
            axes[1, 0].set_ylabel('Yield (tons/hectare)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Market demand by crop
        if 'market_demand' in df.columns:
            sns.barplot(data=df, x='crop', y='market_demand', ax=axes[1, 1], palette='husl')
            axes[1, 1].set_title('Market Demand by Crop', fontweight='bold')
            axes[1, 1].set_xlabel('Crop Type')
            axes[1, 1].set_ylabel('Market Demand')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df, save_path=None):
        """Create feature importance visualization"""
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar plot
        sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature',
                   palette='viridis')
        
        plt.title('Top 15 Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        
        # Add value labels on bars
        for i, v in enumerate(feature_importance_df.head(15)['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance(self, model_results, save_path=None):
        """Create model performance comparison visualization"""
        if not model_results:
            print("No model results to plot")
            return
        
        # Prepare data
        models = list(model_results.keys())
        accuracies = [result['accuracy'] for result in model_results.values()]
        
        plt.figure(figsize=(12, 8))
        
        # Bar plot with gradient colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = plt.bar(models, accuracies, color=colors)
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, classes, save_path=None):
        """Create confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class InteractiveVisualization:
    """Class for creating interactive visualizations using Plotly"""
    
    def create_interactive_scatter(self, df, x_col, y_col, color_col='crop', size_col=None):
        """Create interactive scatter plot"""
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                        hover_data=df.columns, title=f'{y_col} vs {x_col} by {color_col}')
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            showlegend=True,
            height=600
        )
        
        return fig
    
    def create_interactive_3d_scatter(self, df, x_col, y_col, z_col, color_col='crop'):
        """Create interactive 3D scatter plot"""
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                           hover_data=df.columns, title=f'3D Plot: {x_col}, {y_col}, {z_col}')
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            showlegend=True,
            height=700
        )
        
        return fig
    
    def create_interactive_box_plot(self, df, x_col, y_col):
        """Create interactive box plot"""
        fig = px.box(df, x=x_col, y=y_col, title=f'{y_col} Distribution by {x_col}')
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_interactive_histogram(self, df, col, color_col='crop'):
        """Create interactive histogram"""
        fig = px.histogram(df, x=col, color=color_col, nbins=30,
                          title=f'Distribution of {col}')
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_model_performance_dashboard(self, model_results):
        """Create interactive model performance dashboard"""
        if not model_results:
            return None
        
        # Prepare data
        models = list(model_results.keys())
        accuracies = [result['accuracy'] for result in model_results.values()]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy', 'Accuracy Distribution', 
                          'Model Comparison', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name='Accuracy'),
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=accuracies, name='Accuracy Distribution'),
            row=1, col=2
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=list(range(len(models))), y=accuracies, 
                      mode='markers+lines', name='Model Performance'),
            row=2, col=1
        )
        
        # Table
        fig.add_trace(
            go.Table(
                header=dict(values=['Model', 'Accuracy']),
                cells=dict(values=[models, [f'{acc:.4f}' for acc in accuracies]])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Model Performance Dashboard',
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig

class AdvancedVisualization:
    """Class for advanced visualizations using Altair and Bokeh"""
    
    def create_altair_chart(self, df, x_col, y_col, color_col='crop'):
        """Create visualization using Altair"""
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X(x_col, scale=alt.Scale(zero=False)),
            y=alt.Y(y_col, scale=alt.Scale(zero=False)),
            color=color_col,
            tooltip=list(df.columns)
        ).properties(
            title=f'{y_col} vs {x_col}',
            width=600,
            height=400
        ).interactive()
        
        return chart
    
    def create_bokeh_plot(self, df, x_col, y_col, color_col='crop'):
        """Create visualization using Bokeh"""
        # Create figure
        p = figure(title=f'{y_col} vs {x_col}', 
                  x_axis_label=x_col, 
                  y_axis_label=y_col,
                  tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                  width=800, height=600)
        
        # Add hover tool
        hover = HoverTool()
        hover.tooltips = [(col, f"@{col}") for col in df.columns]
        p.add_tools(hover)
        
        # Plot data points
        colors = df[color_col].astype('category').cat.codes
        scatter = p.circle(df[x_col], df[y_col], size=10, color=colors, alpha=0.6)
        
        return p
    
    def create_market_analysis_dashboard(self, df):
        """Create comprehensive market analysis dashboard"""
        if 'market_demand' not in df.columns or 'price_per_ton' not in df.columns:
            print("Market data not available")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Market Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Market demand by crop
        sns.barplot(data=df, x='crop', y='market_demand', ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title('Market Demand by Crop')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Price distribution
        sns.boxplot(data=df, x='crop', y='price_per_ton', ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_title('Price Distribution by Crop')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Revenue analysis
        if 'yield_tons_per_hectare' in df.columns:
            df['revenue_per_hectare'] = df['yield_tons_per_hectare'] * df['price_per_ton']
            sns.barplot(data=df, x='crop', y='revenue_per_hectare', ax=axes[0, 2], palette='viridis')
            axes[0, 2].set_title('Revenue per Hectare by Crop')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Demand vs Price scatter
        sns.scatterplot(data=df, x='market_demand', y='price_per_ton', 
                       hue='crop', ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Demand vs Price Analysis')
        
        # Yield vs Market Demand
        if 'yield_tons_per_hectare' in df.columns:
            sns.scatterplot(data=df, x='market_demand', y='yield_tons_per_hectare',
                           hue='crop', ax=axes[1, 1], palette='viridis')
            axes[1, 1].set_title('Yield vs Market Demand')
        
        # Market efficiency (Price/Yield ratio)
        if 'yield_tons_per_hectare' in df.columns:
            df['price_yield_ratio'] = df['price_per_ton'] / df['yield_tons_per_hectare']
            sns.barplot(data=df, x='crop', y='price_yield_ratio', ax=axes[1, 2], palette='viridis')
            axes[1, 2].set_title('Price per Unit Yield')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Utility functions
def create_comprehensive_dashboard(df, model_results=None, feature_importance=None):
    """Create a comprehensive dashboard with all visualizations"""
    
    # Initialize visualization classes
    viz = CropVisualization()
    interactive_viz = InteractiveVisualization()
    advanced_viz = AdvancedVisualization()
    
    print("=== Creating Comprehensive Dashboard ===")
    
    # Static visualizations
    viz.plot_data_distribution(df)
    viz.plot_correlation_matrix(df)
    viz.plot_crop_distribution(df)
    
    if feature_importance is not None:
        viz.plot_feature_importance(feature_importance)
    
    if model_results is not None:
        viz.plot_model_performance(model_results)
    
    # Market analysis
    advanced_viz.create_market_analysis_dashboard(df)
    
    # Interactive visualizations
    if 'soil_pH' in df.columns and 'yield_tons_per_hectare' in df.columns:
        scatter_fig = interactive_viz.create_interactive_scatter(
            df, 'soil_pH', 'yield_tons_per_hectare'
        )
        scatter_fig.show()
    
    if model_results is not None:
        dashboard_fig = interactive_viz.create_model_performance_dashboard(model_results)
        if dashboard_fig:
            dashboard_fig.show()
    
    print("Dashboard creation complete!")

if __name__ == "__main__":
    # Test visualizations with sample data
    from .data_preprocessing import load_sample_data
    
    df = load_sample_data()
    if df is not None:
        create_comprehensive_dashboard(df)
