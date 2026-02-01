"""
Model evaluation and metrics module for crop recommendation system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.metrics_history = []
    
    def evaluate_classification(self, model, X_test, y_test, model_name="Model", label_encoder=None):
        """Comprehensive classification evaluation"""
        print(f"\n=== Evaluating {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC AUC (for multi-class)
        roc_auc = None
        if y_pred_proba is not None and len(np.unique(y_test)) > 2:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                pass
        elif y_pred_proba is not None and len(np.unique(y_test)) == 2:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                pass
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.evaluation_results[model_name] = results
        return results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """Perform cross-validation"""
        print(f"\n=== Cross-Validation ({cv}-fold) ===")
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
    
    def learning_curve_analysis(self, model, X, y, cv=5, train_sizes=None):
        """Generate learning curve analysis"""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        print(f"\n=== Learning Curve Analysis ===")
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, 
            scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        print(f"Final training score: {train_mean[-1]:.4f}")
        print(f"Final validation score: {val_mean[-1]:.4f}")
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_mean,
            'train_scores_std': train_std,
            'val_scores_mean': val_mean,
            'val_scores_std': val_std
        }
    
    def validation_curve_analysis(self, model, X, y, param_name, param_range, cv=5):
        """Generate validation curve analysis"""
        print(f"\n=== Validation Curve Analysis for {param_name} ===")
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]
        
        print(f"Best {param_name}: {best_param}")
        print(f"Best validation score: {best_score:.4f}")
        
        return {
            'param_range': param_range,
            'train_scores_mean': train_mean,
            'train_scores_std': train_std,
            'val_scores_mean': val_mean,
            'val_scores_std': val_std,
            'best_param': best_param,
            'best_score': best_score
        }
    
    def compare_models(self, models_dict, X_test, y_test, label_encoder=None):
        """Compare multiple models"""
        print("\n=== Model Comparison ===")
        
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            try:
                results = self.evaluate_classification(
                    model, X_test, y_test, model_name, label_encoder
                )
                comparison_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        # Create comparison table
        comparison_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC AUC': results.get('roc_auc', 'N/A')
            }
            for name, results in comparison_results.items()
        ])
        
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Comparison Results:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_results, comparison_df
    
    def plot_confusion_matrix(self, cm, classes, model_name="Model", save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curve(self, learning_data, model_name="Model", save_path=None):
        """Plot learning curve"""
        plt.figure(figsize=(10, 6))
        
        train_sizes = learning_data['train_sizes']
        train_mean = learning_data['train_scores_mean']
        train_std = learning_data['train_scores_std']
        val_mean = learning_data['val_scores_mean']
        val_std = learning_data['val_scores_std']
        
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.title(f'Learning Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_validation_curve(self, validation_data, param_name, model_name="Model", save_path=None):
        """Plot validation curve"""
        plt.figure(figsize=(10, 6))
        
        param_range = validation_data['param_range']
        train_mean = validation_data['train_scores_mean']
        train_std = validation_data['train_scores_std']
        val_mean = validation_data['val_scores_mean']
        val_std = validation_data['val_scores_std']
        
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        plt.title(f'Validation Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel(param_name)
        plt.ylabel('Accuracy Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, comparison_df, save_path=None):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=plt.cm.viridis(np.linspace(0, 1, len(comparison_df))))
            
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_evaluation_report(self, models_dict, X_test, y_test, X_train=None, y_train=None, 
                                label_encoder=None, save_path=None):
        """Create comprehensive evaluation report"""
        print("=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("=" * 60)
        
        # Compare models
        comparison_results, comparison_df = self.compare_models(models_dict, X_test, y_test, label_encoder)
        
        # Best model analysis
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = models_dict[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        
        # Learning curve for best model
        if X_train is not None and y_train is not None:
            learning_data = self.learning_curve_analysis(best_model, X_train, y_train)
            self.plot_learning_curve(learning_data, best_model_name)
        
        # Confusion matrix for best model
        best_results = comparison_results[best_model_name]
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = [f'Class {i}' for i in range(len(best_results['confusion_matrix']))]
        
        self.plot_confusion_matrix(best_results['confusion_matrix'], class_names, best_model_name)
        
        # Model comparison plot
        self.plot_model_comparison(comparison_df)
        
        # Save report
        report_data = {
            'comparison_results': comparison_results,
            'comparison_table': comparison_df.to_dict(),
            'best_model': best_model_name,
            'best_accuracy': comparison_df.iloc[0]['Accuracy']
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"\nEvaluation report saved to {save_path}")
        
        return report_data

class InteractiveEvaluator:
    """Interactive evaluation with Plotly"""
    
    def __init__(self):
        pass
    
    def interactive_confusion_matrix(self, cm, classes, model_name="Model"):
        """Create interactive confusion matrix"""
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=classes, y=classes,
                       title=f"Confusion Matrix - {model_name}")
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            width=600,
            height=600
        )
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                fig.add_annotation(
                    x=j, y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                )
        
        return fig
    
    def interactive_model_comparison(self, comparison_df):
        """Create interactive model comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], 
                      y=comparison_df[metric],
                      name=metric,
                      marker_color=colors[i]),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def interactive_learning_curve(self, learning_data, model_name="Model"):
        """Create interactive learning curve"""
        fig = go.Figure()
        
        train_sizes = learning_data['train_sizes']
        train_mean = learning_data['train_scores_mean']
        train_std = learning_data['train_scores_std']
        val_mean = learning_data['val_scores_mean']
        val_std = learning_data['val_scores_std']
        
        # Training score
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            error_y=dict(type='data', array=train_std, visible=True)
        ))
        
        # Validation score
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red'),
            error_y=dict(type='data', array=val_std, visible=True)
        ))
        
        fig.update_layout(
            title=f'Learning Curve - {model_name}',
            title_x=0.5,
            xaxis_title='Training Set Size',
            yaxis_title='Accuracy Score',
            hovermode='x unified'
        )
        
        return fig

# Utility functions
def comprehensive_model_evaluation(models_dict, X_train, X_test, y_train, y_test, 
                                 label_encoder=None, save_dir='./evaluation_results'):
    """Run comprehensive evaluation on all models"""
    import os
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize evaluators
    evaluator = ModelEvaluator()
    interactive_eval = InteractiveEvaluator()
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(
        models_dict, X_test, y_test, X_train, y_train, label_encoder,
        save_path=os.path.join(save_dir, 'evaluation_report.json')
    )
    
    # Generate interactive plots
    comparison_df = pd.DataFrame(report['comparison_table'])
    
    # Interactive confusion matrix for best model
    best_model_name = report['best_model']
    best_results = report['comparison_results'][best_model_name]
    
    if label_encoder:
        class_names = label_encoder.classes_
    else:
        class_names = [f'Class {i}' for i in range(len(best_results['confusion_matrix']))]
    
    cm_fig = interactive_eval.interactive_confusion_matrix(
        best_results['confusion_matrix'], class_names, best_model_name
    )
    
    # Interactive model comparison
    comp_fig = interactive_eval.interactive_model_comparison(comparison_df)
    
    return {
        'report': report,
        'confusion_matrix_fig': cm_fig,
        'comparison_fig': comp_fig,
        'evaluator': evaluator
    }

if __name__ == "__main__":
    # Test the evaluation module
    print("Model Evaluation Module - Ready for use!")
    print("Import this module and use the classes for comprehensive model evaluation.")
