"""
Energy Demand Prediction Model Training
Uses Random Forest and ANN with direct training (no iterative optimization)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import time
import warnings
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
warnings.filterwarnings('ignore')

class EnergyDemandPredictor:
    """
    Main class for energy demand prediction using Random Forest and ANN
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {}
        self.metrics = {}
        self.best_params = {}
        self.training_time = {}
        self.cv_results = {}
        
    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess the dataset
        """
        print("\n" + "="*60)
        print("Loading and preprocessing data...")
        print("="*60)
        
        # Load data
        data = pd.read_csv(filepath)
        print(f"Dataset shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        print(f"\nFirst few rows:")
        print(data.head())
        
        # Separate features and target
        X = data[['voltage', 'current', 'temp', 'humidity']].values
        y = data['DEMAND'].values.reshape(-1, 1)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state
        )  # 0.25 * 0.8 = 0.2 validation
        
        print(f"\nData Split:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Scale target
        y_train_scaled = self.scaler_y.fit_transform(y_train).ravel()
        y_val_scaled = self.scaler_y.transform(y_val).ravel()
        y_test_scaled = self.scaler_y.transform(y_test).ravel()
        
        # Store original values for later use
        self.y_train_orig = y_train.ravel()
        self.y_val_orig = y_val.ravel()
        self.y_test_orig = y_test.ravel()
        
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train_scaled, y_val_scaled, y_test_scaled,
                X_train, X_val, X_test)
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Train Random Forest Regressor with optimized hyperparameters
        """
        print("\n" + "-"*40)
        print("Training Random Forest Regressor...")
        start_time = time.time()
        
        # Define hyperparameter grid for Random Forest
        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Create Random Forest model
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Use RandomizedSearchCV for faster hyperparameter tuning
        print("Performing randomized hyperparameter search...")
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the random search model
        rf_random.fit(X_train, y_train)
        
        # Get best parameters
        self.best_params['random_forest'] = rf_random.best_params_
        print(f"\nBest Random Forest parameters:")
        for param, value in rf_random.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV Score: {-rf_random.best_score_:.4f} MSE")
        
        # Train final model with best parameters
        final_model = RandomForestRegressor(
            **rf_random.best_params_,
            random_state=self.random_state,
            n_jobs=-1
        )
        final_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)
        test_pred = final_model.predict(X_test)
        
        # Inverse transform for actual values
        train_pred_actual = self.scaler_y.inverse_transform(train_pred.reshape(-1, 1)).ravel()
        val_pred_actual = self.scaler_y.inverse_transform(val_pred.reshape(-1, 1)).ravel()
        test_pred_actual = self.scaler_y.inverse_transform(test_pred.reshape(-1, 1)).ravel()
        
        end_time = time.time()
        self.training_time['random_forest'] = end_time - start_time
        
        # Store CV results
        self.cv_results['random_forest'] = {
            'best_score': -rf_random.best_score_,
            'cv_results': rf_random.cv_results_
        }
        
        # Calculate metrics
        self.metrics['random_forest'] = self._calculate_metrics(
            train_pred_actual, val_pred_actual, test_pred_actual,
            self.y_train_orig, self.y_val_orig, self.y_test_orig
        )
        
        self.models['random_forest'] = final_model
        
        # Store feature importance
        self.feature_importance = {
            'random_forest': final_model.feature_importances_
        }
        
        return final_model
    
    def train_ann(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Train Neural Network with optimized hyperparameters
        """
        print("\n" + "-"*40)
        print("Training Neural Network Regressor...")
        start_time = time.time()
        
        # Define hyperparameter grid for ANN
        param_dist = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': uniform(0.0001, 0.01),  # L2 regularization
            'learning_rate_init': uniform(0.0001, 0.01),
            'batch_size': [32, 64, 128],
            'max_iter': [500, 1000]
        }
        
        # Create ANN model
        ann = MLPRegressor(random_state=self.random_state, early_stopping=True, 
                          validation_fraction=0.1, n_iter_no_change=20)
        
        # Use RandomizedSearchCV for faster hyperparameter tuning
        print("Performing randomized hyperparameter search...")
        ann_random = RandomizedSearchCV(
            estimator=ann,
            param_distributions=param_dist,
            n_iter=20,  # Number of parameter settings sampled
            cv=5,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the random search model
        ann_random.fit(X_train, y_train)
        
        # Get best parameters
        self.best_params['ann'] = ann_random.best_params_
        print(f"\nBest ANN parameters:")
        for param, value in ann_random.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV Score: {-ann_random.best_score_:.4f} MSE")
        
        # Train final model with best parameters
        final_model = MLPRegressor(
            **ann_random.best_params_,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        final_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = final_model.predict(X_train)
        val_pred = final_model.predict(X_val)
        test_pred = final_model.predict(X_test)
        
        # Inverse transform for actual values
        train_pred_actual = self.scaler_y.inverse_transform(train_pred.reshape(-1, 1)).ravel()
        val_pred_actual = self.scaler_y.inverse_transform(val_pred.reshape(-1, 1)).ravel()
        test_pred_actual = self.scaler_y.inverse_transform(test_pred.reshape(-1, 1)).ravel()
        
        end_time = time.time()
        self.training_time['ann'] = end_time - start_time
        
        # Store CV results
        self.cv_results['ann'] = {
            'best_score': -ann_random.best_score_,
            'cv_results': ann_random.cv_results_
        }
        
        # Calculate metrics
        self.metrics['ann'] = self._calculate_metrics(
            train_pred_actual, val_pred_actual, test_pred_actual,
            self.y_train_orig, self.y_val_orig, self.y_test_orig
        )
        
        self.models['ann'] = final_model
        
        return final_model
    
    def _calculate_metrics(self, train_pred, val_pred, test_pred, y_train, y_val, y_test):
        """
        Calculate comprehensive metrics for model evaluation
        """
        metrics = {}
        
        for dataset_name, pred, actual in [
            ('train', train_pred, y_train),
            ('validation', val_pred, y_val),
            ('test', test_pred, y_test)
        ]:
            # Convert to numpy arrays if needed
            pred = np.array(pred).ravel()
            actual = np.array(actual).ravel()
            
            mse = mean_squared_error(actual, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            variance = np.var(actual - pred)
            
            metrics[dataset_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'variance': variance
            }
        
        return metrics
    
    def plot_results(self):
        """
        Plot model predictions and comparisons
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Feature Importance for Random Forest
        ax = axes[0, 0]
        if hasattr(self, 'feature_importance'):
            features = ['Voltage', 'Current', 'Temp', 'Humidity']
            importances = self.feature_importance['random_forest']
            ax.barh(features, importances, color='skyblue', edgecolor='black')
            ax.set_xlabel('Importance')
            ax.set_title('Random Forest Feature Importance')
            ax.grid(True, alpha=0.3)
        
        # Plot 2-3: Predictions vs Actual
        model_colors = {'random_forest': 'green', 'ann': 'red'}
        model_names = {'random_forest': 'Random Forest', 'ann': 'Neural Network'}
        
        for idx, (model_type, color) in enumerate(model_colors.items()):
            if model_type in self.models:
                ax = axes[0, idx+1]
                pred = self.test_predictions[model_type]
                actual = self.test_actual
                
                ax.scatter(actual, pred, alpha=0.6, c=color, edgecolors='black', linewidth=0.5)
                ax.plot([actual.min(), actual.max()], 
                       [actual.min(), actual.max()], 'k--', linewidth=2, label='Perfect Prediction')
                ax.set_xlabel('Actual Demand')
                ax.set_ylabel('Predicted Demand')
                ax.set_title(f'{model_names[model_type]} - Test Set')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add R² text
                r2 = self.metrics[model_type]['test']['r2']
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot 4: Training Time Comparison
        ax = axes[1, 0]
        models = list(self.training_time.keys())
        times = [self.training_time[m] for m in models]
        model_names_display = ['Random Forest' if m == 'random_forest' else 'Neural Network' 
                               for m in models]
        
        bars = ax.bar(model_names_display, times, color=['green', 'red'], edgecolor='black')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Model Training Time Comparison')
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{time_val:.1f}s', ha='center', va='bottom')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: RMSE Comparison
        ax = axes[1, 1]
        x_pos = np.arange(len(models))
        width = 0.25
        
        train_rmse = [self.metrics[m]['train']['rmse'] for m in models]
        val_rmse = [self.metrics[m]['validation']['rmse'] for m in models]
        test_rmse = [self.metrics[m]['test']['rmse'] for m in models]
        
        ax.bar(x_pos - width, train_rmse, width, label='Train', alpha=0.7)
        ax.bar(x_pos, val_rmse, width, label='Validation', alpha=0.7)
        ax.bar(x_pos + width, test_rmse, width, label='Test', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Random Forest', 'Neural Network'])
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: R² Comparison
        ax = axes[1, 2]
        
        train_r2 = [self.metrics[m]['train']['r2'] for m in models]
        val_r2 = [self.metrics[m]['validation']['r2'] for m in models]
        test_r2 = [self.metrics[m]['test']['r2'] for m in models]
        
        ax.bar(x_pos - width, train_r2, width, label='Train', alpha=0.7)
        ax.bar(x_pos, val_r2, width, label='Validation', alpha=0.7)
        ax.bar(x_pos + width, test_r2, width, label='Test', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Random Forest', 'Neural Network'])
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results_summary(self):
        """
        Print comprehensive results summary
        """
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        
        # Convergence Speed (time per model)
        print("\n" + "="*40)
        print("CONVERGENCE SPEED (Training Time)")
        print("="*40)
        for model_type, train_time in self.training_time.items():
            model_name = 'Random Forest' if model_type == 'random_forest' else 'Neural Network'
            print(f"{model_name}: {train_time:.2f} seconds")
        
        # Final Cost (Test MSE)
        print("\n" + "="*40)
        print("FINAL COST (Test MSE)")
        print("="*40)
        for model_type in self.metrics.keys():
            model_name = 'Random Forest' if model_type == 'random_forest' else 'Neural Network'
            test_mse = self.metrics[model_type]['test']['mse']
            print(f"{model_name}: {test_mse:.4f}")
        
        # Variance of errors
        print("\n" + "="*40)
        print("VARIANCE OF ERRORS")
        print("="*40)
        for model_type in self.metrics.keys():
            model_name = 'Random Forest' if model_type == 'random_forest' else 'Neural Network'
            for dataset in ['train', 'validation', 'test']:
                variance = self.metrics[model_type][dataset]['variance']
                print(f"{model_name} - {dataset}: {variance:.4f}")
        
        # Runtime
        print("\n" + "="*40)
        print("RUNTIME")
        print("="*40)
        total_time = sum(self.training_time.values())
        print(f"Total training time: {total_time:.2f} seconds")
        
        # Detailed metrics table
        print("\n" + "="*70)
        print("DETAILED PERFORMANCE METRICS")
        print("="*70)
        
        for model_type in self.metrics.keys():
            model_name = 'Random Forest' if model_type == 'random_forest' else 'Neural Network'
            print(f"\n{model_name}:")
            print("-" * 60)
            print(f"{'Dataset':<12} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Variance':<12}")
            print("-" * 60)
            
            for dataset in ['train', 'validation', 'test']:
                m = self.metrics[model_type][dataset]
                print(f"{dataset:<12} {m['mse']:<12.4f} {m['rmse']:<12.4f} "
                      f"{m['mae']:<12.4f} {m['r2']:<12.4f} {m['variance']:<12.4f}")
    
    def save_models(self, filepath='models/'):
        """
        Save trained models and scalers
        """
        import os
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{filepath}{name}_model.pkl")
        
        # Save scalers
        joblib.dump(self.scaler_X, f"{filepath}scaler_X.pkl")
        joblib.dump(self.scaler_y, f"{filepath}scaler_y.pkl")
        
        # Save metrics and configuration
        config = {
            'metrics': self.metrics,
            'best_params': self.best_params,
            'training_time': self.training_time,
            'cv_results': self.cv_results,
            'random_state': self.random_state
        }
        joblib.dump(config, f"{filepath}config.pkl")
        
        print(f"\nModels and scalers saved to {filepath}")
    
    def train_all_models(self, filepath):
        """
        Complete training pipeline
        """
        overall_start = time.time()
        
        # Load and preprocess data
        (X_train, X_val, X_test, 
         y_train_scaled, y_val_scaled, y_test_scaled,
         X_train_orig, X_val_orig, X_test_orig) = self.load_and_preprocess_data(filepath)
        
        # Store test actual for plotting
        self.test_actual = self.y_test_orig
        self.test_predictions = {}
        
        # Train Random Forest model
        rf_model = self.train_random_forest(
            X_train, y_train_scaled, X_val, y_val_scaled, X_test, y_test_scaled
        )
        self.test_predictions['random_forest'] = self.scaler_y.inverse_transform(
            rf_model.predict(X_test).reshape(-1, 1)
        ).ravel()
        
        # Train ANN model
        ann_model = self.train_ann(
            X_train, y_train_scaled, X_val, y_val_scaled, X_test, y_test_scaled
        )
        self.test_predictions['ann'] = self.scaler_y.inverse_transform(
            ann_model.predict(X_test).reshape(-1, 1)
        ).ravel()
        
        overall_end = time.time()
        print(f"\nTotal training time: {overall_end - overall_start:.2f} seconds")
        
        # Print results summary
        self.print_results_summary()
        
        # Plot results
        self.plot_results()
        
        return self.models

# Run training if script is executed directly
if __name__ == "__main__":
    # Initialize predictor
    predictor = EnergyDemandPredictor(random_state=42)
    
    # Train all models
    models = predictor.train_all_models('DATASET.csv')
    
    # Save models
    predictor.save_models('trained_models/')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)