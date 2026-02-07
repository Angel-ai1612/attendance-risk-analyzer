"""
Model Training for Smart Attendance Risk Analyzer
Train XGBoost model to predict attendance risk.
"""

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AttendanceRiskModel:
    """Train and evaluate attendance risk prediction model."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize with config file."""
        config_file = Path(config_path)
        if not config_file.exists():
            script_dir = Path(__file__).parent.parent
            config_file = script_dir / 'config.yaml'
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        logger.info(f"Initialized model trainer")
    
    def load_features(self):
        """Load engineered features."""
        features_path = Path(self.config['output']['paths']['data']) / 'features.csv'
        
        if not features_path.exists():
            logger.error(f"❌ Features not found at {features_path}")
            logger.error("Run feature_engineering.py first!")
            import sys
            sys.exit(1)
        
        df = pd.read_csv(features_path)
        logger.info(f"✅ Loaded features for {len(df):,} students")
        
        return df
    
    def prepare_data(self, df):
        """Prepare features and target for training.
        
        Removes attendance_rate features to prevent data leakage.
        Uses only pattern-based features (trends, streaks, consistency).
        """
        # Feature columns - EXCLUDE attendance rates (they leak the target)
        # We keep ONLY behavioral patterns and trends
        feature_cols = [
            'consistency_score',
            'absence_streak_max', 
            'absence_streak_current',
            'trend_slope',
            'weekend_vs_weekday_ratio',
            'early_vs_late_month_ratio',
            'days_since_last_absence',
            'total_sessions_missed',
            'has_multiple_risk_factors'
        ]
        
        X = df[feature_cols]
        y = df['future_at_risk']  # Predict FUTURE risk, not current state
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Using ONLY pattern-based features (no attendance rates)")
        logger.info(f"Target: Future risk (14 days ahead)")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def split_data(self, X, y):
        """Split data into train/test sets."""
        test_size = self.model_config['test_size']
        random_state = self.model_config['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        xgb_params = self.model_config['xgboost']
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        
        logger.info("✅ XGBoost training complete")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, feature_cols):
        """Evaluate model performance."""
        logger.info("\nEvaluating model performance...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return metrics, y_pred, y_pred_proba, feature_importance
    
    def cross_validate(self, model, X, y):
        """Perform cross-validation."""
        logger.info("\nPerforming 5-fold cross-validation...")
        
        cv_scores = cross_val_score(
            model, X, y, cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        logger.info(f"CV ROC-AUC scores: {cv_scores}")
        logger.info(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_scores
    
    def save_model(self, model, metrics, feature_importance):
        """Save trained model and metadata."""
        models_dir = Path(self.config['output']['paths']['models'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = models_dir / 'xgboost_baseline.pkl'
        joblib.dump(model, model_path)
        logger.info(f"✅ Saved model to {model_path}")
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'trained_at': datetime.now().isoformat(),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'feature_importance': feature_importance.to_dict('records'),
            'config': self.model_config['xgboost']
        }
        
        metadata_path = models_dir / 'model_metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"✅ Saved model metadata to {metadata_path}")
        
        return model_path


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("SMART ATTENDANCE RISK ANALYZER - Model Training")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = AttendanceRiskModel()
    
    # Load features
    df = trainer.load_features()
    
    # Prepare data
    X, y, feature_cols = trainer.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train model
    model = trainer.train_xgboost(X_train, y_train)
    
    # Evaluate
    metrics, y_pred, y_pred_proba, feature_importance = trainer.evaluate_model(
        model, X_test, y_test, feature_cols
    )
    
    # Cross-validation
    cv_scores = trainer.cross_validate(model, X, y)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("MODEL PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    logger.info(f"Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"Recall:    {metrics['recall']*100:.2f}%")
    logger.info(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TOP 5 FEATURES")
    logger.info("=" * 60)
    print(feature_importance.head())
    
    logger.info("\n" + "=" * 60)
    logger.info("CONFUSION MATRIX")
    logger.info("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{cm}")
    logger.info(f"\nTrue Negatives:  {cm[0][0]:,}")
    logger.info(f"False Positives: {cm[0][1]:,}")
    logger.info(f"False Negatives: {cm[1][0]:,}")
    logger.info(f"True Positives:  {cm[1][1]:,}")
    
    # Save model
    model_path = trainer.save_model(model, metrics, feature_importance)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ MODEL TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\n📁 Model saved to: {model_path}")
    logger.info(f"🎯 Test Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"🎯 ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"\n🚀 Next step: python src/model_evaluation.py")


if __name__ == '__main__':
    main()