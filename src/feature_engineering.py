"""
Feature Engineering for Smart Attendance Risk Analyzer
Calculates 12 risk features from raw attendance data.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AttendanceFeatureEngineer:
    """Calculate attendance risk features from raw data."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize with config file."""
        config_file = Path(config_path)
        if not config_file.exists():
            script_dir = Path(__file__).parent.parent
            config_file = script_dir / 'config.yaml'
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.windows = self.config['features']['windows']
        logger.info(f"Initialized feature engineer with windows: {self.windows}")
    
    def load_data(self):
        """Load generated attendance data."""
        data_path = Path(self.config['output']['paths']['data']) / 'attendance_data.csv'
        
        if not data_path.exists():
            logger.error(f"❌ Data not found at {data_path}")
            logger.error("Run data_generator.py first!")
            import sys
            sys.exit(1)
        
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"✅ Loaded {len(df):,} attendance records")
        return df
    
    def calculate_rolling_attendance(self, student_data, window):
        """Calculate rolling attendance rate for a given window."""
        if len(student_data) < window:
            return student_data['attended'].mean()
        
        return student_data['attended'].rolling(window=window, min_periods=1).mean().iloc[-1]
    
    def calculate_consistency_score(self, student_data):
        """Calculate consistency (inverse of standard deviation)."""
        std = student_data['attended'].std()
        return 1 - std if std < 1 else 0
    
    def calculate_absence_streaks(self, student_data):
        """Find longest and current absence streaks."""
        absences = (student_data['attended'] == 0).astype(int)
        
        # Find all streaks
        streaks = []
        current_streak = 0
        
        for val in absences:
            if val == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # Add final streak if exists
        if current_streak > 0:
            streaks.append(current_streak)
        
        max_streak = max(streaks) if streaks else 0
        current_absence_streak = current_streak
        
        return max_streak, current_absence_streak
    
    def calculate_trend_slope(self, student_data, window=14):
        """Calculate trend (improving/declining) using linear regression."""
        if len(student_data) < window:
            return 0
        
        recent = student_data.tail(window)
        x = np.arange(len(recent))
        y = recent['attended'].values
        
        # Simple linear regression slope
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0
    
    def calculate_weekend_ratio(self, student_data):
        """Calculate weekend vs weekday attendance ratio."""
        student_data['day_of_week'] = student_data['date'].dt.dayofweek
        
        weekend = student_data[student_data['day_of_week'].isin([5, 6])]
        weekday = student_data[~student_data['day_of_week'].isin([5, 6])]
        
        weekend_rate = weekend['attended'].mean() if len(weekend) > 0 else 0
        weekday_rate = weekday['attended'].mean() if len(weekday) > 0 else 0
        
        if weekday_rate > 0:
            return weekend_rate / weekday_rate
        return 1.0
    
    def calculate_month_ratio(self, student_data):
        """Calculate early-month vs late-month attendance ratio."""
        student_data['day_of_month'] = student_data['date'].dt.day
        
        early = student_data[student_data['day_of_month'] <= 15]
        late = student_data[student_data['day_of_month'] > 15]
        
        early_rate = early['attended'].mean() if len(early) > 0 else 0
        late_rate = late['attended'].mean() if len(late) > 0 else 0
        
        if late_rate > 0:
            return early_rate / late_rate
        return 1.0
    
    def calculate_features_for_student(self, student_data):
        """Calculate all features for a single student.
        
        Uses ONLY first 60 days to predict risk in days 75-90 (future prediction).
        This prevents data leakage and creates a real prediction problem.
        """
        # Use only first 60 days for features (simulate "we're at day 60")
        train_window = student_data.head(60).copy()
        
        # Calculate FUTURE attendance (days 75-90) as target
        future_window = student_data.iloc[75:90] if len(student_data) >= 90 else student_data.tail(15)
        future_attendance_rate = future_window['attended'].mean()
        
        features = {}
        
        # 1-3: Rolling attendance rates (from first 60 days only)
        features['attendance_rate_7d'] = self.calculate_rolling_attendance(train_window, self.windows['short'])
        features['attendance_rate_14d'] = self.calculate_rolling_attendance(train_window, self.windows['medium'])
        features['attendance_rate_30d'] = self.calculate_rolling_attendance(train_window, self.windows['long'])
        
        # 4: Consistency score
        features['consistency_score'] = self.calculate_consistency_score(train_window)
        
        # 5-6: Absence streaks
        max_streak, current_streak = self.calculate_absence_streaks(train_window)
        features['absence_streak_max'] = max_streak
        features['absence_streak_current'] = current_streak
        
        # 7: Trend
        features['trend_slope'] = self.calculate_trend_slope(train_window)
        
        # 8-9: Behavioral ratios
        features['weekend_vs_weekday_ratio'] = self.calculate_weekend_ratio(train_window)
        features['early_vs_late_month_ratio'] = self.calculate_month_ratio(train_window)
        
        # 10: Days since last absence
        last_absence_idx = train_window[train_window['attended'] == 0].index
        if len(last_absence_idx) > 0:
            features['days_since_last_absence'] = len(train_window) - last_absence_idx[-1] - 1
        else:
            features['days_since_last_absence'] = len(train_window)
        
        # 11: Total sessions missed (in first 60 days)
        features['total_sessions_missed'] = (train_window['attended'] == 0).sum()
        
        # 12: Multiple risk factors flag
        has_multiple_risks = (
            features['attendance_rate_30d'] < 0.75 and
            features['absence_streak_max'] > 3 and
            features['consistency_score'] < 0.5
        )
        features['has_multiple_risk_factors'] = 1 if has_multiple_risks else 0
        
        # TARGET: Future risk (based on days 75-90, not current state)
        features['future_attendance_rate'] = future_attendance_rate
        features['future_at_risk'] = 1 if future_attendance_rate < 0.75 else 0
        
        return features
    
    def generate_features(self, df):
        """Generate features for all students."""
        logger.info("Calculating features for all students...")
        
        all_features = []
        student_ids = df['student_id'].unique()
        
        for idx, student_id in enumerate(student_ids):
            if idx % 1000 == 0:
                logger.info(f"Progress: {idx}/{len(student_ids)} students")
            
            student_data = df[df['student_id'] == student_id].sort_values('date').reset_index(drop=True)
            
            features = self.calculate_features_for_student(student_data)
            features['student_id'] = student_id
            
            # Add metadata
            features['student_type'] = student_data['student_type'].iloc[0]
            features['department'] = student_data['department'].iloc[0]
            features['year'] = student_data['year'].iloc[0]
            
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        logger.info(f"✅ Generated features for {len(features_df):,} students")
        
        return features_df
    
    def save_features(self, features_df):
        """Save feature matrix to file."""
        output_dir = Path(self.config['output']['paths']['data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'features.csv'
        features_df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Saved features to {output_path}")
        
        # Save feature summary
        summary = {
            'num_students': len(features_df),
            'num_features': len([c for c in features_df.columns if c not in ['student_id', 'student_type', 'department', 'year', 'future_attendance_rate', 'future_at_risk']]),
            'future_at_risk_count': int(features_df['future_at_risk'].sum()),
            'future_at_risk_percentage': f"{features_df['future_at_risk'].mean()*100:.2f}%",
            'prediction_horizon': '14 days ahead (using days 1-60 to predict days 75-90)',
            'feature_names': [c for c in features_df.columns if c not in ['student_id', 'student_type', 'department', 'year']],
            'generated_at': datetime.now().isoformat()
        }
        
        summary_path = output_dir / 'features_metadata.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"✅ Saved feature metadata to {summary_path}")
        
        return output_path


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("SMART ATTENDANCE RISK ANALYZER - Feature Engineering")
    logger.info("=" * 60)
    
    # Initialize feature engineer
    engineer = AttendanceFeatureEngineer()
    
    # Load data
    df = engineer.load_data()
    
    # Generate features
    features_df = engineer.generate_features(df)
    
    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total students: {len(features_df):,}")
    logger.info(f"Students at FUTURE risk (<75% in days 75-90): {features_df['future_at_risk'].sum():,} ({features_df['future_at_risk'].mean()*100:.1f}%)")
    logger.info(f"Average FUTURE attendance rate: {features_df['future_attendance_rate'].mean()*100:.2f}%")
    logger.info("\nFeature statistics:")
    print(features_df[['attendance_rate_7d', 'attendance_rate_14d', 'attendance_rate_30d', 
                       'consistency_score', 'absence_streak_max']].describe())
    
    # Save features
    output_path = engineer.save_features(features_df)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ FEATURE ENGINEERING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\n📁 Features saved to: {output_path}")
    logger.info(f"\n🚀 Next step: python src/model_training.py")


if __name__ == '__main__':
    main()