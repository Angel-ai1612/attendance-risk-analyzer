"""
Data Generator for Smart Attendance Risk Analyzer
Generates realistic student attendance data with multiple student types and patterns.
"""

import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AttendanceDataGenerator:
    """Generate realistic student attendance data with configurable parameters."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize generator with config file."""
        # Convert to Path object and resolve to absolute path
        config_file = Path(config_path)
        
        # If not found, try looking in parent directory (for when running from src/)
        if not config_file.exists():
            script_dir = Path(__file__).parent.parent  # Go up from src/ to root
            config_file = script_dir / 'config.yaml'
            
        if not config_file.exists():
            logger.error(f"❌ Config file not found!")
            logger.error(f"Looked in: {Path(config_path).absolute()}")
            logger.error(f"Also tried: {config_file.absolute()}")
            logger.error(f"Current directory: {Path.cwd()}")
            logger.error("\nMake sure config.yaml is in your project root!")
            import sys
            sys.exit(1)
        
        logger.info(f"Loading config from: {config_file.absolute()}")
        
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            if self.config is None:
                logger.error("❌ Config file is empty or invalid YAML!")
                import sys
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"❌ Error reading config: {e}")
            import sys
            sys.exit(1)
        
        self.num_students = self.config['data']['num_students']
        self.num_days = self.config['data']['num_days']
        self.start_date = datetime.strptime(self.config['data']['start_date'], '%Y-%m-%d')
        
        self.student_types = self.config['data']['student_types']
        self.attendance_probs = self.config['data']['attendance_probabilities']
        
        logger.info(f"✅ Config loaded successfully")
        logger.info(f"Initialized generator: {self.num_students} students, {self.num_days} days")
    
    def generate_student_profiles(self):
        """Generate student metadata (ID, type, department, year)."""
        np.random.seed(42)
        
        # Assign student types based on distribution
        type_choices = list(self.student_types.keys())
        type_probs = list(self.student_types.values())
        student_types_assigned = np.random.choice(
            type_choices, 
            size=self.num_students, 
            p=type_probs
        )
        
        # Assign departments and years
        departments = np.random.choice(
            self.config['data']['departments'], 
            size=self.num_students
        )
        years = np.random.choice(
            self.config['data']['years'], 
            size=self.num_students
        )
        
        profiles = pd.DataFrame({
            'student_id': [f'STU_{i:05d}' for i in range(1, self.num_students + 1)],
            'student_type': student_types_assigned,
            'department': departments,
            'year': years
        })
        
        logger.info(f"Generated {len(profiles)} student profiles")
        logger.info(f"Type distribution:\n{profiles['student_type'].value_counts()}")
        
        return profiles
    
    def generate_attendance_for_student(self, student_type):
        """Generate daily attendance (0/1) for a single student based on their type."""
        # Get attendance probability range for this student type
        prob_min, prob_max = self.attendance_probs[student_type]
        
        # Sample base probability for this student (within type's range)
        base_prob = np.random.uniform(prob_min, prob_max)
        
        # Generate daily attendance with some temporal variation
        attendance = []
        for day in range(self.num_days):
            # Add temporal patterns
            day_of_week = (self.start_date + timedelta(days=day)).weekday()
            
            # Lower attendance on Mondays/Fridays for some types
            prob_adjusted = base_prob
            if student_type in ['struggling', 'erratic']:
                if day_of_week == 0:  # Monday
                    prob_adjusted *= 0.85
                elif day_of_week == 4:  # Friday
                    prob_adjusted *= 0.90
            
            # Add random noise
            prob_adjusted += np.random.normal(0, 0.05)
            prob_adjusted = np.clip(prob_adjusted, 0, 1)
            
            # Generate attendance (1 = present, 0 = absent)
            attended = 1 if np.random.random() < prob_adjusted else 0
            attendance.append(attended)
        
        return attendance
    
    def generate_full_dataset(self):
        """Generate complete dataset with all students and attendance records."""
        # Generate student profiles
        profiles = self.generate_student_profiles()
        
        # Generate date range
        dates = pd.date_range(
            start=self.start_date, 
            periods=self.num_days, 
            freq='D'
        )
        
        # Generate attendance for each student
        logger.info("Generating attendance data...")
        all_records = []
        
        for idx, row in profiles.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Progress: {idx}/{len(profiles)} students")
            
            student_id = row['student_id']
            student_type = row['student_type']
            
            # Generate attendance sequence
            attendance = self.generate_attendance_for_student(student_type)
            
            # Create records for each day
            for day_idx, date in enumerate(dates):
                all_records.append({
                    'student_id': student_id,
                    'date': date,
                    'attended': attendance[day_idx],
                    'student_type': student_type,
                    'department': row['department'],
                    'year': row['year']
                })
        
        df = pd.DataFrame(all_records)
        logger.info(f"✅ Generated {len(df):,} attendance records")
        
        return df, profiles
    
    def add_label_noise(self, df, noise_rate=0.10):
        """Add label noise for realism (flip some attendance labels)."""
        df = df.copy()
        
        # Randomly flip some labels
        num_to_flip = int(len(df) * noise_rate)
        flip_indices = np.random.choice(df.index, size=num_to_flip, replace=False)
        
        df.loc[flip_indices, 'attended'] = 1 - df.loc[flip_indices, 'attended']
        
        logger.info(f"Added label noise: flipped {num_to_flip} records ({noise_rate*100:.1f}%)")
        
        return df
    
    def save_data(self, df, profiles):
        """Save generated data to files."""
        # Create output directory
        output_dir = Path(self.config['output']['paths']['data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        data_path = output_dir / 'attendance_data.csv'
        df.to_csv(data_path, index=False)
        logger.info(f"✅ Saved attendance data to {data_path}")
        
        # Save student profiles
        profiles_path = output_dir / 'student_profiles.csv'
        profiles.to_csv(profiles_path, index=False)
        logger.info(f"✅ Saved student profiles to {profiles_path}")
        
        # Save metadata
        metadata = {
            'num_students': self.num_students,
            'num_days': self.num_days,
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'overall_attendance_rate': f"{df['attended'].mean()*100:.2f}%",
            'generated_at': datetime.now().isoformat()
        }
        
        metadata_path = output_dir / 'metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        logger.info(f"✅ Saved metadata to {metadata_path}")
        
        return data_path, profiles_path


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("SMART ATTENDANCE RISK ANALYZER - Data Generation")
    logger.info("=" * 60)
    
    # Initialize generator
    generator = AttendanceDataGenerator()
    
    # Generate data
    df, profiles = generator.generate_full_dataset()
    
    # Add realism (label noise)
    noise_rate = generator.config['data']['noise']['label_flip_rate']
    df = generator.add_label_noise(df, noise_rate=noise_rate)
    
    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total students: {profiles['student_id'].nunique():,}")
    logger.info(f"Total days: {df['date'].nunique()}")
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Overall attendance rate: {df['attended'].mean()*100:.2f}%")
    logger.info("\nAttendance by student type:")
    type_summary = df.groupby('student_type')['attended'].agg(['mean', 'count'])
    type_summary['mean'] = type_summary['mean'] * 100
    type_summary.columns = ['Attendance %', 'Records']
    print(type_summary)
    
    # Save data
    data_path, profiles_path = generator.save_data(df, profiles)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ DATA GENERATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\n📁 Files saved:")
    logger.info(f"   - {data_path}")
    logger.info(f"   - {profiles_path}")
    logger.info(f"\n🚀 Next step: python src/feature_engineering.py")


if __name__ == '__main__':
    main()