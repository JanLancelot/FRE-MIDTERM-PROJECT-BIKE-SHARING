import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

class BikeDataCurator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.main_data = None
        self.cleaned_data = None
        self.correlation_data = None
        self.descriptive_data = None
        self.predictive_data = None
        self.preprocessing_report = []
    
    # Data Loading
    def load_data(self):
        try:
            self.main_data = pd.read_csv(self.filepath)
            print(f"✓ Data loaded successfully: {len(self.main_data)} records")
            
            print("\n=== BASIC DATASET INFORMATION ===")
            print(f"Dataset shape: {self.main_data.shape}")
            print(f"Rows: {self.main_data.shape[0]}, Columns: {self.main_data.shape[1]}")
            
            print("\n=== First 5 Rows (head) ===")
            print(self.main_data.head())
            
            print("\n=== Dataset Info ===")
            print(self.main_data.info())
            
            print("\n=== Data Types ===")
            print(self.main_data.dtypes)
            
            self.preprocessing_report.append("✓ Data Loading: Loaded dataset with {} records and {} columns".format(
                self.main_data.shape[0], self.main_data.shape[1]))
            
            return True
        except FileNotFoundError:
            print(f"✗ Error: File '{self.filepath}' not found")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    # Comprehensive Pre-processing
    def comprehensive_data_cleaning(self, output_path='Cleaned_Dataset.csv'):
        if self.main_data is None:
            print("✗ Please load data first")
            return False
        
        print("\n" + "="*60)
        print("COMPREHENSIVE DATA PREPROCESSING")
        print("="*60)
        
        self.cleaned_data = self.main_data.copy()
        initial_shape = self.cleaned_data.shape
        
        print("\n=== STEP 1: Handling Missing Values ===")
        missing_before = self.cleaned_data.isnull().sum()
        total_missing = missing_before.sum()
        
        if total_missing > 0:
            print(f"Found {total_missing} missing values across {(missing_before > 0).sum()} columns:")
            print(missing_before[missing_before > 0])
            
            critical_cols = ['cnt', 'temp', 'hum', 'windspeed']
            for col in critical_cols:
                if col in self.cleaned_data.columns:
                    rows_before = len(self.cleaned_data)
                    self.cleaned_data = self.cleaned_data.dropna(subset=[col])
                    rows_removed = rows_before - len(self.cleaned_data)
                    if rows_removed > 0:
                        print(f"  Removed {rows_removed} rows with missing {col}")
            
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.cleaned_data[col].isnull().any():
                    median_val = self.cleaned_data[col].median()
                    self.cleaned_data[col].fillna(median_val, inplace=True)
                    print(f"  Filled missing {col} with median: {median_val:.2f}")
            
            self.preprocessing_report.append(f"✓ Missing Values: Handled {total_missing} missing values")
        else:
            print("✓ No missing values found")
            self.preprocessing_report.append("✓ Missing Values: No missing values detected")
        
        print("\n=== STEP 2: Removing Duplicates ===")
        duplicates_before = self.cleaned_data.duplicated().sum()
        if duplicates_before > 0:
            self.cleaned_data = self.cleaned_data.drop_duplicates()
            print(f"✓ Removed {duplicates_before} duplicate rows")
            self.preprocessing_report.append(f"✓ Duplicates: Removed {duplicates_before} duplicate rows")
        else:
            print("✓ No duplicate rows found")
            self.preprocessing_report.append("✓ Duplicates: No duplicates detected")
        
        print("\n=== STEP 3: Detecting and Handling Outliers ===")
        outliers_handled = 0
        for col in ['cnt', 'casual', 'registered']:
            if col in self.cleaned_data.columns:
                Q1 = self.cleaned_data[col].quantile(0.25)
                Q3 = self.cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = self.cleaned_data[(self.cleaned_data[col] < lower_bound) | 
                                            (self.cleaned_data[col] > upper_bound)]
                
                if len(outliers) > 0:
                    print(f"  Found {len(outliers)} extreme outliers in {col}")
                    print(f"    Bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
                    self.cleaned_data[col] = self.cleaned_data[col].clip(lower_bound, upper_bound)
                    outliers_handled += len(outliers)
                    print(f"    ✓ Capped outliers to boundary values")
        
        if outliers_handled > 0:
            self.preprocessing_report.append(f"✓ Outliers: Handled {outliers_handled} extreme outliers using capping method")
        else:
            print("✓ No extreme outliers requiring treatment")
            self.preprocessing_report.append("✓ Outliers: No extreme outliers detected")
        
        print("\n=== STEP 4: Normalizing/Standardizing Variables ===")
        normalized_vars = ['temp', 'atemp', 'hum', 'windspeed']
        normalization_needed = False
        
        for var in normalized_vars:
            if var in self.cleaned_data.columns:
                min_val = self.cleaned_data[var].min()
                max_val = self.cleaned_data[var].max()
                if min_val < 0 or max_val > 1:
                    normalization_needed = True
                    print(f"  {var}: Range [{min_val:.2f}, {max_val:.2f}] - Needs normalization")
                else:
                    print(f"  {var}: Already normalized (range: [{min_val:.2f}, {max_val:.2f}])")
        
        if normalization_needed:
            scaler = StandardScaler()
            for var in normalized_vars:
                if var in self.cleaned_data.columns:
                    if self.cleaned_data[var].min() < 0 or self.cleaned_data[var].max() > 1:
                        original_values = self.cleaned_data[var].values.reshape(-1, 1)
                        self.cleaned_data[var] = scaler.fit_transform(original_values).flatten()
                        print(f"    ✓ Normalized {var}")
            self.preprocessing_report.append("✓ Normalization: Applied normalization to weather variables")
        else:
            print("✓ All weather variables are already normalized")
            self.preprocessing_report.append("✓ Normalization: Weather variables already in normalized form (0-1 range)")
        
        print("\n=== STEP 5: Converting Data Types ===")
        type_conversions = 0
        
        int_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 
                      'workingday', 'weathersit', 'cnt', 'casual', 'registered']
        for col in int_columns:
            if col in self.cleaned_data.columns:
                if self.cleaned_data[col].dtype != 'int64':
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int64')
                    type_conversions += 1
                    print(f"  Converted {col} to integer")
        
        float_columns = ['temp', 'atemp', 'hum', 'windspeed']
        for col in float_columns:
            if col in self.cleaned_data.columns:
                if self.cleaned_data[col].dtype != 'float64':
                    self.cleaned_data[col] = self.cleaned_data[col].astype('float64')
                    type_conversions += 1
                    print(f"  Converted {col} to float")
        
        if type_conversions > 0:
            print(f"✓ Converted {type_conversions} columns to appropriate types")
            self.preprocessing_report.append(f"✓ Data Types: Converted {type_conversions} columns to appropriate types")
        else:
            print("✓ All columns already have appropriate data types")
            self.preprocessing_report.append("✓ Data Types: All columns already have appropriate types")
        
        print("\n=== STEP 6: Renaming Columns ===")
        column_mapping = {
            'yr': 'year',
            'mnth': 'month',
            'hr': 'hour',
            'hum': 'humidity',
            'cnt': 'total_rentals',
            'casual': 'casual_users',
            'registered': 'registered_users',
            'weathersit': 'weather_situation',
            'atemp': 'feeling_temperature'
        }
        
        renamed_count = 0
        for old_name, new_name in column_mapping.items():
            if old_name in self.cleaned_data.columns:
                self.cleaned_data.rename(columns={old_name: new_name}, inplace=True)
                renamed_count += 1
                print(f"  Renamed '{old_name}' → '{new_name}'")
        
        if renamed_count > 0:
            print(f"✓ Renamed {renamed_count} columns for better clarity")
            self.preprocessing_report.append(f"✓ Column Names: Renamed {renamed_count} columns for clarity")
        else:
            print("✓ No column renaming needed")
            self.preprocessing_report.append("✓ Column Names: No renaming required")
        
        print("\n=== STEP 7: Encoding Categorical Variables ===")
        
        if 'season' in self.cleaned_data.columns:
            season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
            self.cleaned_data['season_label'] = self.cleaned_data['season'].map(season_map)
            print("  Added 'season_label' column with text labels")
        
        if 'weather_situation' in self.cleaned_data.columns:
            weather_map = {
                1: 'Clear',
                2: 'Mist_Cloudy',
                3: 'Light_Rain_Snow',
                4: 'Heavy_Rain_Snow'
            }
            self.cleaned_data['weather_label'] = self.cleaned_data['weather_situation'].map(weather_map)
            print("  Added 'weather_label' column with text labels")
        
        binary_cols = ['holiday', 'workingday']
        for col in binary_cols:
            if col in self.cleaned_data.columns:
                unique_vals = self.cleaned_data[col].unique()
                if not set(unique_vals).issubset({0, 1}):
                    print(f"  Warning: {col} has non-binary values: {unique_vals}")
                else:
                    print(f"  ✓ {col} is properly binary encoded (0/1)")
        
        self.preprocessing_report.append("✓ Encoding: Added label columns for categorical variables while preserving numeric encoding")
        
        print("\n=== STEP 8: Feature Selection ===")
        print("  Keeping all features for comprehensive analysis")
        print(f"  Total features: {len(self.cleaned_data.columns)}")
        print(f"  Feature list: {list(self.cleaned_data.columns)}")
        self.preprocessing_report.append(f"✓ Features: Retained all {len(self.cleaned_data.columns)} features")
        
        print("\n=== STEP 9: Final Data Validation ===")
        validations_passed = True
        
        if 'hour' in self.cleaned_data.columns:
            invalid = self.cleaned_data[(self.cleaned_data['hour'] < 0) | (self.cleaned_data['hour'] > 23)]
            if len(invalid) > 0:
                print(f"  ⚠ Found {len(invalid)} invalid hour values")
                validations_passed = False
            else:
                print("  ✓ Hour values are valid (0-23)")
        
        if 'month' in self.cleaned_data.columns:
            invalid = self.cleaned_data[(self.cleaned_data['month'] < 1) | (self.cleaned_data['month'] > 12)]
            if len(invalid) > 0:
                print(f"  ⚠ Found {len(invalid)} invalid month values")
                validations_passed = False
            else:
                print("  ✓ Month values are valid (1-12)")
        
        if validations_passed:
            self.preprocessing_report.append("✓ Validation: All data ranges validated successfully")
        
        # Data Export
        print("\n=== Saving Cleaned Dataset ===")
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"✓ Cleaned dataset saved as: {output_path}")
        self.preprocessing_report.append(f"✓ Output: Saved cleaned dataset to {output_path}")
        
        final_shape = self.cleaned_data.shape
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Original shape:  {initial_shape}")
        print(f"Final shape:     {final_shape}")
        print(f"Rows removed:    {initial_shape[0] - final_shape[0]}")
        print(f"Columns added:   {final_shape[1] - initial_shape[1]}")
        print("\nPreprocessing Steps Completed:")
        for step in self.preprocessing_report:
            print(f"  {step}")
        print("="*60)
        
        return True
        
    def create_correlation_dataset(self, output_path='correlation.csv'):
        if self.cleaned_data is None:
            print("✗ Please run comprehensive_data_cleaning first")
            return False
        
        print("\n=== Creating Correlation Dataset ===")
        
        correlation_vars = [
            'temp',
            'feeling_temperature',
            'humidity',
            'windspeed',
            'season',
            'month',
            'hour',
            'weekday',
            'holiday',
            'workingday',
            'weather_situation',
            'casual_users',
            'registered_users',
            'total_rentals'
        ]
        
        available_vars = [col for col in correlation_vars if col in self.cleaned_data.columns]
        
        self.correlation_data = self.cleaned_data[available_vars].copy()
        
        print(f"✓ Selected {len(available_vars)} variables for correlation analysis")
        print(f"  Records: {len(self.correlation_data)}")
        
        self.correlation_data.to_csv(output_path, index=False)
        print(f"✓ Correlation dataset created: {output_path}")
        
        return True
    
    def create_descriptive_dataset(self, output_path='descriptive.csv'):
        if self.cleaned_data is None:
            print("✗ Please run comprehensive_data_cleaning first")
            return False

        print("\n=== Creating Descriptive Dataset ===")
        df = self.cleaned_data.copy()

        variables = {
            'dependent': ['total_rentals', 'casual_users', 'registered_users'],
            'independent': {
                'temporal': ['hour', 'weekday', 'workingday', 'month', 'season', 'year', 'holiday'],
                'environmental': ['weather_situation', 'temp', 'feeling_temperature', 'humidity', 'windspeed']
            }
        }

        plan_rows = [
            {'Section': 'PLAN', 'Item': 'Variables Selected - Dependent', 'Details': ', '.join(variables['dependent'])},
            {'Section': 'PLAN', 'Item': 'Variables Selected - Independent (Temporal)', 'Details': ', '.join(variables['independent']['temporal'])},
            {'Section': 'PLAN', 'Item': 'Variables Selected - Independent (Environmental)', 'Details': ', '.join(variables['independent']['environmental'])},
            {'Section': 'PLAN', 'Item': 'Objective', 'Details': 'Understand hourly demand patterns, seasonal/weekday effects, weather impact, and user-type behavior.'},
            {'Section': 'PLAN', 'Item': 'Theoretical Framework', 'Details': 'Demand Elasticity; Temporal Patterns; Weather–Demand; User Segmentation'},
            {'Section': 'PLAN', 'Item': 'Type of Analysis', 'Details': 'count, mean, median, mode, std, variance, quartiles, IQR, range, skewness, kurtosis; frequency distributions; cross-tabulations; correlations'}
        ]
        plan_df = pd.DataFrame(plan_rows)

        numeric_cols = ['total_rentals', 'casual_users', 'registered_users', 'temp', 'feeling_temperature', 'humidity', 'windspeed']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        basic_rows = []
        for col in numeric_cols:
            s = df[col].dropna()
            mode_vals = s.mode()
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
            q1 = s.quantile(0.25); q3 = s.quantile(0.75)
            
            basic_rows.append({
                'Section': 'BASIC_STATS', 
                'Variable': col, 
                'Count': int(s.count()), 
                'Mean': float(s.mean()),
                'Median': float(s.median()), 
                'Mode': float(mode_val) if pd.notnull(mode_val) else np.nan,
                'Std': float(s.std()), 
                'Variance': float(s.var()),
                'CV': float(s.std()/s.mean()) if s.mean() != 0 else np.nan,
                'Min': float(s.min()), 
                'Q1': float(q1), 
                'Q3': float(q3),
                'Max': float(s.max()), 
                'Range': float(s.max() - s.min()), 
                'IQR': float(q3 - q1),
                'Skew': float(s.skew()), 
                'Kurt': float(s.kurt())
            })
        basic_df = pd.DataFrame(basic_rows)

        tab_rows = []
        cat_vars = ['season', 'month', 'hour', 'weekday', 'holiday', 'workingday', 'weather_situation', 'year']
        n = len(df)
        for var in cat_vars:
            if var not in df.columns:
                continue
            counts = df[var].value_counts().sort_index()
            for key, count in counts.items():
                pct = (count / n) * 100 if n else 0
                mean_cnt = df.loc[df[var] == key, 'total_rentals'].mean() if 'total_rentals' in df.columns else np.nan
                std_cnt = df.loc[df[var] == key, 'total_rentals'].std() if 'total_rentals' in df.columns else np.nan
                tab_rows.append({
                    'Section': 'TABULATION', 
                    'Variable': var, 
                    'Category': int(key) if pd.notnull(key) else key,
                    'Count': int(count), 
                    'Percent': round(pct, 2), 
                    'Avg_cnt': round(float(mean_cnt), 2) if pd.notnull(mean_cnt) else np.nan,
                    'Std_cnt': round(float(std_cnt), 2) if pd.notnull(std_cnt) else np.nan
                })
        tabs_df = pd.DataFrame(tab_rows)

        comp_rows = []
        if 'year' in df.columns:
            for y in sorted(df['year'].unique()):
                sub = df[df['year'] == y]['total_rentals']
                comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'Year', 'Group': int(y), 
                                 'Mean': float(sub.mean()), 'Median': float(sub.median()), 
                                 'Std': float(sub.std()), 'N': int(sub.count())})
        if {'casual_users', 'registered_users', 'total_rentals'}.issubset(df.columns):
            total = df['total_rentals'].sum()
            casual_p = (df['casual_users'].sum() / total) * 100 if total else 0
            reg_p = (df['registered_users'].sum() / total) * 100 if total else 0
            comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'User Type Share', 
                            'Group': 'Casual_%', 'Value': round(casual_p, 2)})
            comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'User Type Share', 
                            'Group': 'Registered_%', 'Value': round(reg_p, 2)})
        if {'workingday', 'total_rentals'}.issubset(df.columns):
            for k in sorted(df['workingday'].unique()):
                sub = df[df['workingday'] == k]['total_rentals']
                comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'Workingday', 'Group': int(k), 
                                 'Mean': float(sub.mean()), 'Median': float(sub.median()), 
                                 'Std': float(sub.std()), 'N': int(sub.count())})
        if {'holiday', 'total_rentals'}.issubset(df.columns):
            for k in sorted(df['holiday'].unique()):
                sub = df[df['holiday'] == k]['total_rentals']
                comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'Holiday', 'Group': int(k), 
                                 'Mean': float(sub.mean()), 'Median': float(sub.median()), 
                                 'Std': float(sub.std()), 'N': int(sub.count())})
        comp_df = pd.DataFrame(comp_rows)

        dist_rows = []
        for var in ['hour', 'weekday', 'month']:
            if var not in df.columns:
                continue
            grp = df.groupby(var)['total_rentals'].agg(['count', 'mean', 'median', 'std'])
            for k in grp.index:
                dist_rows.append({'Section': 'DISTRIBUTION', 'Dimension': var, 'Category': int(k),
                                  'Count': int(grp.loc[k, 'count']), 'Mean': round(float(grp.loc[k, 'mean']), 2),
                                  'Median': round(float(grp.loc[k, 'median']), 2), 'Std': round(float(grp.loc[k, 'std']), 2)})
        dist_df = pd.DataFrame(dist_rows)

        if 'total_rentals' in df.columns:
            q1 = df['total_rentals'].quantile(0.25); q3 = df['total_rentals'].quantile(0.75)
            iqr = q3 - q1; lb = q1 - 1.5 * iqr; ub = q3 + 1.5 * iqr
            mask = (df['total_rentals'] < lb) | (df['total_rentals'] > ub)
            out_df = pd.DataFrame([{'Section': 'OUTLIERS', 'Total_Records': int(len(df)), 
                                   'Outliers_Count': int(mask.sum()), 
                                   'Outliers_%': round(float(mask.mean() * 100), 2), 
                                   'Lower_Bound': float(lb), 'Upper_Bound': float(ub), 
                                   'Min_cnt': int(df['total_rentals'].min()), 'Max_cnt': int(df['total_rentals'].max())}])
        else:
            out_df = pd.DataFrame([{'Section': 'OUTLIERS', 'Msg': 'total_rentals not found'}])

        self.descriptive_data = pd.concat([plan_df, basic_df, tabs_df, comp_df, dist_df, out_df], 
                                         ignore_index=True, sort=False)

        self.descriptive_data.to_csv(output_path, index=False)
        print(f"✓ Descriptive dataset created: {output_path}")
        print(f"  Total rows: {len(self.descriptive_data)}")
        for section in self.descriptive_data['Section'].unique():
            count = len(self.descriptive_data[self.descriptive_data['Section'] == section])
            print(f"    {section}: {count} rows")

        return True
    
    def create_predictive_dataset(self, output_path='predictive.csv'):
        if self.cleaned_data is None:
            print("✗ Please run comprehensive_data_cleaning first")
            return False

        print("\n=== Creating Predictive Dataset ===")

        predictive_vars = [
            'hour',
            'weekday',
            'workingday',
            'month',
            'season',
            'temp',
            'feeling_temperature',
            'humidity',
            'windspeed',
            'weather_situation',
            'holiday',
            'total_rentals'
        ]
        
        available_vars = [col for col in predictive_vars if col in self.cleaned_data.columns]
        
        self.predictive_data = self.cleaned_data[available_vars].copy()
        
        print(f"✓ Selected {len(available_vars)} variables for predictive modeling")
        print(f"  Records: {len(self.predictive_data)}")
        
        print(f"\n=== Predictive Sample Statistics ===")
        print(f"Average rentals per hour: {self.predictive_data['total_rentals'].mean():.2f}")
        print(f"Median rentals per hour:  {self.predictive_data['total_rentals'].median():.2f}")
        print(f"Min rentals per hour:     {self.predictive_data['total_rentals'].min()}")
        print(f"Max rentals per hour:     {self.predictive_data['total_rentals'].max()}")
        
        self.predictive_data.to_csv(output_path, index=False)
        print(f"✓ Predictive dataset created: {output_path}")
        
        return True

    def create_descriptive_visualizations(self, output_dir='descriptive_charts'):
        if self.cleaned_data is None:
            print("✗ Please run comprehensive_data_cleaning first")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        df = self.cleaned_data.copy()

        print(f"\n=== Visualizing Descriptive Charts ===")

        saved = []

        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['total_rentals'], bins=30, color='#69b3a2', edgecolor='black')
            ax.set_title('Distribution of Hourly Rentals')
            ax.set_xlabel('Total Rentals')
            ax.set_ylabel('Frequency')
            fig.tight_layout()
            p = os.path.join(output_dir, '01_hist_total_rentals.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Histogram failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            grp = df.groupby('hour')['total_rentals'].mean()
            ax.bar(grp.index, grp.values, color='#4c72b0', edgecolor='black')
            ax.set_title('Average Rentals by Hour')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Average Rentals')
            fig.tight_layout()
            p = os.path.join(output_dir, '02_bar_avg_by_hour.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar hour failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            grp = df.groupby('weekday')['total_rentals'].mean()
            ax.bar(grp.index, grp.values, color='#55a868', edgecolor='black')
            ax.set_title('Average Rentals by Weekday (0=Sun)')
            ax.set_xlabel('Weekday (0–6)')
            ax.set_ylabel('Average Rentals')
            ax.set_xticks(range(7)); ax.set_xticklabels(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
            fig.tight_layout()
            p = os.path.join(output_dir, '03_bar_avg_by_weekday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar weekday failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            grp = df.groupby('month')['total_rentals'].mean()
            ax.bar(grp.index, grp.values, color='#c44e52', edgecolor='black')
            ax.set_title('Average Rentals by Month')
            ax.set_xlabel('Month (1–12)')
            ax.set_ylabel('Average Rentals')
            fig.tight_layout()
            p = os.path.join(output_dir, '04_bar_avg_by_month.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar month failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            grp = df.groupby('season')['total_rentals'].mean()
            labels_map = {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}
            labels = [labels_map.get(i, str(i)) for i in grp.index]
            ax.bar(range(len(grp)), grp.values, color='#8172b3', edgecolor='black')
            ax.set_xticks(range(len(grp))); ax.set_xticklabels(labels)
            ax.set_title('Average Rentals by Season')
            ax.set_ylabel('Average Rentals')
            fig.tight_layout()
            p = os.path.join(output_dir, '05_bar_avg_by_season.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar season failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            grp = df.groupby('weather_situation')['total_rentals'].mean()
            weather_map = {1:'Clear',2:'Mist/Cloudy',3:'Light Rain',4:'Heavy Rain'}
            labels = [weather_map.get(i, str(i)) for i in grp.index]
            ax.bar(range(len(grp)), grp.values, color='#937860', edgecolor='black')
            ax.set_xticks(range(len(grp))); ax.set_xticklabels(labels, rotation=15)
            ax.set_title('Average Rentals by Weather')
            ax.set_ylabel('Average Rentals')
            fig.tight_layout()
            p = os.path.join(output_dir, '06_bar_avg_by_weather.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar weather failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df['temp'], df['total_rentals'], s=8, alpha=0.4, color='#4c72b0')
            ax.set_title('Temperature vs Rentals')
            ax.set_xlabel('Temperature (normalized)')
            ax.set_ylabel('Total Rentals')
            fig.tight_layout()
            p = os.path.join(output_dir, '07_scatter_temp_rentals.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Scatter temp failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df['humidity'], df['total_rentals'], s=8, alpha=0.4, color='#55a868')
            ax.set_title('Humidity vs Rentals')
            ax.set_xlabel('Humidity (normalized)')
            ax.set_ylabel('Total Rentals')
            fig.tight_layout()
            p = os.path.join(output_dir, '08_scatter_humidity_rentals.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Scatter humidity failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df['windspeed'], df['total_rentals'], s=8, alpha=0.4, color='#c44e52')
            ax.set_title('Windspeed vs Rentals')
            ax.set_xlabel('Windspeed (normalized)')
            ax.set_ylabel('Total Rentals')
            fig.tight_layout()
            p = os.path.join(output_dir, '09_scatter_windspeed_rentals.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Scatter windspeed failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(x='workingday', y='total_rentals', data=df, ax=ax)
            ax.set_title('Rentals by Workingday (0/1)')
            fig.tight_layout()
            p = os.path.join(output_dir, '10_box_rentals_by_workingday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Box workingday failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(x='holiday', y='total_rentals', data=df, ax=ax)
            ax.set_title('Rentals by Holiday (0/1)')
            fig.tight_layout()
            p = os.path.join(output_dir, '11_box_rentals_by_holiday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Box holiday failed: {e}")

        try:
            pivot = df.pivot_table(index='hour', columns='weekday', values='total_rentals', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.heatmap(pivot, cmap='YlGnBu', ax=ax)
            ax.set_title('Average Rentals by Hour × Weekday')
            ax.set_xlabel('Weekday (0–6)'); ax.set_ylabel('Hour (0–23)')
            fig.tight_layout()
            p = os.path.join(output_dir, '12_heatmap_hour_weekday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Heatmap failed: {e}")

        print(f"✓ Generated {len(saved)} charts.")
        return True

    def create_all_datasets(self):
        print("\n=== Creating All Datasets ===")
        
        print("\n>>> STEP 1: Comprehensive Data Preprocessing <<<")
        self.comprehensive_data_cleaning('MailigCruzGalopeTorrefrancaTamondong_CleanedDataset.csv')
        
        print("\n>>> STEP 2: Creating Specialized Datasets from Cleaned Data <<<")
        self.create_correlation_dataset()
        self.create_descriptive_dataset()
        self.create_predictive_dataset()
        self.create_descriptive_visualizations()
        
        print("\n=== All datasets created successfully from cleaned data ===\n")

class CorrelationAnalyzer:
    def __init__(self, data):
        self.data = data
        self.correlation_matrix = None
        
    def compute_correlations(self):
        print("\n=== Computing Correlation Matrix ===")
        self.correlation_matrix = self.data.corr(method='pearson')
        print("✓ Correlation matrix computed")
        return self.correlation_matrix
    
    def analyze_environmental_correlations(self):
        print("\n=== Environmental Correlations with Rentals ===")
        print("Objective: Identify how weather conditions influence bike rental demand")
        print("Type: Linear Pearson correlation (continuous variables)")
        print()
        
        env_vars = ['temp', 'feeling_temperature', 'humidity', 'windspeed', 'weather_situation']
        target = 'total_rentals'
        
        results = []
        for var in env_vars:
            if var in self.correlation_matrix.columns:
                corr = self.correlation_matrix.loc[var, target]
                results.append({
                    'Variable': var,
                    'Correlation': corr,
                    'Strength': self._classify_strength(abs(corr)),
                    'Direction': 'Positive' if corr > 0 else 'Negative'
                })
        
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        return df_results
    
    def analyze_temporal_correlations(self):
        print("\n=== Temporal Correlations with Rentals ===")
        print("Objective: Identify how time-based factors influence rental patterns")
        print("Type: Linear Pearson correlation (ordinal/binary variables)")
        print()
        
        temp_vars = ['hour', 'weekday', 'month', 'season', 'workingday', 'holiday']
        target = 'total_rentals'
        
        results = []
        for var in temp_vars:
            if var in self.correlation_matrix.columns:
                corr = self.correlation_matrix.loc[var, target]
                results.append({
                    'Variable': var,
                    'Correlation': corr,
                    'Strength': self._classify_strength(abs(corr)),
                    'Direction': 'Positive' if corr > 0 else 'Negative'
                })
        
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        return df_results
    
    def analyze_user_type_correlations(self):
        print("\n=== User Type Correlations ===")
        print("Objective: Understand relationship between casual and registered users")
        print("Type: Linear Pearson correlation")
        print()
        
        if 'casual_users' in self.correlation_matrix.columns and 'registered_users' in self.correlation_matrix.columns:
            corr_casual_registered = self.correlation_matrix.loc['casual_users', 'registered_users']
            corr_casual_total = self.correlation_matrix.loc['casual_users', 'total_rentals']
            corr_registered_total = self.correlation_matrix.loc['registered_users', 'total_rentals']
            
            print(f"Casual vs Registered:     {corr_casual_registered:>7.4f} ({self._classify_strength(abs(corr_casual_registered))})")
            print(f"Casual vs Total:          {corr_casual_total:>7.4f} ({self._classify_strength(abs(corr_casual_total))})")
            print(f"Registered vs Total:      {corr_registered_total:>7.4f} ({self._classify_strength(abs(corr_registered_total))})")
            
            return {
                'casual_registered': corr_casual_registered,
                'casual_total': corr_casual_total,
                'registered_total': corr_registered_total
            }
        else:
            print("User type columns not found in data")
            return {}
    
    def find_strongest_correlations(self, threshold=0.3):
        print(f"\n=== Strongest Correlations with Rentals (|r| > {threshold}) ===")
        
        target = 'total_rentals'
        if target in self.correlation_matrix.columns:
            corr_series = self.correlation_matrix[target].drop(target).abs().sort_values(ascending=False)
            strong_corr = corr_series[corr_series > threshold]
            
            print(f"Found {len(strong_corr)} variables with strong correlations:")
            for var, corr in strong_corr.items():
                direction = 'Positive' if self.correlation_matrix.loc[var, target] > 0 else 'Negative'
                print(f"  {var:25s} r={self.correlation_matrix.loc[var, target]:>7.4f} ({direction}, {self._classify_strength(corr)})")
            
            return strong_corr
        else:
            print(f"Target variable '{target}' not found")
            return pd.Series()
    
    def analyze_multicollinearity(self):
        print("\n=== Multicollinearity Analysis ===")
        print("Objective: Identify highly correlated predictor variables")
        print("Threshold: |r| > 0.8 indicates potential multicollinearity")
        print()
        
        high_corr_pairs = []
        target = 'total_rentals'
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                var1 = self.correlation_matrix.columns[i]
                var2 = self.correlation_matrix.columns[j]
                corr = self.correlation_matrix.iloc[i, j]
                
                if abs(corr) > 0.8 and var1 != target and var2 != target:
                    high_corr_pairs.append({
                        'Variable 1': var1,
                        'Variable 2': var2,
                        'Correlation': corr
                    })
        
        if high_corr_pairs:
            df_pairs = pd.DataFrame(high_corr_pairs).sort_values('Correlation', 
                                                                   key=abs, 
                                                                   ascending=False)
            print(df_pairs.to_string(index=False))
        else:
            print("✓ No significant multicollinearity detected")
        
        return high_corr_pairs
    
    def visualize_correlation_heatmap(self, save_path='./correlation_graphs/correlation_heatmap.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(self.correlation_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix - Bike Sharing Variables', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Correlation heatmap saved: {save_path}")
        plt.close()
    
    def visualize_rental_correlations(self, save_path='./correlation_graphs/rental_correlations.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        target = 'total_rentals'
        if target in self.correlation_matrix.columns:
            cnt_corr = self.correlation_matrix[target].drop(target).sort_values()
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'green' for x in cnt_corr.values]
            cnt_corr.plot(kind='barh', color=colors, edgecolor='black', linewidth=0.5)
            
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.xlabel('Correlation Coefficient', fontsize=12)
            plt.ylabel('Variables', fontsize=12)
            plt.title('Correlation with Total Bike Rentals', 
                      fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Rental correlations chart saved: {save_path}")
            plt.close()

    def visualize_individual_correlations(self, output_dir='./correlation_graphs/individual'):
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== Generating Individual Correlation Charts ===")
        
        variables = self.correlation_matrix.columns.tolist()
        
        for var in variables:
            var_corr = self.correlation_matrix[var].drop(var).sort_values()
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'green' for x in var_corr.values]
            var_corr.plot(kind='barh', color=colors, edgecolor='black', linewidth=0.5)
            
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            plt.xlabel('Correlation Coefficient', fontsize=12)
            plt.ylabel('Variables', fontsize=12)
            plt.title(f'Correlations with {var}', 
                      fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'{var}_correlations.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {save_path}")
        
        print(f"\n✓ Generated {len(variables)} individual correlation charts")
    
    def generate_correlation_report(self):
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS REPORT")
        print("="*60)
        
        self.compute_correlations()
        self.analyze_environmental_correlations()
        self.analyze_temporal_correlations()
        self.analyze_user_type_correlations()
        self.find_strongest_correlations()
        self.analyze_multicollinearity()
        
        print("\n" + "="*60)
        print("✓ Correlation analysis complete!")
        print("="*60)
    
    @staticmethod
    def _classify_strength(corr):
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "Very Strong"
        elif abs_corr >= 0.5:
            return "Strong"
        elif abs_corr >= 0.3:
            return "Moderate"
        elif abs_corr >= 0.1:
            return "Weak"
        else:
            return "Very Weak"

class DescriptiveAnalyzer:
    def __init__(self, data):
        self.data = data
        self.basic_stats = None
        self.tabulations = None
        self.distributions = None
        self.outliers = None

    def compute_basic_statistics(self):
        numeric_cols = ['total_rentals', 'casual_users', 'registered_users', 
                       'temp', 'feeling_temperature', 'humidity', 'windspeed']
        numeric_cols = [c for c in numeric_cols if c in self.data.columns]
        
        rows = []
        for col in numeric_cols:
            s = self.data[col].dropna()
            mode_vals = s.mode()
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            
            rows.append({
                'Variable': col, 
                'Count': int(s.count()), 
                'Mean': float(s.mean()), 
                'Median': float(s.median()), 
                'Mode': float(mode_val) if pd.notnull(mode_val) else np.nan, 
                'Std': float(s.std()), 
                'Min': float(s.min()), 
                'Q1': float(q1), 
                'Q3': float(q3), 
                'Max': float(s.max()), 
                'Range': float(s.max() - s.min()), 
                'IQR': float(q3 - q1), 
                'Skew': float(s.skew()), 
                'Kurt': float(s.kurt())
            })
        
        self.basic_stats = pd.DataFrame(rows)
        return self.basic_stats

    def analyze_categorical(self):
        cat_vars = ['season', 'month', 'hour', 'weekday', 'holiday', 
                   'workingday', 'weather_situation', 'year']
        rows = []
        n = len(self.data)
        
        for var in cat_vars:
            if var not in self.data.columns:
                continue
            counts = self.data[var].value_counts().sort_index()
            for key, count in counts.items():
                pct = (count / n) * 100 if n else 0
                mean_cnt = self.data.loc[self.data[var] == key, 'total_rentals'].mean() if 'total_rentals' in self.data.columns else np.nan
                rows.append({
                    'Variable': var, 
                    'Category': int(key) if pd.notnull(key) else key, 
                    'Count': int(count), 
                    'Percent': round(pct, 2), 
                    'Avg_rentals': round(float(mean_cnt), 2) if pd.notnull(mean_cnt) else np.nan
                })
        
        self.tabulations = pd.DataFrame(rows)
        return self.tabulations

    def analyze_distributions(self):
        rows = []
        for var in ['hour', 'weekday', 'month']:
            if var not in self.data.columns:
                continue
            grp = self.data.groupby(var)['total_rentals'].agg(['count', 'mean', 'median', 'std'])
            for k in grp.index:
                rows.append({
                    'Dimension': var, 
                    'Category': int(k), 
                    'Count': int(grp.loc[k, 'count']), 
                    'Mean': round(float(grp.loc[k, 'mean']), 2), 
                    'Median': round(float(grp.loc[k, 'median']), 2), 
                    'Std': round(float(grp.loc[k, 'std']), 2)
                })
        
        self.distributions = pd.DataFrame(rows)
        return self.distributions

    def detect_outliers(self):
        if 'total_rentals' not in self.data.columns:
            self.outliers = pd.DataFrame([{'Msg': 'total_rentals not found'}])
            return self.outliers
        
        q1 = self.data['total_rentals'].quantile(0.25)
        q3 = self.data['total_rentals'].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        mask = (self.data['total_rentals'] < lb) | (self.data['total_rentals'] > ub)
        
        self.outliers = pd.DataFrame([{
            'Total_Records': int(len(self.data)), 
            'Outliers_Count': int(mask.sum()), 
            'Outliers_%': round(float(mask.mean() * 100), 2), 
            'Lower_Bound': float(lb), 
            'Upper_Bound': float(ub)
        }])
        return self.outliers

    def get_visualization_list(self):
        return [
            'Histogram: Distribution of hourly total_rentals',
            'Bar: Average total_rentals by hour',
            'Bar: Average total_rentals by weekday',
            'Bar: Average total_rentals by month',
            'Bar: Average total_rentals by season',
            'Bar: Average total_rentals by weather situation',
            'Scatter: temp vs total_rentals',
            'Scatter: humidity vs total_rentals',
            'Scatter: windspeed vs total_rentals',
            'Boxplot: total_rentals by workingday',
            'Boxplot: total_rentals by holiday',
            'Heatmap: hour x weekday average rentals'
        ]


class PredictiveAnalyzer:    
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lr_model = None
        self.dt_model = None
        self.results = {}
    
    def prepare_data(self, test_size=0.2, random_state=42):
        target_col = 'total_rentals'
        
        if target_col not in self.data.columns:
            print(f"✗ Target column '{target_col}' not found")
            return False
        
        X = self.data.drop(target_col, axis=1)
        y = self.data[target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n=== Data Preparation ===")
        print(f"Training set: {len(self.X_train)} records")
        print(f"Testing set: {len(self.X_test)} records")
        print(f"Features: {list(X.columns)}")
        
        return True
    
    def train_linear_regression(self):
        print("\n=== Training Linear Regression ===")
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)
        
        y_pred_train = self.lr_model.predict(self.X_train)
        y_pred_test = self.lr_model.predict(self.X_test)
        
        self.results['lr'] = {
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'mae': mean_absolute_error(self.y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test
        }
        
        print(f"Training R²: {self.results['lr']['train_r2']:.4f}")
        print(f"Testing R²: {self.results['lr']['test_r2']:.4f}")
        print(f"MAE: {self.results['lr']['mae']:.2f}")
        print(f"RMSE: {self.results['lr']['rmse']:.2f}")
        
        return self.lr_model
    
    def train_decision_tree(self, max_depth=10):
        print("\n=== Training Decision Tree ===")
        self.dt_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.dt_model.fit(self.X_train, self.y_train)
        
        y_pred_train = self.dt_model.predict(self.X_train)
        y_pred_test = self.dt_model.predict(self.X_test)
        
        self.results['dt'] = {
            'train_r2': r2_score(self.y_train, y_pred_train),
            'test_r2': r2_score(self.y_test, y_pred_test),
            'mae': mean_absolute_error(self.y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
            'train_predictions': y_pred_train,
            'test_predictions': y_pred_test
        }
        
        print(f"Training R²: {self.results['dt']['train_r2']:.4f}")
        print(f"Testing R²: {self.results['dt']['test_r2']:.4f}")
        print(f"MAE: {self.results['dt']['mae']:.2f}")
        print(f"RMSE: {self.results['dt']['rmse']:.2f}")
        
        return self.dt_model
    
    def get_feature_importance(self):
        if self.dt_model is None:
            print("✗ Please train decision tree model first")
            return None
        
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Feature Importance (Decision Tree) ===")
        print(feature_importance)
        return feature_importance
    
    def compare_models(self):
        if not self.results:
            print("✗ Please train models first")
            return
        
        print("\n=== Model Comparison ===")
        comparison = pd.DataFrame(self.results).T
        comparison = comparison[['train_r2', 'test_r2', 'mae', 'rmse']]
        print(comparison)
        
        best_model = 'Linear Regression' if self.results['lr']['test_r2'] > self.results['dt']['test_r2'] else 'Decision Tree'
        print(f"\n✓ Best performing model: {best_model}")
        
        return comparison
    
    def save_prediction_metrics(self, output_path='predictive_data/prediction_metrics.csv'):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        if not self.results:
            print("✗ Please train models first")
            return
        
        metrics_data = []
        
        lr_residuals = self.y_test.values - self.results['lr']['test_predictions']
        metrics_data.append({
            'Model': 'Linear Regression',
            'Train_R2': self.results['lr']['train_r2'],
            'Test_R2': self.results['lr']['test_r2'],
            'MAE': self.results['lr']['mae'],
            'RMSE': self.results['lr']['rmse'],
            'Mean_Residual': np.mean(lr_residuals),
            'Std_Residual': np.std(lr_residuals),
            'Min_Residual': np.min(lr_residuals),
            'Max_Residual': np.max(lr_residuals)
        })
        
        dt_residuals = self.y_test.values - self.results['dt']['test_predictions']
        metrics_data.append({
            'Model': 'Decision Tree',
            'Train_R2': self.results['dt']['train_r2'],
            'Test_R2': self.results['dt']['test_r2'],
            'MAE': self.results['dt']['mae'],
            'RMSE': self.results['dt']['rmse'],
            'Mean_Residual': np.mean(dt_residuals),
            'Std_Residual': np.std(dt_residuals),
            'Min_Residual': np.min(dt_residuals),
            'Max_Residual': np.max(dt_residuals)
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(output_path, index=False)
        print(f"✓ Prediction metrics saved: {output_path}")
        
        return metrics_df
    
    def save_prediction_results(self, output_path='predictive_data/prediction_results.csv'):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        if not self.results:
            print("✗ Please train models first")
            return
        
        results_df = pd.DataFrame({
            'Actual': self.y_test.values,
            'LR_Predicted': self.results['lr']['test_predictions'],
            'LR_Residual': self.y_test.values - self.results['lr']['test_predictions'],
            'DT_Predicted': self.results['dt']['test_predictions'],
            'DT_Residual': self.y_test.values - self.results['dt']['test_predictions']
        })
        
        results_df.to_csv(output_path, index=False)
        print(f"✓ Prediction results saved: {output_path}")
        
        return results_df
    
    def visualize_predictions(self, output_dir='predictive_charts'):
        if self.lr_model is None or self.dt_model is None:
            print("✗ Please train both models first")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        lr_pred = self.results['lr']['test_predictions']
        dt_pred = self.results['dt']['test_predictions']
        lr_residuals = self.y_test.values - lr_pred
        dt_residuals = self.y_test.values - dt_pred
        
        print(f"\n=== Creating Predictive Visualizations ===")
        saved_charts = []
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.y_test, lr_pred, alpha=0.5, s=15, color='#4c72b0')
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Rentals', fontsize=11)
            ax.set_ylabel('Predicted Rentals', fontsize=11)
            ax.set_title(f'Linear Regression: Actual vs Predicted\nR² = {self.results["lr"]["test_r2"]:.4f}', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, '01_lr_actual_vs_predicted.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 1 failed: {e}")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(self.y_test, dt_pred, alpha=0.5, s=15, color='#55a868')
            ax.plot([self.y_test.min(), self.y_test.max()], 
                   [self.y_test.min(), self.y_test.max()], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Rentals', fontsize=11)
            ax.set_ylabel('Predicted Rentals', fontsize=11)
            ax.set_title(f'Decision Tree: Actual vs Predicted\nR² = {self.results["dt"]["test_r2"]:.4f}', 
                        fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, '02_dt_actual_vs_predicted.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 2 failed: {e}")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(lr_pred, lr_residuals, alpha=0.5, s=15, color='#4c72b0')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Rentals', fontsize=11)
            ax.set_ylabel('Residuals', fontsize=11)
            ax.set_title('Linear Regression: Residual Plot', fontweight='bold')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, '03_lr_residual_plot.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 3 failed: {e}")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(dt_pred, dt_residuals, alpha=0.5, s=15, color='#55a868')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Rentals', fontsize=11)
            ax.set_ylabel('Residuals', fontsize=11)
            ax.set_title('Decision Tree: Residual Plot', fontweight='bold')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, '04_dt_residual_plot.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 4 failed: {e}")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(lr_residuals, bins=50, color='#4c72b0', edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Residuals', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Linear Regression: Residual Distribution\nMean = {np.mean(lr_residuals):.2f}, Std = {np.std(lr_residuals):.2f}', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, '05_lr_residual_distribution.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 5 failed: {e}")
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(dt_residuals, bins=50, color='#55a868', edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Residuals', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Decision Tree: Residual Distribution\nMean = {np.mean(dt_residuals):.2f}, Std = {np.std(dt_residuals):.2f}', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = os.path.join(output_dir, '06_dt_residual_distribution.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 6 failed: {e}")
        
        try:
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.dt_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(feature_importance['feature'], feature_importance['importance'], 
                   color='#55a868', edgecolor='black')
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_ylabel('Features', fontsize=11)
            ax.set_title('Decision Tree: Feature Importance', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            fig.tight_layout()
            path = os.path.join(output_dir, '07_feature_importance.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 7 failed: {e}")
        
        try:
            metrics = ['Train R²', 'Test R²', 'MAE', 'RMSE']
            lr_values = [self.results['lr']['train_r2'], self.results['lr']['test_r2'], 
                        self.results['lr']['mae'], self.results['lr']['rmse']]
            dt_values = [self.results['dt']['train_r2'], self.results['dt']['test_r2'], 
                        self.results['dt']['mae'], self.results['dt']['rmse']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width/2, lr_values, width, label='Linear Regression', color='#4c72b0', edgecolor='black')
            ax.bar(x + width/2, dt_values, width, label='Decision Tree', color='#55a868', edgecolor='black')
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            fig.tight_layout()
            path = os.path.join(output_dir, '08_model_comparison.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 8 failed: {e}")
        
        try:
            if 'hour' in self.X_test.columns:
                test_data = self.X_test.copy()
                test_data['actual'] = self.y_test.values
                test_data['lr_pred'] = lr_pred
                test_data['dt_pred'] = dt_pred
                test_data['lr_error'] = np.abs(lr_residuals)
                test_data['dt_error'] = np.abs(dt_residuals)
                
                hourly_error = test_data.groupby('hour').agg({
                    'lr_error': 'mean',
                    'dt_error': 'mean'
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(hourly_error.index, hourly_error['lr_error'], 
                       marker='o', label='Linear Regression', color='#4c72b0', linewidth=2)
                ax.plot(hourly_error.index, hourly_error['dt_error'], 
                       marker='s', label='Decision Tree', color='#55a868', linewidth=2)
                ax.set_xlabel('Hour of Day', fontsize=11)
                ax.set_ylabel('Mean Absolute Error', fontsize=11)
                ax.set_title('Prediction Error by Hour of Day', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = os.path.join(output_dir, '09_error_by_hour.png')
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_charts.append(path)
                print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 9 failed: {e}")
        
        try:
            if 'season' in self.X_test.columns:
                test_data = self.X_test.copy()
                test_data['actual'] = self.y_test.values
                test_data['lr_pred'] = lr_pred
                test_data['dt_pred'] = dt_pred
                test_data['lr_error'] = np.abs(lr_residuals)
                test_data['dt_error'] = np.abs(dt_residuals)
                
                seasonal_error = test_data.groupby('season').agg({
                    'lr_error': 'mean',
                    'dt_error': 'mean'
                })
                
                season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
                x_labels = [season_labels.get(i, str(i)) for i in seasonal_error.index]
                
                x = np.arange(len(seasonal_error))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(x - width/2, seasonal_error['lr_error'], width, 
                      label='Linear Regression', color='#4c72b0', edgecolor='black')
                ax.bar(x + width/2, seasonal_error['dt_error'], width, 
                      label='Decision Tree', color='#55a868', edgecolor='black')
                
                ax.set_ylabel('Mean Absolute Error', fontsize=11)
                ax.set_title('Prediction Error by Season', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                fig.tight_layout()
                path = os.path.join(output_dir, '10_error_by_season.png')
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_charts.append(path)
                print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 10 failed: {e}")
        
        try:
            if 'weather_situation' in self.X_test.columns:
                test_data = self.X_test.copy()
                test_data['actual'] = self.y_test.values
                test_data['lr_pred'] = lr_pred
                test_data['dt_pred'] = dt_pred
                test_data['lr_error'] = np.abs(lr_residuals)
                test_data['dt_error'] = np.abs(dt_residuals)
                
                weather_error = test_data.groupby('weather_situation').agg({
                    'lr_error': 'mean',
                    'dt_error': 'mean'
                })
                
                weather_labels = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain', 4: 'Heavy Rain'}
                x_labels = [weather_labels.get(i, str(i)) for i in weather_error.index]
                
                x = np.arange(len(weather_error))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.bar(x - width/2, weather_error['lr_error'], width, 
                      label='Linear Regression', color='#4c72b0', edgecolor='black')
                ax.bar(x + width/2, weather_error['dt_error'], width, 
                      label='Decision Tree', color='#55a868', edgecolor='black')
                
                ax.set_ylabel('Mean Absolute Error', fontsize=11)
                ax.set_title('Prediction Error by Weather Situation', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=15)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                fig.tight_layout()
                path = os.path.join(output_dir, '11_error_by_weather.png')
                fig.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_charts.append(path)
                print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 11 failed: {e}")
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].scatter(self.y_test, lr_pred, alpha=0.5, s=15, color='#4c72b0')
            axes[0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Rentals')
            axes[0].set_ylabel('Predicted Rentals')
            axes[0].set_title(f'Linear Regression\nR² = {self.results["lr"]["test_r2"]:.4f}, MAE = {self.results["lr"]["mae"]:.2f}')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].scatter(self.y_test, dt_pred, alpha=0.5, s=15, color='#55a868')
            axes[1].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1].set_xlabel('Actual Rentals')
            axes[1].set_ylabel('Predicted Rentals')
            axes[1].set_title(f'Decision Tree\nR² = {self.results["dt"]["test_r2"]:.4f}, MAE = {self.results["dt"]["mae"]:.2f}')
            axes[1].grid(True, alpha=0.3)
            
            fig.suptitle('Model Comparison: Actual vs Predicted', fontweight='bold', fontsize=14)
            fig.tight_layout()
            path = os.path.join(output_dir, '12_combined_comparison.png')
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_charts.append(path)
            print(f"✓ Saved: {path}")
        except Exception as e:
            print(f"! Chart 12 failed: {e}")
        
        print(f"\n✓ Generated {len(saved_charts)} predictive charts in '{output_dir}/' folder")
        return saved_charts

if __name__ == "__main__":  
    print("="*60)
    print("GROUP 1 - BIKE SHARING DATA ANALYSIS")
    print("MIDTERM PROJECT - DATA PRE-PROCESSING & ANALYSIS")
    print("="*60)
    print("MEMBERS:")
    print("  • Jan Lancelot Mailig")
    print("  • Jocas Arabella Cruz")
    print("  • Eleazar James Galope")
    print("  • Jecho Torrefranca")
    print("  • John Neil Tamondong")
    print("="*60)
    
    curator = BikeDataCurator('hour.csv')
    
    if curator.load_data():
        curator.create_all_datasets()
        
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE ANALYSIS")
        print("="*60)

        print("\n" + "="*60)
        print("PART 1: CORRELATION ANALYSIS")
        print("="*60)

        try:
            correlation_data = pd.read_csv('correlation.csv')
            print(f"✓ Loaded correlation dataset: {len(correlation_data)} records")
            
            correlator = CorrelationAnalyzer(correlation_data)
            correlator.generate_correlation_report()
            correlator.visualize_correlation_heatmap()
            correlator.visualize_rental_correlations()
            correlator.visualize_individual_correlations()
            
            print("\n✓ Correlation analysis complete!")
        except FileNotFoundError:
            print("✗ Error: correlation.csv not found. Please ensure data preprocessing completed.")
        except Exception as e:
            print(f"✗ Error in correlation analysis: {e}")
        
        # Descriptive Statistics and Analysis
        print("\n" + "="*60)
        print("PART 2: DESCRIPTIVE ANALYSIS")
        print("="*60)
        
        try:
            desc_df = pd.read_csv('descriptive.csv')
            print(f"✓ Loaded descriptive dataset: {len(desc_df)} rows")
            
            if 'Section' in desc_df.columns:
                print("\nDescriptive Data Sections:")
                sections = ['PLAN', 'BASIC_STATS', 'TABULATION', 'COMPARISON', 'DISTRIBUTION', 'OUTLIERS']
                for sec in sections:
                    sec_count = len(desc_df[desc_df['Section'] == sec])
                    print(f"  {sec:<15}: {sec_count:>4} rows")
            
            try:
                basic_stats = desc_df[desc_df['Section'] == 'BASIC_STATS'].head(5)
                if not basic_stats.empty:
                    print("\nBasic Statistics Preview (first 5 variables):")
                    if 'Variable' in basic_stats.columns:
                        for _, row in basic_stats.iterrows():
                            if pd.notna(row.get('Mean')) and pd.notna(row.get('Std')) and pd.notna(row.get('Median')):
                                print(f"  {row['Variable']:<25}: Mean={float(row.get('Mean')):>10.2f}, "
                                      f"Std={float(row.get('Std')):>10.2f}, "
                                      f"Median={float(row.get('Median')):>10.2f}")
            except Exception as e:
                print(f"  Note: Could not display basic stats preview: {e}")
            
            cleaned_data = pd.read_csv('MailigCruzGalopeTorrefrancaTamondong_CleanedDataset.csv')
            desc_analyzer = DescriptiveAnalyzer(cleaned_data)
            
            print("\n=== Running Descriptive Analysis ===")
            basic_stats_df = desc_analyzer.compute_basic_statistics()
            print(f"✓ Computed basic statistics for {len(basic_stats_df)} variables")
            
            categorical_df = desc_analyzer.analyze_categorical()
            if not categorical_df.empty:
                unique_vars = categorical_df['Variable'].unique()
                print(f"✓ Analyzed {len(unique_vars)} categorical variables")
            
            distribution_df = desc_analyzer.analyze_distributions()
            if not distribution_df.empty:
                print(f"✓ Analyzed distributions for key dimensions")
            
            outlier_df = desc_analyzer.detect_outliers()
            if 'Outliers_Count' in outlier_df.columns:
                outlier_count = outlier_df['Outliers_Count'].iloc[0]
                outlier_pct = outlier_df['Outliers_%'].iloc[0]
                print(f"✓ Detected {outlier_count} outliers ({outlier_pct:.1f}% of data)")
            
            print("\n✓ Descriptive analysis complete!")
            
            print("\n=== Available Descriptive Visualizations ===")
            viz_list = desc_analyzer.get_visualization_list()
            for i, viz in enumerate(viz_list, 1):
                print(f"  {i:2d}. {viz}")
                
        except FileNotFoundError:
            print("✗ Error: descriptive.csv not found. Please ensure data preprocessing completed.")
        except Exception as e:
            print(f"✗ Error in descriptive analysis: {e}")

        print("\n" + "="*60)
        print("PART 3: PREDICTIVE ANALYSIS")
        print("="*60)
        
        try:
            predictive_data = pd.read_csv('predictive.csv')
            print(f"✓ Loaded predictive dataset: {len(predictive_data)} records")
            print(f"  Features: {list(predictive_data.columns)}")
            
            predictor = PredictiveAnalyzer(predictive_data)
            
            if predictor.prepare_data(test_size=0.2, random_state=42):
                
                print("\n=== Training Machine Learning Models ===")
                predictor.train_linear_regression()
                predictor.train_decision_tree(max_depth=10)
                
                print("\n=== Analyzing Feature Importance ===")
                feature_importance = predictor.get_feature_importance()
                if feature_importance is not None and len(feature_importance) > 0:
                    top_features = feature_importance.head(5)
                    print("\nTop 5 Most Important Features:")
                    for idx, row in top_features.iterrows():
                        print(f"  {row['feature']:<25}: {row['importance']:.4f}")
                
                print("\n=== Comparing Model Performance ===")
                comparison = predictor.compare_models()
                
                print("\n=== Saving Results ===")
                metrics_df = predictor.save_prediction_metrics()
                results_df = predictor.save_prediction_results()
                
                print("\n=== Generating Predictive Visualizations ===")
                saved_charts = predictor.visualize_predictions()
                
                print("\n✓ Predictive analysis complete!")
                
                print("\n" + "="*50)
                print("MODEL PERFORMANCE SUMMARY")
                print("="*50)
                if comparison is not None and not comparison.empty:
                    lr_r2 = comparison.loc['lr', 'test_r2']
                    dt_r2 = comparison.loc['dt', 'test_r2']
                    lr_mae = comparison.loc['lr', 'mae']
                    dt_mae = comparison.loc['dt', 'mae']
                    lr_rmse = comparison.loc['lr', 'rmse']
                    dt_rmse = comparison.loc['dt', 'rmse']
                    
                    print(f"\n📊 Linear Regression:")
                    print(f"  • Test R²: {lr_r2:.4f}")
                    print(f"  • MAE: {lr_mae:.2f} rentals")
                    print(f"  • RMSE: {lr_rmse:.2f} rentals")
                    
                    print(f"\n🌳 Decision Tree:")
                    print(f"  • Test R²: {dt_r2:.4f}")
                    print(f"  • MAE: {dt_mae:.2f} rentals")
                    print(f"  • RMSE: {dt_rmse:.2f} rentals")
                    
                    print("\n" + "-"*50)
                    if lr_r2 > dt_r2:
                        improvement = ((lr_r2 - dt_r2) / dt_r2) * 100
                        print(f"🏆 Linear Regression performs better")
                        print(f"   R² is {lr_r2-dt_r2:.4f} higher ({improvement:.1f}% improvement)")
                    else:
                        improvement = ((dt_r2 - lr_r2) / lr_r2) * 100
                        print(f"🏆 Decision Tree performs better")
                        print(f"   R² is {dt_r2-lr_r2:.4f} higher ({improvement:.1f}% improvement)")
                    print("="*50)
                        
        except FileNotFoundError:
            print("✗ Error: predictive.csv not found. Please ensure data preprocessing completed.")
        except Exception as e:
            print(f"✗ Error in predictive analysis: {e}")
        
    else:
        print("\n" + "="*60)
        print("❌ ERROR: FAILED TO LOAD DATA")
        print("="*60)
        print("\n✗ Could not load 'hour.csv' file.")
        print("\nCurrent working directory:", os.getcwd())
        print("="*60)