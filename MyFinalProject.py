import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BikeDataCurator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.main_data = None
        self.correlation_data = None
        self.descriptive_data = None
        self.predictive_data = None
        
    def load_data(self):
        try:
            self.main_data = pd.read_csv(self.filepath)
            print(f"✓ Data loaded successfully: {len(self.main_data)} records")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File '{self.filepath}' not found")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def create_correlation_dataset(self, output_path='correlation.csv'):

        if self.main_data is None:
            print("✗ Please load data first")
            return False
        
        print("\n=== Creating Correlation Dataset ===")
        
        correlation_vars = [
            'temp',
            'atemp',
            'hum',
            'windspeed',
            'season',
            'mnth',
            'hr',
            'weekday',
            'holiday',
            'workingday',
            'weathersit',
            'casual',
            'registered',
            'cnt'
        ]
        
        self.correlation_data = self.main_data[correlation_vars].copy()
        initial_count = len(self.correlation_data)
        print(f"Initial records: {initial_count}")
        
        print("\n=== Cleaning Correlation Data ===")
        
        missing_before = self.correlation_data.isnull().sum()
        if missing_before.sum() > 0:
            print("\n⚠ Missing values detected:")
            print(missing_before[missing_before > 0])
            self.correlation_data = self.correlation_data.dropna()
            print(f"✓ Removed {initial_count - len(self.correlation_data)} rows with missing values")
        else:
            print("✓ No missing values found")
        
        duplicates = self.correlation_data.duplicated().sum()
        if duplicates > 0:
            self.correlation_data = self.correlation_data.drop_duplicates()
            print(f"✓ Removed {duplicates} duplicate rows")
        else:
            print("✓ No duplicate rows found")
        
        print("\n=== Validating Normalized Variables ===")
        normalized_vars = ['temp', 'atemp', 'hum', 'windspeed']
        for var in normalized_vars:
            invalid = self.correlation_data[(self.correlation_data[var] < 0) | 
                                            (self.correlation_data[var] > 1)]
            if len(invalid) > 0:
                print(f"⚠ Found {len(invalid)} invalid {var} values")
                self.correlation_data = self.correlation_data[
                    (self.correlation_data[var] >= 0) & 
                    (self.correlation_data[var] <= 1)
                ]
        print("✓ Normalized variables validated")
        
        print("\n=== Validating Categorical Variables ===")
        validations = {
            'season': (1, 4),
            'mnth': (1, 12),
            'hr': (0, 23),
            'weekday': (0, 6),
            'holiday': (0, 1),
            'workingday': (0, 1),
            'weathersit': (1, 4)
        }
        
        for var, (min_val, max_val) in validations.items():
            invalid = self.correlation_data[
                (self.correlation_data[var] < min_val) | 
                (self.correlation_data[var] > max_val)
            ]
            if len(invalid) > 0:
                print(f"⚠ Found {len(invalid)} invalid {var} values")
                self.correlation_data = self.correlation_data[
                    (self.correlation_data[var] >= min_val) & 
                    (self.correlation_data[var] <= max_val)
                ]
        print("✓ Categorical variables validated")
        
        print("\n=== Detecting Outliers in Rental Counts ===")
        for col in ['cnt', 'casual', 'registered']:
            Q1 = self.correlation_data[col].quantile(0.25)
            Q3 = self.correlation_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            
            outliers = self.correlation_data[
                (self.correlation_data[col] < lower) | 
                (self.correlation_data[col] > upper)
            ]
            
            if len(outliers) > 0:
                print(f"⚠ Found {len(outliers)} outliers in {col}")
                self.correlation_data = self.correlation_data[
                    (self.correlation_data[col] >= lower) & 
                    (self.correlation_data[col] <= upper)
                ]
        
        final_count = len(self.correlation_data)
        removed_count = initial_count - final_count
        
        print("\n" + "="*50)
        print("CORRELATION PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Initial records:    {initial_count}")
        print(f"Final records:      {final_count}")
        print(f"Removed records:    {removed_count} ({(removed_count/initial_count)*100:.2f}%)")
        print(f"Data quality:       {(final_count/initial_count)*100:.2f}%")
        print("="*50)
        
        self.correlation_data.to_csv(output_path, index=False)
        print(f"\n✓ Correlation dataset created: {output_path}")
        print(f"  Variables: {len(correlation_vars)}")
        print(f"  Records: {final_count}")
        
        return True
    
    def create_descriptive_dataset(self, output_path='descriptive.csv'):
        if self.main_data is None:
            print("✗ Please load data first")
            return False

        print("\n=== Creating Descriptive Dataset ===")
        df = self.main_data.copy()

        variables = {
            'dependent': ['cnt', 'casual', 'registered'],
            'independent': {
                'temporal': ['hr', 'weekday', 'workingday', 'mnth', 'season', 'yr', 'holiday'],
                'environmental': ['weathersit', 'temp', 'atemp', 'hum', 'windspeed']
            }
        }

        plan_rows = [
            {'Section': 'PLAN', 'Item': 'Variables Selected - Dependent', 'Details': ', '.join(variables['dependent'])},
            {'Section': 'PLAN', 'Item': 'Variables Selected - Independent (Temporal)', 'Details': ', '.join(variables['independent']['temporal'])},
            {'Section': 'PLAN', 'Item': 'Variables Selected - Independent (Environmental)', 'Details': ', '.join(variables['independent']['environmental'])},
            {'Section': 'PLAN', 'Item': 'Objective', 'Details': 'Understand hourly demand patterns, seasonal/weekday effects, weather impact, and user-type behavior.'},
            {'Section': 'PLAN', 'Item': 'Theoretical Framework', 'Details': 'Demand Elasticity; Temporal Patterns; Weather–Demand; User Segmentation'},
            {'Section': 'PLAN', 'Item': 'Type of Analysis', 'Details': 'count, mean, median, mode, std, quartiles, IQR; simple tabulation with percent; comparisons; distributions; outliers'}
        ]
        plan_df = pd.DataFrame(plan_rows)

        numeric_cols = [c for c in ['cnt', 'casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed'] if c in df.columns]
        basic_rows = []
        for col in numeric_cols:
            s = df[col].dropna()
            mode_vals = s.mode()
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
            q1 = s.quantile(0.25); q3 = s.quantile(0.75)
            basic_rows.append({
                'Section': 'BASIC_STATS', 'Variable': col, 'Count': int(s.count()), 'Mean': float(s.mean()),
                'Median': float(s.median()), 'Mode': float(mode_val) if pd.notnull(mode_val) else np.nan,
                'Std': float(s.std()), 'Min': float(s.min()), 'Q1': float(q1), 'Q3': float(q3),
                'Max': float(s.max()), 'Range': float(s.max() - s.min()), 'IQR': float(q3 - q1),
                'Skew': float(s.skew()), 'Kurt': float(s.kurt())
            })
        basic_df = pd.DataFrame(basic_rows)

        tab_rows = []
        cat_vars = ['season', 'mnth', 'hr', 'weekday', 'holiday', 'workingday', 'weathersit', 'yr']
        n = len(df)
        for var in cat_vars:
            if var not in df.columns:
                continue
            counts = df[var].value_counts().sort_index()
            for key, count in counts.items():
                pct = (count / n) * 100 if n else 0
                mean_cnt = df.loc[df[var] == key, 'cnt'].mean() if 'cnt' in df.columns else np.nan
                tab_rows.append({'Section': 'TABULATION', 'Variable': var, 'Category': int(key) if pd.notnull(key) else key,
                                 'Count': int(count), 'Percent': round(pct, 2), 'Avg_cnt': round(float(mean_cnt), 2) if pd.notnull(mean_cnt) else np.nan})
        tabs_df = pd.DataFrame(tab_rows)

        comp_rows = []
        if 'yr' in df.columns:
            for y in sorted(df['yr'].unique()):
                sub = df[df['yr'] == y]['cnt']
                comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'Year', 'Group': int(y), 'Mean': float(sub.mean()), 'Median': float(sub.median()), 'Std': float(sub.std()), 'N': int(sub.count())})
        if {'casual', 'registered', 'cnt'}.issubset(df.columns):
            total = df['cnt'].sum()
            casual_p = (df['casual'].sum() / total) * 100 if total else 0
            reg_p = (df['registered'].sum() / total) * 100 if total else 0
            comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'User Type Share', 'Group': 'Casual_%', 'Value': round(casual_p, 2)})
            comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'User Type Share', 'Group': 'Registered_%', 'Value': round(reg_p, 2)})
        if {'workingday', 'cnt'}.issubset(df.columns):
            for k in sorted(df['workingday'].unique()):
                sub = df[df['workingday'] == k]['cnt']
                comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'Workingday', 'Group': int(k), 'Mean': float(sub.mean()), 'Median': float(sub.median()), 'Std': float(sub.std()), 'N': int(sub.count())})
        if {'holiday', 'cnt'}.issubset(df.columns):
            for k in sorted(df['holiday'].unique()):
                sub = df[df['holiday'] == k]['cnt']
                comp_rows.append({'Section': 'COMPARISON', 'Comparison': 'Holiday', 'Group': int(k), 'Mean': float(sub.mean()), 'Median': float(sub.median()), 'Std': float(sub.std()), 'N': int(sub.count())})
        comp_df = pd.DataFrame(comp_rows)

        dist_rows = []
        for var in ['hr', 'weekday', 'mnth']:
            if var not in df.columns:
                continue
            grp = df.groupby(var)['cnt'].agg(['count', 'mean', 'median', 'std'])
            for k in grp.index:
                dist_rows.append({'Section': 'DISTRIBUTION', 'Dimension': var, 'Category': int(k),
                                  'Count': int(grp.loc[k, 'count']), 'Mean': round(float(grp.loc[k, 'mean']), 2),
                                  'Median': round(float(grp.loc[k, 'median']), 2), 'Std': round(float(grp.loc[k, 'std']), 2)})
        dist_df = pd.DataFrame(dist_rows)

        if 'cnt' in df.columns:
            q1 = df['cnt'].quantile(0.25); q3 = df['cnt'].quantile(0.75)
            iqr = q3 - q1; lb = q1 - 1.5 * iqr; ub = q3 + 1.5 * iqr
            mask = (df['cnt'] < lb) | (df['cnt'] > ub)
            out_df = pd.DataFrame([{'Section': 'OUTLIERS', 'Total_Records': int(len(df)), 'Outliers_Count': int(mask.sum()), 'Outliers_%': round(float(mask.mean() * 100), 2), 'Lower_Bound': float(lb), 'Upper_Bound': float(ub), 'Min_cnt': int(df['cnt'].min()), 'Max_cnt': int(df['cnt'].max())}])
        else:
            out_df = pd.DataFrame([{'Section': 'OUTLIERS', 'Msg': 'cnt not found'}])

        self.descriptive_data = pd.concat([plan_df, basic_df, tabs_df, comp_df, dist_df, out_df], ignore_index=True, sort=False)

        try:
            import os
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        except Exception:
            pass

        self.descriptive_data.to_csv(output_path, index=False)
        print(f"✓ Descriptive dataset created: {output_path}")

        return True
    
    def create_predictive_dataset(self, output_path='predictive.csv'):
        if self.main_data is None:
            print("✗ Please load data first")
            return False

        print("\n=== Creating Predictive Dataset ===")

        predictive_vars = [
            'hr',
            'weekday',
            'workingday',
            'mnth',
            'season',
            'temp',
            'atemp',
            'hum',
            'windspeed',
            'weathersit',
            'holiday',
            'cnt'
        ]
        
        self.predictive_data = self.main_data[predictive_vars].copy()
        initial_count = len(self.predictive_data)
        print(f"Initial records: {initial_count}")
        
        missing_before = self.predictive_data.isnull().sum()
        if missing_before.sum() > 0:
            print("\n⚠ Missing values detected:")
            print(missing_before[missing_before > 0])
            
            self.predictive_data = self.predictive_data.dropna()
            print(f"✓ Removed {initial_count - len(self.predictive_data)} rows with missing values")
        else:
            print("✓ No missing values found")
        
        duplicates = self.predictive_data.duplicated().sum()
        if duplicates > 0:
            self.predictive_data = self.predictive_data.drop_duplicates()
            print(f"✓ Removed {duplicates} duplicate rows")
        else:
            print("✓ No duplicate rows found")
        
        print("\n=== Validating Data Ranges ===")
        
        invalid_rows = 0
        
        invalid_hr = self.predictive_data[(self.predictive_data['hr'] < 0) | 
                                          (self.predictive_data['hr'] > 23)]
        if len(invalid_hr) > 0:
            print(f"⚠ Found {len(invalid_hr)} rows with invalid hour values")
            self.predictive_data = self.predictive_data[(self.predictive_data['hr'] >= 0) & 
                                                        (self.predictive_data['hr'] <= 23)]
            invalid_rows += len(invalid_hr)
        
        invalid_weekday = self.predictive_data[(self.predictive_data['weekday'] < 0) | 
                                               (self.predictive_data['weekday'] > 6)]
        if len(invalid_weekday) > 0:
            print(f"⚠ Found {len(invalid_weekday)} rows with invalid weekday values")
            self.predictive_data = self.predictive_data[(self.predictive_data['weekday'] >= 0) & 
                                                        (self.predictive_data['weekday'] <= 6)]
            invalid_rows += len(invalid_weekday)
        
        invalid_mnth = self.predictive_data[(self.predictive_data['mnth'] < 1) | 
                                            (self.predictive_data['mnth'] > 12)]
        if len(invalid_mnth) > 0:
            print(f"⚠ Found {len(invalid_mnth)} rows with invalid month values")
            self.predictive_data = self.predictive_data[(self.predictive_data['mnth'] >= 1) & 
                                                        (self.predictive_data['mnth'] <= 12)]
            invalid_rows += len(invalid_mnth)
        
        invalid_season = self.predictive_data[(self.predictive_data['season'] < 1) | 
                                              (self.predictive_data['season'] > 4)]
        if len(invalid_season) > 0:
            print(f"⚠ Found {len(invalid_season)} rows with invalid season values")
            self.predictive_data = self.predictive_data[(self.predictive_data['season'] >= 1) & 
                                                        (self.predictive_data['season'] <= 4)]
            invalid_rows += len(invalid_season)
        
        invalid_weather = self.predictive_data[(self.predictive_data['weathersit'] < 1) | 
                                               (self.predictive_data['weathersit'] > 4)]
        if len(invalid_weather) > 0:
            print(f"⚠ Found {len(invalid_weather)} rows with invalid weather values")
            self.predictive_data = self.predictive_data[(self.predictive_data['weathersit'] >= 1) & 
                                                        (self.predictive_data['weathersit'] <= 4)]
            invalid_rows += len(invalid_weather)
        
        normalized_cols = ['temp', 'atemp', 'hum', 'windspeed']
        for col in normalized_cols:
            invalid_norm = self.predictive_data[(self.predictive_data[col] < 0) | 
                                                (self.predictive_data[col] > 1)]
            if len(invalid_norm) > 0:
                print(f"⚠ Found {len(invalid_norm)} rows with invalid {col} values")
                self.predictive_data = self.predictive_data[(self.predictive_data[col] >= 0) & 
                                                            (self.predictive_data[col] <= 1)]
                invalid_rows += len(invalid_norm)
        
        binary_cols = ['workingday', 'holiday']
        for col in binary_cols:
            invalid_binary = self.predictive_data[~self.predictive_data[col].isin([0, 1])]
            if len(invalid_binary) > 0:
                print(f"⚠ Found {len(invalid_binary)} rows with invalid {col} values")
                self.predictive_data = self.predictive_data[self.predictive_data[col].isin([0, 1])]
                invalid_rows += len(invalid_binary)
        
        invalid_cnt = self.predictive_data[self.predictive_data['cnt'] < 0]
        if len(invalid_cnt) > 0:
            print(f"⚠ Found {len(invalid_cnt)} rows with negative rental counts")
            self.predictive_data = self.predictive_data[self.predictive_data['cnt'] >= 0]
            invalid_rows += len(invalid_cnt)
        
        if invalid_rows == 0:
            print("✓ All data ranges are valid")
        else:
            print(f"✓ Removed {invalid_rows} rows with invalid values")
        
        print("\n=== Detecting Outliers ===")
        
        Q1 = self.predictive_data['cnt'].quantile(0.25)
        Q3 = self.predictive_data['cnt'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = self.predictive_data[(self.predictive_data['cnt'] < lower_bound) | 
                                        (self.predictive_data['cnt'] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"⚠ Found {len(outliers)} outlier records")
            print(f"  Outlier bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
            print(f"  Outlier range: {outliers['cnt'].min():.0f} - {outliers['cnt'].max():.0f}")
            
            self.predictive_data = self.predictive_data[(self.predictive_data['cnt'] >= lower_bound) & 
                                                        (self.predictive_data['cnt'] <= upper_bound)]
            print(f"✓ Removed {len(outliers)} outlier rows")
        else:
            print("✓ No extreme outliers detected")
        
        print("\n=== Validating Data Types ===")
        
        integer_cols = ['hr', 'weekday', 'workingday', 'mnth', 'season', 
                       'weathersit', 'holiday', 'cnt']
        for col in integer_cols:
            self.predictive_data[col] = self.predictive_data[col].astype(int)
        
        float_cols = ['temp', 'atemp', 'hum', 'windspeed']
        for col in float_cols:
            self.predictive_data[col] = self.predictive_data[col].astype(float)
        
        print("✓ Data types validated and corrected")
        
        final_count = len(self.predictive_data)
        removed_count = initial_count - final_count
        removed_pct = (removed_count / initial_count) * 100
        
        print("\n" + "="*50)
        print("PREDICTIVE PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Initial records:    {initial_count}")
        print(f"Final records:      {final_count}")
        print(f"Removed records:    {removed_count} ({removed_pct:.2f}%)")
        print(f"Data quality:       {((final_count/initial_count)*100):.2f}%")
        print("="*50)
        
        self.predictive_data.to_csv(output_path, index=False)
        print(f"\n✓ Predictive dataset created: {output_path}")
        print(f"  Variables: {len(predictive_vars)}")
        print(f"  Records: {final_count}")
        
        print(f"\n=== Predictive Sample Statistics ===")
        print(f"Average rentals per hour: {self.predictive_data['cnt'].mean():.2f}")
        print(f"Median rentals per hour:  {self.predictive_data['cnt'].median():.2f}")
        print(f"Min rentals per hour:     {self.predictive_data['cnt'].min()}")
        print(f"Max rentals per hour:     {self.predictive_data['cnt'].max()}")
        
        return True

    def create_descriptive_visualizations(self, output_dir='descriptive_charts'):
        if self.main_data is None:
            print("✗ Please load data first")
            return False
        os.makedirs(output_dir, exist_ok=True)

        df = self.main_data.copy()

        print(f"\n=== Visualizing Descriptive Charts ===")

        saved = []

        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['cnt'], bins=30, color='#69b3a2', edgecolor='black')
            ax.set_title('Distribution of Hourly Rentals (cnt)')
            ax.set_xlabel('cnt')
            ax.set_ylabel('Frequency')
            fig.tight_layout()
            p = os.path.join(output_dir, '01_hist_cnt.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Histogram failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            grp = df.groupby('hr')['cnt'].mean()
            ax.bar(grp.index, grp.values, color='#4c72b0', edgecolor='black')
            ax.set_title('Average Rentals by Hour (hr)')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Average cnt')
            fig.tight_layout()
            p = os.path.join(output_dir, '02_bar_avg_by_hr.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar hr failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            grp = df.groupby('weekday')['cnt'].mean()
            ax.bar(grp.index, grp.values, color='#55a868', edgecolor='black')
            ax.set_title('Average Rentals by Weekday (0=Sun)')
            ax.set_xlabel('Weekday (0–6)')
            ax.set_ylabel('Average cnt')
            ax.set_xticks(range(7)); ax.set_xticklabels(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
            fig.tight_layout()
            p = os.path.join(output_dir, '03_bar_avg_by_weekday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar weekday failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            grp = df.groupby('mnth')['cnt'].mean()
            ax.bar(grp.index, grp.values, color='#c44e52', edgecolor='black')
            ax.set_title('Average Rentals by Month')
            ax.set_xlabel('Month (1–12)')
            ax.set_ylabel('Average cnt')
            fig.tight_layout()
            p = os.path.join(output_dir, '04_bar_avg_by_month.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar month failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            grp = df.groupby('season')['cnt'].mean()
            labels_map = {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}
            labels = [labels_map.get(i, str(i)) for i in grp.index]
            ax.bar(range(len(grp)), grp.values, color='#8172b3', edgecolor='black')
            ax.set_xticks(range(len(grp))); ax.set_xticklabels(labels)
            ax.set_title('Average Rentals by Season')
            ax.set_ylabel('Average cnt')
            fig.tight_layout()
            p = os.path.join(output_dir, '05_bar_avg_by_season.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar season failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            grp = df.groupby('weathersit')['cnt'].mean()
            weather_map = {1:'Clear',2:'Mist/Cloudy',3:'Light Rain',4:'Heavy Rain'}
            labels = [weather_map.get(i, str(i)) for i in grp.index]
            ax.bar(range(len(grp)), grp.values, color='#937860', edgecolor='black')
            ax.set_xticks(range(len(grp))); ax.set_xticklabels(labels, rotation=15)
            ax.set_title('Average Rentals by Weather')
            ax.set_ylabel('Average cnt')
            fig.tight_layout()
            p = os.path.join(output_dir, '06_bar_avg_by_weathersit.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Bar weather failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df['temp'], df['cnt'], s=8, alpha=0.4, color='#4c72b0')
            ax.set_title('Temperature vs Rentals')
            ax.set_xlabel('temp (normalized)')
            ax.set_ylabel('cnt')
            fig.tight_layout()
            p = os.path.join(output_dir, '07_scatter_temp_cnt.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Scatter temp failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df['hum'], df['cnt'], s=8, alpha=0.4, color='#55a868')
            ax.set_title('Humidity vs Rentals')
            ax.set_xlabel('hum (normalized)')
            ax.set_ylabel('cnt')
            fig.tight_layout()
            p = os.path.join(output_dir, '08_scatter_hum_cnt.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Scatter hum failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df['windspeed'], df['cnt'], s=8, alpha=0.4, color='#c44e52')
            ax.set_title('Windspeed vs Rentals')
            ax.set_xlabel('windspeed (normalized)')
            ax.set_ylabel('cnt')
            fig.tight_layout()
            p = os.path.join(output_dir, '09_scatter_windspeed_cnt.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Scatter windspeed failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(x='workingday', y='cnt', data=df, ax=ax)
            ax.set_title('cnt by Workingday (0/1)')
            fig.tight_layout()
            p = os.path.join(output_dir, '10_box_cnt_by_workingday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Box workingday failed: {e}")

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(x='holiday', y='cnt', data=df, ax=ax)
            ax.set_title('cnt by Holiday (0/1)')
            fig.tight_layout()
            p = os.path.join(output_dir, '11_box_cnt_by_holiday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Box holiday failed: {e}")

        try:
            pivot = df.pivot_table(index='hr', columns='weekday', values='cnt', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.heatmap(pivot, cmap='YlGnBu', ax=ax)
            ax.set_title('Average cnt by Hour × Weekday')
            ax.set_xlabel('weekday (0–6)'); ax.set_ylabel('hr (0–23)')
            fig.tight_layout()
            p = os.path.join(output_dir, '12_heatmap_hr_weekday.png')
            fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig)
            saved.append(p); print(f"✓ Saved: {p}")
        except Exception as e:
            print(f"! Heatmap failed: {e}")

        print(f"✓ Generated {len(saved)} charts.")
        return True

    def create_all_datasets(self):
        print("\n=== Creating Sub-Datasets ===")
        self.create_correlation_dataset()
        self.create_descriptive_dataset()
        self.create_predictive_dataset()
        self.create_descriptive_visualizations()
        print("=== Dataset creation complete ===\n")

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
        
        env_vars = ['temp', 'atemp', 'hum', 'windspeed']
        target = 'cnt'
        
        results = []
        for var in env_vars:
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
        
        temp_vars = ['hr', 'weekday', 'mnth', 'season', 'workingday', 'holiday']
        target = 'cnt'
        
        results = []
        for var in temp_vars:
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
        
        corr_casual_registered = self.correlation_matrix.loc['casual', 'registered']
        corr_casual_total = self.correlation_matrix.loc['casual', 'cnt']
        corr_registered_total = self.correlation_matrix.loc['registered', 'cnt']
        
        print(f"Casual vs Registered:     {corr_casual_registered:>7.4f} ({self._classify_strength(abs(corr_casual_registered))})")
        print(f"Casual vs Total:          {corr_casual_total:>7.4f} ({self._classify_strength(abs(corr_casual_total))})")
        print(f"Registered vs Total:      {corr_registered_total:>7.4f} ({self._classify_strength(abs(corr_registered_total))})")
        
        return {
            'casual_registered': corr_casual_registered,
            'casual_total': corr_casual_total,
            'registered_total': corr_registered_total
        }
    
    def find_strongest_correlations(self, threshold=0.3):
        print(f"\n=== Strongest Correlations with Rentals (|r| > {threshold}) ===")
        
        cnt_corr = self.correlation_matrix['cnt'].drop('cnt').abs().sort_values(ascending=False)
        strong_corr = cnt_corr[cnt_corr > threshold]
        
        print(f"Found {len(strong_corr)} variables with strong correlations:")
        for var, corr in strong_corr.items():
            direction = 'Positive' if self.correlation_matrix.loc[var, 'cnt'] > 0 else 'Negative'
            print(f"  {var:15s} r={self.correlation_matrix.loc[var, 'cnt']:>7.4f} ({direction}, {self._classify_strength(corr)})")
        
        return strong_corr
    
    def analyze_multicollinearity(self):
        print("\n=== Multicollinearity Analysis ===")
        print("Objective: Identify highly correlated predictor variables")
        print("Threshold: |r| > 0.8 indicates potential multicollinearity")
        print()
        
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                var1 = self.correlation_matrix.columns[i]
                var2 = self.correlation_matrix.columns[j]
                corr = self.correlation_matrix.iloc[i, j]
                
                if abs(corr) > 0.8 and var1 != 'cnt' and var2 != 'cnt':
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
        
        cnt_corr = self.correlation_matrix['cnt'].drop('cnt').sort_values()
        
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
        numeric_cols = [c for c in ['cnt', 'casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed'] if c in self.data.columns]
        rows = []
        for col in numeric_cols:
            s = self.data[col].dropna()
            mode_vals = s.mode(); mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
            q1 = s.quantile(0.25); q3 = s.quantile(0.75)
            rows.append({'Variable': col, 'Count': int(s.count()), 'Mean': float(s.mean()), 'Median': float(s.median()), 'Mode': float(mode_val) if pd.notnull(mode_val) else np.nan, 'Std': float(s.std()), 'Min': float(s.min()), 'Q1': float(q1), 'Q3': float(q3), 'Max': float(s.max()), 'Range': float(s.max() - s.min()), 'IQR': float(q3 - q1), 'Skew': float(s.skew()), 'Kurt': float(s.kurt())})
        self.basic_stats = pd.DataFrame(rows)
        return self.basic_stats

    def analyze_categorical(self):
        cat_vars = ['season', 'mnth', 'hr', 'weekday', 'holiday', 'workingday', 'weathersit', 'yr']
        rows = []
        n = len(self.data)
        for var in cat_vars:
            if var not in self.data.columns:
                continue
            counts = self.data[var].value_counts().sort_index()
            for key, count in counts.items():
                pct = (count / n) * 100 if n else 0
                mean_cnt = self.data.loc[self.data[var] == key, 'cnt'].mean() if 'cnt' in self.data.columns else np.nan
                rows.append({'Variable': var, 'Category': int(key) if pd.notnull(key) else key, 'Count': int(count), 'Percent': round(pct, 2), 'Avg_cnt': round(float(mean_cnt), 2) if pd.notnull(mean_cnt) else np.nan})
        self.tabulations = pd.DataFrame(rows)
        return self.tabulations

    def analyze_distributions(self):
        rows = []
        for var in ['hr', 'weekday', 'mnth']:
            if var not in self.data.columns:
                continue
            grp = self.data.groupby(var)['cnt'].agg(['count', 'mean', 'median', 'std'])
            for k in grp.index:
                rows.append({'Dimension': var, 'Category': int(k), 'Count': int(grp.loc[k, 'count']), 'Mean': round(float(grp.loc[k, 'mean']), 2), 'Median': round(float(grp.loc[k, 'median']), 2), 'Std': round(float(grp.loc[k, 'std']), 2)})
        self.distributions = pd.DataFrame(rows)
        return self.distributions

    def detect_outliers(self):
        if 'cnt' not in self.data.columns:
            self.outliers = pd.DataFrame([{'Msg': 'cnt not found'}])
            return self.outliers
        q1 = self.data['cnt'].quantile(0.25); q3 = self.data['cnt'].quantile(0.75)
        iqr = q3 - q1; lb = q1 - 1.5 * iqr; ub = q3 + 1.5 * iqr
        mask = (self.data['cnt'] < lb) | (self.data['cnt'] > ub)
        self.outliers = pd.DataFrame([{'Total_Records': int(len(self.data)), 'Outliers_Count': int(mask.sum()), 'Outliers_%': round(float(mask.mean() * 100), 2), 'Lower_Bound': float(lb), 'Upper_Bound': float(ub)}])
        return self.outliers

    def get_visualization_list(self):
        return [
            'Histogram: Distribution of hourly cnt',
            'Bar: Average cnt by hr',
            'Bar: Average cnt by weekday',
            'Bar: Average cnt by month',
            'Bar: Average cnt by season',
            'Bar: Average cnt by weather situation',
            'Scatter: temp vs cnt',
            'Scatter: hum vs cnt',
            'Scatter: windspeed vs cnt',
            'Boxplot: cnt by workingday',
            'Boxplot: cnt by holiday'
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
        X = self.data.drop('cnt', axis=1)
        y = self.data['cnt']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n=== Data Preparation ===")
        print(f"Training set: {len(self.X_train)} records")
        print(f"Testing set: {len(self.X_test)} records")
        print(f"Features: {list(X.columns)}")
    
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
        """Save detailed prediction metrics to CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
        """Save actual vs predicted values"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
        """Generate comprehensive predictive visualizations"""
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
            test_data = self.X_test.copy()
            test_data['actual'] = self.y_test.values
            test_data['lr_pred'] = lr_pred
            test_data['dt_pred'] = dt_pred
            test_data['lr_error'] = np.abs(lr_residuals)
            test_data['dt_error'] = np.abs(dt_residuals)
            
            hourly_error = test_data.groupby('hr').agg({
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
            weather_error = test_data.groupby('weathersit').agg({
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
    print(""" ###%%%%%####%%%%%%#%%#%%%%%%%%#%%%%%**====%++*%**#*#*****+#%#***++*+*#######%%####%#####%#%%%@@@@%%##==#
    #######%*###%%###%#%%%%#########*#*##*====-=+**####+**#++#######+*+#*########%#%%%%%###%#%#######*+=-=-:
    ##########*##############%#########**+----#%%#*****###*#**############%%%%%%###%##%%%#%%@%%%%%%%#%*#**##
    @%######*##*###%###%%%%%%#%###%%####*+----%###***###%%%##%%%####*####%##%##%%%%%%###%%%%%#%%%##%##*+****
    ########%####%#%###%####%####%#####**+----*###***###%#%#####*###*#####%%%%%#####%%%%%%%%%%%%%%###%##***#
    #################%#####*####*#**####*+::::*##%%%%#%###%##%%%@@%#####%#%%%###%%##%#%%%%%%####%####****++*
    #########%%#*###***##########%%#%###+=::::*####%%%##%%%%@@@###%%#########%%%#%#%#%%##%######%###*##*****
    +*###%*#*###%#######%#%#%#%%%########+::::*###****##*##@@*----:+%####%#%#####%%%%%%%%############%######
    --==+#+++*#+**%%%#####%#=+*##%%%@@@@@%+:::*##%%%%%%%%%#%%=##**#=####%%#%%%%%%%##%%%#%#%#%%%%@@@%%%%%%###
    *#%*--:-*#**#**+==-=+=++*+*##*=%@%+=-=*+::*%#%%%%%%%#%#==---=--.-###%%%%#%#%%%%%%%%%%%%%%%%%%%%%%@%%%@%%
    =*+--:..:--=====-:..::--==**++=%@+=-::-+::#%%%%%%%%%%%%%*++*#*+-*##%#######%%%%%%%%%%%%%%%%%@@@@@@@%####
    +*=--...=+====---:.:--:-=+++=-:++=++:=:-::+#%%#%%%%%####*+*+*+=#%%#%%%##%%%%%%%%%@@@@@@@@@@@@@%%#%%@@@@@
    +=====++==+=---:--::------===+=--+=++--:::-==*+*#%%%%##=#=+##=#%%%%%%%%%%%%%%%%%%%%@@@@@%#*#%@@@@@@@@@@@
    =-+%##@*+#@%=---.....----*===*%*-=**+=-::::-===++**+**#%+%%#+#*+*##%%%%%%%#######%%%%+===#@@@@@@@@@@@@@@
    @%#@%%%#*#%%*+:.-=-==+=+###%%%%##:+#*:::::=+#%******+##%%*#@*%=*#*+*%%#%%%##*+=-:.......:*@@@@@@@@@@@@@@
    @@@%##%%#%@@@@%%%%%%%%%%%%%%**#%+@++=+::::*@@@@%@%%*##%%%%#%@#=%*%%#+###=:...............=@@@@@@@@@@@@@@
    @@%%##%%#%%%%%%%%%%%%%%%%%%#*###%+@%=*-**=*@@@@@@%##*+*%@@#@%=+@%%@%*++=:.....-====......-@@@@@@@@@@@@@@
    @%@@%#%%%%%%%%#############%##%%%+*%#-:**###@@@@@%##=:-*@@#%##**##@@%++=..-=--:...-......-@@@@@@@@@@@@@@
    @%%@%#%%#@@@@@@@@@@@@@@@@@@%##*#%#*%+:-##%%*#@@@@%#+:...%@#=+%#**#@@@*:...:.......-......-@@@@@@@@@@@@@@
    @#%@%####@@@@@@@@@@@@@@@@@@%%#+=*%*#**++#@@%**@@@%@@+:...#**%%%@@%%##*-:........:.-......-@@@@@@@@@@@@@@
    @#%@%****@@@@@@@@@@@@@@@@@@@#+:.:#+=+%@@%%@@#**@@%%@@+:...+%%%%@@%%#*+:=:.........-......:@@@@@%#*++=---
    @*#@%****@@@@@@@@@@@@@@@@@@%@@=:..+*#%%@%%%@*..-%%@@@@*-:..=%%%%%%@#++:.-=-:..::..-........::-:::::::...
    %*#@#****%@@@@@@@@@@@@@@@@@%@@@-...*%%%@@*+@@=...%@@@@#*+:...:-*@@@#++:...:+=::----.........::::::::....
    %*#@#***+%@@@@@@@@@@@@@@@@@@@@@#-..:*##@@*+@@%*=%@@@###%@@#:-=-:-*==-:.....-===-*=..........::::::::....
    :.:-:::::-===+++++++***###%@@#**%+:..*##%++@@@@@@@*++*:.:#%#*+*#-++*+=-=.......*:.::........:::::::.....
    ..........................-@#=+%:....--:*==@@@@%+**++#-..*##%@###%@#*+.::.......:=.*=.......:::::::.....
    ..........................-#*=*#*..=*****+=@%@%+++*==*=+*%@@%%@@*=##*+:..-:===-::.--........:::::::::...
    ::::::::.................:+#**=....+%@@@%%+@%#++*##**......-+=-==....:....-=...............:::::::::....
    .........................+#%#*:......=+==::###*%@%%%:.......*%*=-::.......-:..............:::::::::.....
    ........................+#@@@%::.......--...+@@%**@@-........=#*+++-:......:.......==-----====---------:
    .......................:#@#%@@*=::..:...-::.:#%#*%@@+:.........++++=-::...:++:-=...=-----------=-:::::::
    ........................%###@@@#=---==++*+=--*%*#%@@@#=:.......-***+*+=-===#%==-...==----------=-----:::
    ........................###%%+.-:===+++*##+=--*#%@@@#%@#--------====**%%#@@@*=%@@@@*=-------:::::::::...
    ........................-#%@=.#=:-=+++=-=*+*+-==-#@@%#%--==-=--+***++#@@%**+@#**+*-=+++=+====-----::::::
    ......................*%%%%@*..:-+*++==--=%@##+#*+@@@@#:-:-===+%%%%@@******@@##@++:+%%%%%%#####*++======
    ...................:#*:.....+%--+*=-===-===%%##%@%*@@#=:--==-=#@@%******+%@@***@%##%%%%#**#%%%%##%##****
    ..................-#=......:+-++-=**++====#@@@@*====+=:--=+==%+-:-+*##++#%=-=%%*%%#*#*##**%#*#@@@%######
    .................-*+...:...#+-=+**=@*++#@@@%%%#=:::::::-=+=-:::::+**++*%+..:#%%#@@=:::::::::*%*=---=----
    .................**.......=*%%##%#+#@@*+=#%%%#=...:+%#-==---:.::=-=+*#@=:.-%###.**..:::::::-:-*%*:::::::
    ................-*=......*:.*@@@@@#+@+--#@%%%*.:*@%+::-=--=-::%#*+*#%%:.:.%#*-*=-#:.:::::-:::::=%#-:::::
    ................-*:...-#%+++-#@@%%%*@#=*%%%%%%@#+:::::=++++=:=@@@@%@%-...*%+:.-#:=*:::::::::::::=@#=::::
    ................=*....*@@@@@#+-+#%%@@@#%@%%@#-::::::-*=-=**+-+***@@@@%:.=@*=:::+*-=*-::-------::-*@%----
    ................-+:..##@@*==***%%@*#@@@@@@@#====---:*@@@@@#@-::*@%*#@@%+*%+-::::+*:=#----------::-#@*:--
    :---------------++---+*#%=---=--#@@#%@@@@@%#==::::.::%@@@@@@@-#@%+#@#+%@@#+:::-::=#--#+=------::::=%%=--
    -----------------*=----+%-----:=%@#*#*@@@@@@#%%%%@%@#+@@@%@@@@%+=:::::--%*+--------%#**#+----::----#@*::
    -----------------=#-=-=**=====+#@++##%**%@@@#::-----=:+%%%%%@@%*::::::..##+:::::::-=##+#-:::.::::::#@%:.
    ----========-=====+#+====--=-=%@=.......:*%@@%*=:::...:=*%@%*@-.........+%=:....:::::..::..:::::..:#@%..
    :-:::::::::::::::::-#*==-::-*@@=::::::.::=+*++=::::-----=##%%+..........:@+-...:......:............#@#..
    ::------------------=+%@%%@@%-:......:::::::::::::::::::..::-............=%=......:...:...........-%@+..
    ::--------::::::---::::::--==--::---=++++====-::::::::::::::::::..........**......................+@@...
    ::::..........::::::::::::::::---:::--------:::....................::......#+......:.............=@@+...
    .......................:::::::::::::::::::::----=---------=========--::::..:%*..................-%@*....
    ...................................::::::::::::::---==---::::------:::::::...+#-...............+%@*.....
    ......................................:::::::::::::::::::::..::::::............#%=....::.....+%@@:......
    ..........................................::::::::::..............::::::::::::::-#@%*+===+*#@@%=........
    ...................................................:::....................:........:-+#@@@#+:...........
    ...........................................................................................::.....:::::-""")
        
    print("="*60)
    print("GROUP 1 - BIKE SHARING DATA")
    print("MIDTERM PROJECT")
    print("MEMBERS:")
    print("  - Jan Lancelot Mailig")
    print("  - Jocas Arabella Cruz")
    print("  - Eleazar James Galope")
    print("  - Jecho Torrefranca")
    print("  - John Neil Tamondong")
    print("="*60)
    
    curator = BikeDataCurator('hour.csv')
    
    if curator.load_data():
        curator.create_all_datasets()
        
        print("\n" + "="*60)
        print("ฅ^>⩊<^ ฅ RUNNING... ฅ^>⩊<^ ฅ")
        print("="*60)

        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)

        try:
            correlation_data = pd.read_csv('correlation.csv')
            correlator = CorrelationAnalyzer(correlation_data)
            correlator.generate_correlation_report()
            correlator.visualize_correlation_heatmap()
            correlator.visualize_rental_correlations()
        except FileNotFoundError:
            print("✗ correlation.csv not found.")
        except Exception as e:
            print(f"✗ Error in correlation analysis: {e}")
        
        print("\n" + "="*60)
        print("DESCRIPTIVE ANALYSIS")
        print("="*60)
        try:
            desc_df = pd.read_csv('descriptive.csv')
            if 'Section' in desc_df.columns:
                for sec in ['PLAN','BASIC_STATS','TABULATION','COMPARISON','DISTRIBUTION','OUTLIERS']:
                    sec_count = len(desc_df[desc_df['Section'] == sec])
                    print(f"{sec:<13}: {sec_count} rows")
            else:
                print(f"descriptive.csv loaded: {len(desc_df)} rows")

            try:
                basic_head = desc_df[desc_df['Section']=='BASIC_STATS'].head(3)
                if not basic_head.empty:
                    print("\nTop BASIC_STATS preview (first 3 rows):")
                    print(basic_head[['Variable','Mean','Median','Std']].to_string(index=False))
            except Exception:
                pass

            print("\n✓ Descriptive analysis summary printed.")
        except FileNotFoundError:
            print("✗ descriptive.csv not found.")

        print("\n" + "="*60)
        print("PREDICTIVE ANALYSIS")
        print("="*60)
        
        try:
            predictive_data = pd.read_csv('predictive.csv')
            predictor = PredictiveAnalyzer(predictive_data)
            
            predictor.prepare_data(test_size=0.2, random_state=42)
            
            predictor.train_linear_regression()
            predictor.train_decision_tree(max_depth=10)
            
            predictor.get_feature_importance()
            
            predictor.compare_models()
            
            predictor.save_prediction_metrics()
            predictor.save_prediction_results()
            
            predictor.visualize_predictions()
            
            print("\n✓ Predictive analysis complete!")
        except FileNotFoundError:
            print("✗ predictive.csv not found.")
        except Exception as e:
            print(f"✗ Error in predictive analysis: {e}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  Data files:")
        print("    - correlation.csv ✓")
        print("    - descriptive.csv ✓")
        print("    - predictive.csv ✓")
        print("    - predictive_data/prediction_metrics.csv ✓")
        print("    - predictive_data/prediction_results.csv ✓")
        print("\n  Visualization folders:")
        print("    - correlation_graphs/ (2 charts) ✓")
        print("    - descriptive_charts/ (12 charts) ✓")
        print("    - predictive_charts/ (12 charts) ✓")
        print("\n" + "="*60)