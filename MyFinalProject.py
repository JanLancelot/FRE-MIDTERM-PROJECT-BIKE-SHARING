import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


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
        # Dito ka Jek
        return False
    
    def create_descriptive_dataset(self, output_path='descriptive.csv'):
        # Dito ka Elea
        return False
    
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
    
    def create_all_datasets(self):
        print("\n=== Creating Sub-Datasets ===")
        # self.create_correlation_dataset()
        # self.create_descriptive_dataset()
        self.create_predictive_dataset()
        print("=== Dataset creation complete ===\n")


class CorrelationAnalyzer:
    
    def __init__(self, data):

        self.data = data

class DescriptiveAnalyzer:
    
    def __init__(self, data):
        self.data = data
    

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
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test))
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
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_test))
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
        print(comparison)
        
        best_model = 'Linear Regression' if self.results['lr']['test_r2'] > self.results['dt']['test_r2'] else 'Decision Tree'
        print(f"\n✓ Best performing model: {best_model}")
        
        return comparison
    
    def visualize_predictions(self, save_path='prediction_comparison.png'):
        if self.lr_model is None or self.dt_model is None:
            print("✗ Please train both models first")
            return
        
        lr_pred = self.lr_model.predict(self.X_test)
        dt_pred = self.dt_model.predict(self.X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(self.y_test, lr_pred, alpha=0.5, s=10)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Rentals')
        axes[0].set_ylabel('Predicted Rentals')
        axes[0].set_title(f'Linear Regression (R² = {self.results["lr"]["test_r2"]:.3f})')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(self.y_test, dt_pred, alpha=0.5, s=10, color='green')
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                     [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Rentals')
        axes[1].set_ylabel('Predicted Rentals')
        axes[1].set_title(f'Decision Tree (R² = {self.results["dt"]["test_r2"]:.3f})')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Prediction comparison chart saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    print(""" *************************++++++++++++++++++++++++++++========================---------------:::.....
************************+++++++++++++++++++++++++++========================---------------::::......
*******************+++++++++++++++++++++++++++++==========================-----------------::::.....
*********+*++++++++++++++++++++++++++++++++++++============================-----------------::::....
+++++++++++++++++++++++++++++++++++++++++++++===========================---------------------:::::::
+++++++++++++++++++++++++++++++++++++++++==============================----------------------:::::::
+++++++++++++++++++++++++++++++++++++++==============================------------------------:::::::
++++++++++++++++++++++++++++++++==================================----------------------------------
+++++++++++++++++++++++++++++++++================================-----------------------------------
+++++++++++++++++++++++++++++++++++==============================-----------------------------------
++++++++++++++++++++++++++++++++++=============================----------------::-------:::::::-:---
++++++++++++++++++++++++++++++++++=====================-----------------------::------::::::::::::::
***********+++++++++++++++++++=====+++++=====================---------------------------------------
###***********************************++++++++++++++++++++===============---------------------------
########*********************************+++*+++++++++++++++++++====================================
##############***************************#*####*###***+++++++++++++=================================
#################***********************###**###*+++***++++++++++===+++++===========================
#################*#*********************##*=+##+=+=++=*++++++++====+++++++++========================
####################********************#****#*+**++*+++++++++===+++++++++++++======================
######################*******************++==++===++===++++==+++++++++++++++++++++++++++++++++++++++
###########################**************++*+++++++*++=+====++++++++++++++++++++++++++++++++++++++++
#######***++++++***############**************++++++++=====++++++++++++++++++++++++++======++++++++++
########*########**+++++++*#########*******+***+++++==+=++++++++++++++++++++==----==++=--==+++=+++++
#####################**+++====+++**#*******#****+++++++++++++++++++++==------===++++++++++++++++++++
%#########################********############*+*+++*+=-+**********++++===++++++++++++++++++++++++++
%%%%###############################%##########********+=*******#*##**+***+++++++++++++++++++++++++++
%%%%%%%%%##########################%##########**###**++***##*######*****************++++************
%%%%%%%%%#%####################################**##**++++*#######***********************************
%%%%%%%%%%%%%%%######################**########*##*****++#####**************************************
%%%%%%%%%%%%%%%%%%%##################****######******#######*++*************************************
%%%%%%%%%%%%%%%%%%%%%#################****#######***#######*++**************************************
%%%%%%%%%%%%%%%%%%%%%%%#################***################*+***************************************
%%%%%%%%%%%%%%%%%%%####################%#***%%############*+*####***********************************
%%%%%%%%%%%%%%%%%%%%%%#################%##***############***########********************************
%%%%%%%%%%%%%%%%%%%%%#################******++++++**####***###############**************************
%%%%%%%%%%%%%%%%%#########*************###%%++++++****##*#####+**********#**************************
%%%%%%%%%%%%%%####**##########%%##*#%#######+++***##**#########***###########***********************
%%%%%%%#####%%%%##################*#########++++*****##########***############+########*************
%%%%%%%%%%%%%###########################%%%#******#####%%%%%%%#***########****+***##****************
%%%%%#%%#####################%%%############*****#############****############+****#********++******
%%%%%%%%%%%%%%%%%%%#%%%#####%%%%%%#%%%%%%%%%*****###%##%%%%%%%####%###########+#############*#######
%%%%%#%####%%%%##%%%%%%%%%%%%%%%%%#%%%%%##%%****########*##%%%#*##%%%%%%%#####*#############**######
############%%%%%%%%%%%%%%%%%%%%%%#%%%##%%%%****######%#####%#################*#############**######
%%%%##%%###%%%%%%%%%%%%%%%%%%%%%%%%%%#####%########%####*#########%%%%%%%%%%##*#############**######
%%%%####%%%%%%%%%%%%%@@@@%%%%%%%%%################%%%###**#########%%%%%%%%%%%*#%%###%######**##%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%@@@%%%%%%#############%%%%%%%%%%%%%%%%%%%%%%%%%%%%#*#######%%##%#*#%%%%%%
%%%%%%%%%%%%%%%@@%%%@@@@%%%%%%%%%%%%%%%%%%%%%%%##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*#%%%%#######%**#%%%%%
%%%%%%%%%%%%@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%###%%%%%%%%%%%%%%%%%%%%%%%%%#####*###%%%%%####%*###%%%#
%%%%%%%%%%%%%%%%%@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#########*##%%%%%
%%%%%%%@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%######
@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%""")
    print("="*60)
    print("GROUP 1 - BIKE SHARING DATA")
    print("MIDTERM PROJECT")
    print("MEMBERS:")
    print("  - Jan Lancelot P. Mailig")
    print("  - Jocas Arabella Cruz")
    print("  - Eleazar Galope")
    print("  - Jecho Parairo Torrefranca")
    print("="*60)
    
    curator = BikeDataCurator('hour.csv')
    
    if curator.load_data():
        curator.create_all_datasets()
        
        print("\n" + "="*60)
        print("𐔌՞. .՞𐦯...RUNNING...𐔌՞. .՞𐦯")
        print("="*60)
        
        # Insert your print here guys for correlation and descriptive analyses
        
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
        # print("    - correlation.csv ✓")
        # print("    - descriptive.csv ✓")
        print("    - predictive.csv ✓")
        print("\n  Visualization files:")
        print("    - prediction_comparison.png ✓")
        print("\n" + "="*60)