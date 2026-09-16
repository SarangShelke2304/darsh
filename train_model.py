import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'decision_tree_model.pkl'
LOG_PATH = os.path.join(BASE_DIR, 'print_logs.xlsx')

def run_full_training():
    files = ['Plus real (0.3-11%).xlsx', 'real job dataset ( pakka wala ).xlsx', 
             'Sample data for 5mm (34_).xlsx', 'Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx', 
             'Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx']
    
    all_dfs = []
    for f in files:
        path = os.path.join(BASE_DIR, f)
        if os.path.exists(path):
            df = pd.read_excel(path)
            df.rename(columns={'Paper Type': 'Paper type', 'Delta E after ': 'Delta E after', 
                               'Initial Density': 'initial density', 'Initial Ink Key Setting': 'initial ink key setting'}, inplace=True)
            all_dfs.append(df)

    if os.path.exists(LOG_PATH):
        df_logs = pd.read_excel(LOG_PATH)
        df_logs.rename(columns={'Delta E before': 'Delta E before', 'Delta E after (Target)': 'Delta E after', 
                               'final ink key setting (ACTUAL)': 'final ink key setting'}, inplace=True)
        all_dfs.append(df_logs)

    data = pd.concat(all_dfs, ignore_index=True).dropna()
    data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
    data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
    data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
    data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

    # ZONE NUMBER REMOVED - Now using 6 features
    features = ['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
    X, y = data[features], data['final ink key setting']

    models = {
        'Linear': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(max_depth=12, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42),
        'GradientBoost': GradientBoostingRegressor(n_estimators=80, max_depth=6, learning_rate=0.1, random_state=42)
    }

    best_r2, best_model = -1, None
    for name, m in models.items():
        m.fit(X, y)
        score = r2_score(y, m.predict(X))
        if score > best_r2:
            best_r2, best_model = score, m

    joblib.dump(best_model, MODEL_NAME, compress=9)
    print(f"✅ Training Complete (6 Features) | Winner: {type(best_model).__name__} | R2: {best_r2:.4f}")

    # Sensitivity Test (6 values only)
    test_1 = np.array([[0, 0, 0.0, 5.0, 1.35, 43.6]]) 
    test_2 = np.array([[0, 0, 0.0, 5.0, 1.35, 25.0]]) 
    print(f"   43.6% Init -> {best_model.predict(test_1)[0]:.2f}%")
    print(f"   25.0% Init -> {best_model.predict(test_2)[0]:.2f}%")

if __name__ == "__main__":
    run_full_training()