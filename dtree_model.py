
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load and Combine all your datasets
files = [
    'Plus real (0.3-11%).xlsx', 
    'real job dataset ( pakka wala ).xlsx', 
    'Sample data for 5mm (34_).xlsx', 
    'Extended data of plus offset(11_), 2.7mm(24_) and 3.9mm(33_) and 1.3mm(15_) and 0.6mm(7_) (1).xlsx', 
    'Extended data of plus offset, 2.7mm and 3.9mm (1).xlsx'
]

all_dfs = []
for f in files:
    try:
        df = pd.read_excel(f)
        all_dfs.append(df)
    except: continue

data = pd.concat(all_dfs, ignore_index=True).dropna()

# Mapping
data['Color'] = data['Color'].map({'Cyan': 0, 'Magenta': 1, 'Yellow': 2, 'Black': 3})
data['Paper type'] = data['Paper type'].map({'Coated': 0, 'Uncoated': 1})
data['Ink key zero setting'] = data['Ink key zero setting'].astype(str).str.replace("mm","",regex=False).astype(float)
data['Delta E improvement'] = data['Delta E before'] - data['Delta E after']

# 6 Specific Features
features = ['Color', 'Paper type', 'Ink key zero setting', 'Delta E improvement', 'initial density', 'initial ink key setting']
X = data[features]
y = data['final ink key setting']

# Train Model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)

# Save
joblib.dump(dt_model, 'decision_tree_model.pkl')
print(f"✅ Model Saved! R2: {r2_score(y_test, dt_model.predict(x_test)):.4f}")