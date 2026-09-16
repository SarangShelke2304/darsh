import joblib
import os

# 1. Load the giant 258MB model
print("Loading model...")
model = joblib.load('consolidated_ink_key_model.pkl')

# 2. Save it with MAX compression (9)
print("Compressing to the limit...")
joblib.dump(model, 'consolidated_ink_key_model.pkl', compress=9)

# 3. Check the size
new_size = os.path.getsize('consolidated_ink_key_model.pkl') / (1024 * 1024)
print(f"✅ New File Size: {new_size:.2f} MB")