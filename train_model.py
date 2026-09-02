import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. Sintetik Datasheet
np.random.seed(42)
num_samples = 1000

ndvi = np.random.uniform(0.2, 0.9, num_samples)
ndwi = np.random.uniform(-0.1, 0.5, num_samples)
soil_moisture = np.random.uniform(10, 40, num_samples)
precipitation_30d = np.random.uniform(0, 150, num_samples)
field_size = np.random.uniform(10, 200, num_samples)

# Scoring
score = (ndvi * 40) + (ndwi * 30) + (precipitation_30d * 0.1)
moisture_effect = np.where((soil_moisture >= 25) & (soil_moisture <= 35), 20, 
                           np.where(soil_moisture < 20, -10, -5))
score += moisture_effect
score = np.clip(score, 0, 100)
field_efficiency_score = np.round(score, 1)

df = pd.DataFrame({
    'field_size_ha': np.round(field_size, 1),
    'ndvi': np.round(ndvi, 3),
    'ndwi': np.round(ndwi, 3),
    'soil_moisture_pct': np.round(soil_moisture, 1),
    'precipitation_30d_mm': np.round(precipitation_30d, 1),
    'field_efficiency_score': field_efficiency_score
})

# 2. Training
X = df[['field_size_ha', 'ndvi', 'ndwi', 'soil_moisture_pct', 'precipitation_30d_mm']]
y = df['field_efficiency_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

# 3. Saving model as a file
model.get_booster().save_model("model/agriflow_xgboost.json")
print("Successful! Model has been learned and saved as a 'model/agriflow_xgboost.json' ")

