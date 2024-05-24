import xgboost as xgb
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2, random_state=42)

dtest = xgb.DMatrix(X_test)

model = xgb.Booster()
model.load_model("deploy/assets/xgboost_model.json")

raw_preds = model.predict(dtest)

preds = np.round(raw_preds).astype(int)

print(preds.shape)

print(f'Predictions', preds)
