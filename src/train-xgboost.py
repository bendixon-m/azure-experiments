import tarfile
import os
from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# create model instance
model_xgboost = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
print('Fitting model')
model_xgboost.fit(X_train, y_train)
# make predictions
print('Making predictions')
preds = model_xgboost.predict(X_test)
print(f'Predictions', preds)
print("Mean squared error: %.2f" % mean_squared_error(y_test, preds))

print('Saving model')
file_name = './deploy/assets/xgboost_model'
model_xgboost.save_model(file_name)
tar_name = file_name + ".tar.gz"

with tarfile.open(tar_name, "w:gz") as tar:
    # Add the file to the tarball
    tar.add(file_name, arcname=os.path.basename(file_name))
    print(f"{file_name} has been archived and compressed into {tar_name}")