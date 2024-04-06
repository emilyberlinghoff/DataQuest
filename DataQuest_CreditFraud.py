
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_nodes, train_x, val_x, train_y, val_y):
    total = 0
    model = DecisionTreeRegressor(max_leaf_nodes = max_nodes, random_state = 13)
    model.fit(train_x, train_y)
    model_Predictions = model.predict(val_x)
    for item in model_Predictions:
        total += item
    results = total/len(model_Predictions)
    MAE = [mean_absolute_error(val_y, model_Predictions), results]

    return (MAE)

train = pd.read_csv("data_train.csv")
test = pd.read_csv("data_test.csv")

train_data = train.dropna()
test_data = test.dropna()

print(train_data.head())
print(test_data.head())

y = train_data.isFraud

