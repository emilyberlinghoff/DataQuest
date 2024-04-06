
import pandas as pd
import numpy as np
import scipy.optimize as so
import itertools as it


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


#function to train model
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

categories = {
    "misc_net" : 1,
    "shopping_net" : 2,
    "misc_pos" : 3,
    "grocery_pos" : 4,
    "entertainment" : 5,
    "gas_transport" : 6,
    "personal_care" : 7,
    "shopping_pos" : 8,
    "food_dining" : 9,
    "home" : 10,
    "kids_pets" : 11,
    "grocery_net" : 12,
    "health_fitness" : 13,
    "travel" : 14
} 

train_data['transDate'] = pd.to_datetime(train_data['transDate'])
train_data['year'] = train_data['transDate'].dt.year
train_data['month'] = train_data['transDate'].dt.month
train_data['day'] = train_data['transDate'].dt.day

train_data["transDate_num"] = train_data['day']*10000 + train_data["month"]*100 + train_data["year"]

train_data.category = [categories[item] for item in train_data.category]

train_data["diff_Long"] = abs(train_data["longitude"] - train_data["merchLongitude"])

train_data["diff_Lat"] = abs(train_data["longitude"] - train_data["merchLongitude"])

list_var = ["category", "amount", "diff_Long", "diff_Lat"]

for max_node in [10, 100, 1000, 10000]:
    train_x, val_x, train_y, val_y = train_test_split(train_data[list_var], train_data["isFraud"], random_state=13)
    my_mae = get_mae(max_node, train_x, val_x, train_y, val_y)
    print(f"Max nodes {max_node} \t\t Mean Abs Error: {my_mae[0]} \t\t Avg Predict: {my_mae[1]}")


