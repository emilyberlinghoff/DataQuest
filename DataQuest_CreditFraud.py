
#importing packages
import pandas as pd
import numpy as np

#importing functions from sklearn.model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


#function to train model
def get_mae(max_nodes, train_x, val_x, train_y, val_y):
    total = 0
    model = DecisionTreeRegressor(max_leaf_nodes = max_nodes, random_state = 13)
    model.fit(train_x, train_y)
    model_Predictions = model.predict(val_x)
    for item in model_Predictions:
        total += item
    results = total/len(model_Predictions)
    MAE = [mean_absolute_error(val_y, model_Predictions), mean_squared_error(val_y, model_Predictions), results]
    return (MAE)

#loding in csv files
train = pd.read_csv("data_train.csv")
test_data = pd.read_csv("data_test.csv")

#dropping all invalid rows
train_data = train.dropna()

#Converting categories to numerical value
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

#making function to manipulating train_data variables
def manip_data(data):
    data['transDate'] = pd.to_datetime(data['transDate'])
    data['year'] = data['transDate'].dt.year
    data['month'] = data['transDate'].dt.month
    data['day'] = data['transDate'].dt.day

    data["transDate_num"] = data['day']*10000 + data["month"]*100 + data["year"]

    data.category = [categories[item] for item in data.category]

    data["diff_Long"] = abs(data["longitude"] - data["merchLongitude"])

    data["diff_Lat"] = abs(data["latitude"] - data["merchLatitude"])

    data["distance"] = round(((data["diff_Long"]**2) + (data["diff_Lat"])**2)**0.5, 4)
    
#making and manipulating train_data and test_data variables
manip_data(train_data)
manip_data(test_data)

#Listing out variables used in training model
list_var = ["category", "amount", "distance"]

#printing accuracy of model 
print("Testing model accuracy")
for max_node in [10, 100, 1000, 10000]:
    train_x, val_x, train_y, val_y = train_test_split(train_data[list_var], train_data["isFraud"], random_state=13)
    my_mae = get_mae(max_node, train_x, val_x, train_y, val_y)
    print(f"Max nodes {max_node} \t Mean Abs Error: {my_mae[0]} \t Mean Sqr Error: {my_mae[1]} \t Avg Predict: {my_mae[2]}")

#getting model and predicting fraud test_data
model = DecisionTreeRegressor(max_leaf_nodes = 1000, random_state = 13)
model.fit(train_x train_y)
test_data["isFraud"] = model.predict(test_data[list_var])
print(test_data.head(20))
