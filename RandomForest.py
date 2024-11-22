from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('crop_yield.csv')
print(df)

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df, one_hot_df], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)
df_encoded.head()

df_encoded['Fertilizer_Used'] = df['Fertilizer_Used'].map({True: 1.0 , False: 0.0})
df['Irrigation_Used'] = df['Irrigation_Used'].map({True: 1.0 , False: 0.0})
X = df_encoded.drop(columns=['Yield_tons_per_hectare'])
y = df_encoded[['Yield_tons_per_hectare']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
print(f"Predictions: {predictions[:5]}")
print(f"True Values: {y_test[:5]}")
print(f"Model Accuracy (R^2): {regressor.score(X_test, y_test)}")

importances = regressor.feature_importances_

# Sort the feature importances
indices = np.argsort(importances)[::-1]

# Plot the feature importances
import shap

# Initialize the explainer
explainer = shap.TreeExplainer(regressor)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Visualize the SHAP values
shap.summary_plot(shap_values, X_test)