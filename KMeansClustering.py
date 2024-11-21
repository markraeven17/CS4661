import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('crop_yield.csv')
print(df.head())
y = df[['Yield_tons_per_hectare']]

categorical_features = ['Region', 'Soil_Type', 'Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition']
numerical_features = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

# Combine transformers into a column transformer
preprocessor = ColumnTransformer(transformers=[
                                ('num', numerical_transformer, numerical_features),
                                ('cat', categorical_transformer, categorical_features)]
)

preprocessed_data = preprocessor.fit_transform(df)

encoded_cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_columns = list(numerical_features) + list(encoded_cat_columns)

preprocessed_df = pd.DataFrame(preprocessed_data, columns=all_columns)

# Display the first few rows
pd.set_option('display.max_columns', None)
print(preprocessed_df.head())
y = df['Yield_tons_per_hectare']
X_train, X_test, y_train, y_test = train_test_split(preprocessed_df, y, test_size=0.3, random_state=42)

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(preprocessed_df)

print(cluster_labels)
u_labels = np.unique(cluster_labels)


plt.figure(figsize=(24, 16))

for i in u_labels:
    plt.scatter(preprocessed_df.iloc[cluster_labels == i, 0], preprocessed_df.iloc[cluster_labels == i, 1] , label = i)

plt.xlim(preprocessed_df.iloc[:, 0].min() - 1, preprocessed_df.iloc[:, 0].max() + 1)
plt.ylim(preprocessed_df.iloc[:, 1].min() - 1, preprocessed_df.iloc[:, 1].max() + 1)
plt.legend()
plt.show()



