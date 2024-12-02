import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('crop_yield.csv')
print("Summary Statistics:\n", df.describe())
print(df)

mean_yield = df['Yield_tons_per_hectare'].mean()
median_yield = df['Yield_tons_per_hectare'].median()
std_dev_price = df["Yield_tons_per_hectare"].std()

print(df['Region'].value_counts())
print(df['Soil_Type'].value_counts())
print(df['Crop'].value_counts())
print(df['Weather_Condition'].value_counts())

df.hist(bins=60, figsize =(20,10))
plt.show()

for column in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
 ax = sns.countplot(data=df, x=f'{column}', color='green')
 for p in ax.patches:
     ax.annotate(f'{p.get_height()}',
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center',
                 fontsize=12, color='black',
                 xytext=(0, 5), textcoords='offset points')
 plt.show()


correlation_matrix = df[['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest', 'Yield_tons_per_hectare']]
print(correlation_matrix.describe())
dataplot = sns.heatmap(correlation_matrix.corr(numeric_only=True), cmap="YlGnBu", annot=True)

for column in ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest', 'Yield_tons_per_hectare']:
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

for column in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
    sns.boxplot(data=df, x=f'{column}', y='Yield_tons_per_hectare', color='palegreen')
    plt.show()

for column in ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest', 'Yield_tons_per_hectare']:
    plt.hexbin(data=df, x=f'{column}', y='Yield_tons_per_hectare', cmap='PuBu')
    plt.title(f"Distribution of {column}")
    plt.xlabel(f'{column}', fontsize=14)
    plt.ylabel("Average Yield (tons per hectare)", fontsize=14)
    plt.colorbar(label='Density')
    plt.show()

sns.boxplot(data=df, x='Irrigation_Used', y='Yield_tons_per_hectare', color='dodgerblue')
plt.show()
sns.boxplot(data=df, x='Fertilizer_Used', y='Yield_tons_per_hectare', color='dodgerblue')
plt.show()

# Pivot data for heatmap
heatmap_data = df.pivot_table(
    index='Region',
    columns='Crop',
    values='Yield_tons_per_hectare',
    aggfunc='mean'
)

# Create heatmap
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f')

# Add title and labels
plt.title('Average Yield by Region and Crop')
plt.xlabel('Crop')
plt.ylabel('Region')
plt.show()

heatmap_data = df.pivot_table(
    index='Soil_Type',
    columns='Crop',
    values='Yield_tons_per_hectare',
    aggfunc='mean'
)

# Create heatmap
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f')

# Add title and labels
plt.title('Average Yield by Soil Type and Crop')
plt.xlabel('Crop')
plt.ylabel('Soil Type')
plt.show()

heatmap_data = df.pivot_table(
    index='Weather_Condition',
    columns='Crop',
    values='Yield_tons_per_hectare',
    aggfunc='mean'
)

# Create heatmap
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f')

# Add title and labels
plt.title('Average Yield by Weather Condition and Crop')
plt.xlabel('Crop')
plt.ylabel('Soil Type')
plt.show()

heatmap_data = df.pivot_table(
    index='Irrigation_Used',
    columns='Soil_Type',
    values='Yield_tons_per_hectare',
    aggfunc='mean'
)

# Create heatmap
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f')

# Add title and labels
plt.title('Average Yield by Weather Condition and Crop')
plt.xlabel('Crop')
plt.ylabel('Soil Type')
plt.show()




