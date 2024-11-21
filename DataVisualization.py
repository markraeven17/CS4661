import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('crop_yield.csv')
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)
print(df)

mean_yield = df['Yield_tons_per_hectare'].mean()
median_yield = df['Yield_tons_per_hectare'].median()
std_dev_price = df["Yield_tons_per_hectare"].std()

print(df['Region'].value_counts())
print(df['Soil_Type'].value_counts())
print(df['Crop'].value_counts())
print(df['Weather_Condition'].value_counts())

for column in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
 sns.countplot(data=df, x=f'{column}', color='green')
 plt.show()

# Display the results
print(f"Mean Crop Yield: {mean_yield:.2f} tons per hectare")
print(f"Median Crop Yield: {median_yield:.2f} tons per hectare")
print(f"Standard Deviation: {std_dev_price:.2f}")

print(df[['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest', 'Yield_tons_per_hectare']].describe())

plt.figure(figsize=(40, 24))
sns.histplot(df['Yield_tons_per_hectare'], bins=10, kde=True, color='green')
plt.title('Histogram of Crop Yield with Density Plot')
plt.xlabel('Crop Yield (tons per hectare)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Create the box plot
sns.boxplot(x='Region', y='Yield_tons_per_hectare', data=df)

# Add title and labels
plt.title('Yield Distribution per Region')
plt.xlabel('Region')
plt.ylabel('Yield (tons per hectare)')

# Display the plot
plt.xticks(rotation=45)
plt.show()

# Calculate the average yield for each weather condition
avg_yield_by_weather = df.groupby('Crop')['Yield_tons_per_hectare'].mean().reset_index()

# Plot the average yield for each weather condition
plt.figure(figsize=(8, 6))
sns.barplot(x='Crop', y='Yield_tons_per_hectare', data=avg_yield_by_weather, palette='coolwarm')

# Add title and labels
plt.title('Average Yield per Crop')
plt.xlabel('Crop')
plt.ylabel('Average Yield (tons per hectare)')

# Annotate each bar with the average yield value
for index, row in avg_yield_by_weather.iterrows():
    plt.text(
        x=index,
        y=row['Yield_tons_per_hectare'] + 0.05,  # Adjust the y position slightly above the bar
        s=f"{row['Yield_tons_per_hectare']:.2f}",  # Format to 2 decimal places
        ha='center',
        color='black'
    )

plt.show()

# Calculate the average yield for each weather condition
avg_yield_by_weather = df.groupby('Weather_Condition')['Yield_tons_per_hectare'].mean().reset_index()

# Plot the average yield for each weather condition
plt.figure(figsize=(8, 6))
sns.barplot(x='Weather_Condition', y='Yield_tons_per_hectare', data=avg_yield_by_weather, palette='coolwarm')

# Add title and labels
plt.title('Average Yield per Hectare by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Yield (tons per hectare)')

# Annotate each bar with the average yield value
for index, row in avg_yield_by_weather.iterrows():
    plt.text(
        x=index,
        y=row['Yield_tons_per_hectare'] + 0.05,  # Adjust the y position slightly above the bar
        s=f"{row['Yield_tons_per_hectare']:.2f}",  # Format to 2 decimal places
        ha='center',
        color='black'
    )

plt.show()

plt.figure(figsize=(10, 6))
plt.hexbin(df["Rainfall_mm"], df["Yield_tons_per_hectare"], gridsize=50, cmap='YlGnBu', alpha=0.7)
plt.title("Hexbin Plot of Rainfall(mm) vs. Yield", fontsize=16)
plt.xlabel("Rainfall (mm)", fontsize=14)
plt.ylabel("Yield (tons per hectare)", fontsize=14)

# Add a color bar to show density
plt.colorbar(label='Density')
plt.show()

plt.figure(figsize=(10, 6))
plt.hexbin(df["Temperature_Celsius"], df["Yield_tons_per_hectare"], gridsize=50, cmap='YlGnBu', alpha=0.7)
plt.title("Hexbin Plot of Temperature(C) vs. Yield", fontsize=16)
plt.xlabel("Temperature(C)", fontsize=14)
plt.ylabel("Yield (tons per hectare)", fontsize=14)

# Add a color bar to show density
plt.colorbar(label='Density')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Fertilizer_Used', y='Yield_tons_per_hectare', data=df, palette='Set2')

# Add titles and labels
plt.title("Crop Yield by Fertilizer Usage", fontsize=16)
plt.xlabel("Fertilizer Used (0 = No, 1 = Yes)", fontsize=14)
plt.ylabel("Average Yield (tons per hectare)", fontsize=14)

# Display the plot
plt.show()
