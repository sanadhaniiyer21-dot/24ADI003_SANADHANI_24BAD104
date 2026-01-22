Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
# Load dataset
df_house = pd.read_csv("housing.csv")
 
# Inspect dataset
print(df_house.head())
print(df_house.info())
 
# Check missing values
print(df_house.isnull().sum())
 
# Scatter plot – Area vs Price
plt.scatter(df_house['area'], df_house['price'])
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()
 
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df_house.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
