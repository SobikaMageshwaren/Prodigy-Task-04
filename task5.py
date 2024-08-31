import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'cleaned.csv'  # Update with your actual dataset path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Data Cleaning and Preprocessing
# Handle missing values
df.fillna('Unknown', inplace=True)

# Convert categorical variables to numeric
categorical_columns = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Vehicle_driver_relation',
                        'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 'Light_conditions',
                        'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 'Pedestrian_movement',
                        'Cause_of_accident']

for col in categorical_columns:
    df[col] = df[col].astype('category').cat.codes

# Check for missing values again
print(df.isnull().sum())

# Analyze Patterns and Trends

# Plot accidents by age band of driver
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Age_band_of_driver', hue='Accident_severity', palette='viridis')
plt.title('Number of Accidents by Age Band of Driver')
plt.xlabel('Age Band of Driver')
plt.ylabel('Number of Accidents')
plt.legend(title='Accident Severity')
plt.show()

# Plot accidents by weather conditions
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Weather_conditions', hue='Accident_severity', palette='viridis')
plt.title('Number of Accidents by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.legend(title='Accident Severity')
plt.show()

# Plot accidents by road surface type
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Road_surface_type', hue='Accident_severity', palette='viridis')
plt.title('Number of Accidents by Road Surface Type')
plt.xlabel('Road Surface Type')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.legend(title='Accident Severity')
plt.show()

# Violin plot for accident severity by weather conditions
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='Weather_conditions', y='Accident_severity', palette='viridis')
plt.title('Accident Severity by Weather Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Accident Severity')
plt.xticks(rotation=45)
plt.show()


# violin plot for accident severity by road surface type
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='Road_surface_type', y='Accident_severity', palette='viridis')
plt.title('Accident Severity by Road Surface Type')
plt.xlabel('Road Surface Type')
plt.ylabel('Accident Severity')
plt.xticks(rotation=45)
plt.show()