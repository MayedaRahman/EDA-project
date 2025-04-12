# 1. Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# 2. Load Dataset
data = pd.read_csv("C:\\Users\\abhin\OneDrive\Desktop\mayeda project python\COVID-19_Outcomes_by_Vaccination_Status_-_Historical.csv")

# 3. Data Cleaning
# Convert date column
if 'Week End' in data.columns:
    data['Week End'] = pd.to_datetime(data['Week End'], errors='coerce')

# Fill missing numeric values with mean
numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Fill missing object values with mode
object_columns = data.select_dtypes(include=['object']).columns
for col in object_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# 4. Dataset Info
print("\nDataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# 5. Visualizations
# Bar Plot: Outcomes by Age Group
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='Age Group', hue='Outcome')
plt.title('Outcomes by Age Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram: Unvaccinated Rate
plt.figure(figsize=(10, 5))
sns.histplot(data['Unvaccinated Rate'], bins=30, kde=True)
plt.title('Distribution of Unvaccinated Rate')
plt.xlabel('Unvaccinated Rate')
plt.tight_layout()
plt.show()

# Pie Chart: Outcome Distribution
plt.figure(figsize=(6, 6))
outcome_counts = data['Outcome'].value_counts()
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Outcome Distribution')
plt.show()

# Boxplot: Boosted Rate by Outcome
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Outcome', y='Boosted Rate')
plt.title('Boosted Rate by Outcome')
plt.tight_layout()
plt.show()

# Line Plot: Vaccinated Rate over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=data.sort_values('Week End'), x='Week End', y='Vaccinated Rate', hue='Outcome')
plt.title('Vaccinated Rate Over Time by Outcome')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Heatmap: Correlation Matrix
plt.figure(figsize=(14, 10))
corr_matrix = data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 6. Regression Analysis: Predicting Death Rate from Case Rate
if 'Unvaccinated Rate' in data.columns and 'Boosted Rate' in data.columns:
    X = data[['Unvaccinated Rate']]
    y = data['Boosted Rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    print("\nLinear Regression Evaluation:")
    print("Coefficient:", reg_model.coef_)
    print("Intercept:", reg_model.intercept_)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    plt.figure(figsize=(8, 5))
    sns.regplot(x='Unvaccinated Rate', y='Boosted Rate', data=data, line_kws={"color": "red"})
    plt.title('Regression: Unvaccinated Rate vs Boosted Rate')
    plt.tight_layout()
    plt.show()

# 7. 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['Unvaccinated Rate'], data['Vaccinated Rate'], data['Boosted Rate'], c=data['Boosted Rate'], cmap='plasma')
ax.set_xlabel('Unvaccinated Rate')
ax.set_ylabel('Vaccinated Rate')
ax.set_zlabel('Boosted Rate')
ax.set_title('3D Scatter Plot: Vaccination Rates')
plt.show()
