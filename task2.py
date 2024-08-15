# Step 1: Load and Explore the Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('sales.csv')

# Data exploration and visualization
sns.pairplot(data)
plt.show()

# Step 2: Data Preprocessing
data.fillna(method='ffill', inplace=True)

# Feature selection
X = data[['Advertising']]
y = data['Sales']

# Step 3: Build and Evaluate the Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Sales vs Advertising')
plt.xlabel('Advertising')
plt.ylabel('Sales')
plt.show()
