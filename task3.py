# Step 1: Load and Explore the Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Data exploration and visualization
sns.countplot(data['Class'])
plt.title('Fraudulent vs. Non-Fraudulent Transactions')
plt.show()

sns.heatmap(data.corr(), cmap="YlGnBu")
plt.title('Feature Correlation')
plt.show()

# Step 2: Data Preprocessing
data['Amount'] = (data['Amount'] - data['Amount'].mean()) / data['Amount'].std()
X = data.drop(columns=['Time', 'Class'])
y = data['Class']

# Handle class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Step 3: Build and Evaluate the Model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Build and tune the model
model = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)

# Evaluate the model
print("Best Parameters:", grid_search.best_params_)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
