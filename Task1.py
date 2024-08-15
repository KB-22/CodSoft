# Step 1: Load and Explore the Data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('titanic.csv')

# Data exploration and visualization
sns.countplot(data['Survived'])
plt.title('Survival Count')
plt.show()

sns.countplot(data['Pclass'])
plt.title('Class Distribution')
plt.show()

sns.countplot(data['Sex'])
plt.title('Gender Distribution')
plt.show()

sns.histplot(data['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.show()

# Step 2: Data Preprocessing
data['FamilySize'] = data['SibSp'] + data['Parch']
data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
data.fillna(method='ffill', inplace=True)
data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Title'])

# Feature selection
features = ['Pclass', 'Sex_male', 'Age', 'FamilySize', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X = data[features]
y = data['Survived']

# Step 3: Build and Evaluate the Model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and tune the model
model = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)

# Evaluate the model
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
