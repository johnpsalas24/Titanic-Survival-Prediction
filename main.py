import data_loader
import data_visualization
import model

# Load and preprocess the data
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = data_loader.load_data(url)

# Visualize the data
data_visualization.visualize_data(data)

# Define features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = pd.get_dummies(data[features])
y = data['Survived']

# Train the model
trained_model, X_test, y_test = model.train_model(X, y)

# Make predictions and evaluate the model
predictions = trained_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, predictions))
