import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the data 
data = np.loadtxt('data.txt')

X = data[:, :-1]  # Features (landmarks)
y = data[:, -1]   # Labels (emotions)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, shuffle=True, stratify=y)

# initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=21)


# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(confusion_matrix(y_test, y_pred))

pickle.dump(model, open('emotion_model.pkl', 'wb'))