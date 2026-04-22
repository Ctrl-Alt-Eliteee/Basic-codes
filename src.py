from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

x = np.array([[18,15000],
              [22,20000],
              [35,40000],
              [45,60000],
              [50,80000]])

y = np.array([0,0,1,1,1])

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

print("Actual:", y_test)
print("Predicted:", y_pred)

# Predict new value
print("Prediction for [30,35000]:", model.predict([[30,35000]]))