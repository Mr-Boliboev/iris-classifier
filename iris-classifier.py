from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ma'lumotlarni yuklash
iris = load_iris()
X = iris.data
y = iris.target

# Train/Test bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression modelini qurish va o'qitish
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Bashorat va aniqlikni baholash
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model aniqligi: {accuracy * 100:.2f}%")
