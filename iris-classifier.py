from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
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
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

sns.pairplot(df, hue='target')
plt.savefig('images/pairplot.png')  # Saqlab qo‘yish
plt.show()
print(f"Model aniqligi: {accuracy * 100:.2f}%")

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
