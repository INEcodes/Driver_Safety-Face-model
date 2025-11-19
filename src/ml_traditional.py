"""Traditional ML training utilities (SVM, KNN, RF, NB, DecisionTree)."""
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report




def run_traditional_ml_models(X_train, X_test, y_train, y_test):
models = {
'SVM': SVC(kernel='rbf', probability=True),
'Naive Bayes': GaussianNB(),
'KNN': KNeighborsClassifier(n_neighbors=5),
'Decision Tree': DecisionTreeClassifier(random_state=42),
'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}


results = {}


for name, model in models.items():
print(f"Training {name}...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
results[name] = accuracy
print(f"{name} Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
print("="*50)


return results
