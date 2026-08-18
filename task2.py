import sklearn
from sklearn.datasets import load_iris
#1.Load dataset
iris=load_iris()

#2. Understand Data
print("Features:",iris.feature_names)
print("Classes:",iris.target_names)
print("Number of samples:",len(iris.data))

#3. Seperate input and Output
X=iris.data
y=iris.target

#4. Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42) #The data is divided into 80% (training) and 20% (testing).
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#5. Create Classification
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)

#6. Train model
model.fit(X_train,y_train)

#7. Make predictions
predictions=model.predict(X_test)

#8. Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)

print("Predictions:", predictions)
print("Actual:", y_test)
print("Accuracy:", accuracy)