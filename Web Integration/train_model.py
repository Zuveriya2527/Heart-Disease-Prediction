import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Example dataset (simple dummy data)
data = {
    "age":[45,50,60,30,25,55,65,40],
    "bmi":[25,30,28,22,20,27,31,24],
    "bp":[130,140,150,120,110,145,160,135],
    "cholesterol":[200,240,260,180,170,250,270,210],
    "smoke":[1,1,0,0,0,1,1,0],
    "diabetes":[0,1,1,0,0,1,1,0],
    "activity":[1,0,0,2,2,1,0,1],
    "target":[1,1,1,0,0,1,1,0]
}

df = pd.DataFrame(data)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,y_train)

pickle.dump(model, open("models/heart_model.pkl","wb"))

print("Model trained and saved successfully!")