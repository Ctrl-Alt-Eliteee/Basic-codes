from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data={
    "Age":[18,22,25,30,35,40,45,50],
    "Salary":[10000,20000,25000,40000,50000,60000,70000,80000],
    "Buys":[0,0,0,1,1,1,1,1]
}

df=pd.DataFrame(data)

x=df[["Age","Salary"]]
y=df["Buys"]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)

print("Train Score:", model.score(x_train, y_train))
print("Test Score:", model.score(x_test, y_test))