from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#Dataset
x=np.array([[22,25000],
            [25,27000],
            [28,32000],
            [30,30000],
            [45,50000],
            [50,60000],
            [55,65000]])

y=np.array([0,0,0,1,1,1,1])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(max_depth=2)
model.fit(x_train,y_train)

plt.figure(figsize=(10,6))
plot_tree(model,
          feature_names=["Age","Salary"],
          class_names=["No","Yes"],
          filled=True)

plt.show()