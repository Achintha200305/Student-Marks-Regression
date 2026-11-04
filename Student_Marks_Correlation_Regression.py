import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
#Load the student marks dataset
df=pd.read_csv("Student_Marks.csv")
#display dataset and descriptive statistic
print(df.to_string(index=False))

print(df.describe())
#Check missing Value
print(df.isna().sum())
#Calculate correlation between numeric variables
print(df.corr(numeric_only=True))

x=df["Study_Hours"]
y=df["Exam_Marks"]

plt.plot(x,y,marker="o",label="Student Marks",color="blue",linestyle="--")
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Exam Marks vs Student Hours")
plt.scatter(x,y,marker="o",label="Student Marks",color="blue")
plt.show()
# Select independent and dependent variables
x=df[["Study_Hours"]]
y=df["Exam_Marks"]
# Create and train the Linear Regression model
model=LinearRegression()
model.fit(x,y)
# Display slope, intercept, and R-squared
print("Slope:",round(model.coef_[0],2))
print("Intercept:",round(model.intercept_,2))
print("R_Squared:",round(model.score(x,y),2))
plt.scatter(x,y,color="blue",label="Student Marks")
plt.plot(x,model.predict(x),color="red",label="Linear Regression")

plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Linear Regression")
plt.legend()
plt.show()

result=pd.DataFrame({"Actual_Marks":y,
                     "Predicted_Marks":model.predict(x),})

print(round(result,2).to_string(index=False))
result["Gap"]=result["Predicted_Marks"]-result["Actual_Marks"]
print(round(result,2).to_string(index=False))






