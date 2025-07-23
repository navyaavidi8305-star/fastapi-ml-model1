import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
import joblib
data={
    'math':[78,65,78,45,30,60,80,40,20,50],
    'science':[20,68,45,82,63,81,20,40,60,60],
    'English':[30,70,45,65,98,42,30,30,40,70],
    'Result':['Fail','Pass','Fail','Fail','Fail','Pass','Fail','Fail','Fail','Pass']
}
df=pd.DataFrame(data)
#print(df)
df['Result']=df['Result'].map({'Pass':1,'Fail':0})
#train the model
x=df[['math','science','English']]
y=df['Result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
#70%i/p,30%i/p,70%o/p,30%o/p
#print(x_train)
#print(x_test)
#call the model
model=LogisticRegression()
model.fit(x_train,y_train)
#res=model.predict(x_test)
#from sklearn.metrics import accuracy_score
#res=model.predict(x_test)
#print("Accuracy",accuracy_score(y_test,res))
#new_student=pd.DataFrame([[60,90,80]],columns=['math','science','English'])
#predict=model.predict(new_student)
#print(predict[0])
joblib.dump(model,'model.pkl')
