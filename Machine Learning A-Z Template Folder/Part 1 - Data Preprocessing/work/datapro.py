import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

data = pd.read_csv("../Data.csv")

x= data.iloc[:,:-1].values # : = all 
y= data.iloc[:,3].values

# missing data by taking mean
from sklearn.preprocessing import Imputer
#ImputerOfImputer = Imputer(missing_values="NaN", strategy="mean", axis=0) #taking mean of column
#ImputerOfImputer = ImputerOfImputer.fit(x[:,1:3]) # mean between lowerbound include, upperbound exclude
#x[:,1:3] = ImputerOfImputer.transform(x[:,1:3]) #change nan values
x[:,1:3]=Imputer(missing_values="NaN", strategy="mean", axis=0).fit(x[:,1:3]).transform(x[:,1:3])



#encode catagorical data or change catagory to number
from sklearn.preprocessing import LabelEncoder
x[:,0]=LabelEncoder().fit_transform(x[:,0])
y=LabelEncoder().fit_transform(y)

#dummy variable
from sklearn.preprocessing import OneHotEncoder
x=OneHotEncoder(categorical_features=[0]).fit_transform(x).toarray()
# no need for y because it is dependence variables



#splitting data into training and test
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)



#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

print("done")