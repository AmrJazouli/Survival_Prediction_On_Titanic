


import os
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import HistGradientBoostingClassifier



df_train = pd.read_csv('train.csv')


# import csv files
def create_dataframe(filename, sep, verbose=True):
    """Create pandas dataframe from a csv file
filename: the csv filename
sep: input here the kind of separator
    """
    df = pd.read_csv(filename, sep=sep)
    if verbose:
        print(df.info())
    return df


def categorise_SibSp(row):
    """
    Cateogorise the SibSp column into 3 categories
    2: if has more than 1 sibling
    1: if has exactly one sciblin
    0: if has no siblings
    """

    temp_dict = {1:"Couple", 0:"Single"}

    if row in temp_dict.keys():
        return temp_dict[row]
    else:
        return "Family"


if __name__=="__main__":
    
    df_test = create_dataframe("test.csv", ",")
    df_train = create_dataframe("train.csv", ",")

    print()
    print("  We are here 0")
    print()
    print(len(df_train))
    #print(df_train["Cabin"].value_counts())
    #print(df_train.columns)
    #print(df_train[["Survived", "Age","SibSp", "Parch", "Pclass", "Fare"]].describe())
    print(df_train[df_train["Fare"]> 500])

    df = df_train.copy()
    y = df['Survived']

    #Check for people with a fare tickets higher than 500
    df_train_rich = df_train[df_train["Fare"]> 500]
    #df_train_rich.to_csv("train_fare_500.csv")

    #Check the Fare for people which were in cabin category B
    df_train_B = df_train[df_train["Cabin"].str[0] =="B"]
    print(df_train[df_train["Cabin"].str[0] =="B"].describe())
    # It seems one is at 0, let's check for this one

    print(df_train["Cabin"].str[0].value_counts())

    df_train["cabin_class"] = df_train["Cabin"].str[0]
   # print(df_train)
   #  df_train["Fare"].hist(by=df_train['cabin_class'])
   #  plt.show()

    #Create a dummy dataframe from the cabin class
    df_dummy_cabin = pd.get_dummies(df_train['cabin_class'], prefix='cabin_class')
    print(df_dummy_cabin)

     


    #Replace the data in the corresponfing dataframe
    df_train = pd.concat( [df_train , df_dummy_cabin], axis=1)


    # Distribution of SibSp variable
    print(df_train["SibSp"] )

    

    # df_train["Survived"].hist(by=df_train['SibSp'])
    #df_train["SibSp"].hist()
    #plt.show()
    # New set of columns
    print(df_train.columns)

    # Categorise the sibs column using lambda
    df_train["SibSp_group"]  = df_train["SibSp"].apply(lambda x: categorise_SibSp(x))
    print(df_train["SibSp_group"] )

    #Chance of survival by the family status
    df_train["Survived"].hist(by=df_train['SibSp_group'])
    plt.show()

    



    # Ratio of survivals:
    survivals =  df_train["Survived"].value_counts()
    print(survivals)


    df_train.info()

    df_train.isnull().sum()


    df_train["Cabin"].str[0].unique()

    df_train.drop(['PassengerId',"Name","Ticket","Cabin"],axis=1,inplace=True)

    df_train_encodedSex = pd.get_dummies(df_train['Sex'],prefix='EncodedSex')


    df_train= pd.concat([df_train,df_train_encodedSex],axis=1)
   

    df_train_encodedEmbarked = pd.get_dummies(df_train['Embarked'],prefix='EncodedEmbarke')


    df_train= pd.concat([df_train,df_train_encodedEmbarked],axis=1)


    df_train.head()


    df_train.info()


    df_train.drop('cabin_class',axis=1,inplace=True)


    df_train['Age'].mean()


    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())


    df_train['Age'] = df_train['Age'].astype(int)


    df_train.head()


    df_train.columns


    df_train.drop(["Embarked","Sex","SibSp"],axis=1,inplace=True)


    df_train_encodedSibSp = pd.get_dummies(df_train['SibSp_group'],prefix='EncodedSibSp')



    df_train =pd.concat([df_train,df_train_encodedSibSp],axis=1)



    df_train = df_train.drop('SibSp_group',axis=1)


    df_train.head()


    
    scaler = StandardScaler()


    df_train_norm = scaler.fit_transform(df_train)

    df_train_norm


    df_train_norm.shape
       
    columns = df_train.columns
    columns

    df1 = pd.DataFrame(data=df_train_norm, columns=columns )
    df1.head()

    X = df1.drop('Survived',axis=1)

    
    #clf = LogisticRegression(random_state=0).fit(X, y)
    clf = HistGradientBoostingClassifier().fit(X, y)

    print()
    print(" Prediction")
    print()
    print(clf.predict(X.iloc[:20, :]))

    #print(y[:2])

    
