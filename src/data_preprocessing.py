import pandas as pd

def drop_unnecessary_columns(df):
    columns_to_drop=['PassengerId', 'Name', 'Ticket', 'Cabin']
    df=df.drop(columns=columns_to_drop,errors='ignore')
    return df



#Handling missinf values
def handle_missing_values(df):
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


#Encoding categorical variables
def encode_categorical_variables(df):
    df=pd.get_dummies(df,columns=['Sex','Embarked',"AgeGroup"],drop_first=True)
    return df


#Feature engineering 
def feature_engineering(df):

    # 1. Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # 2. Is Alone
    df["IsAlone"] = 1
    df.loc[df["FamilySize"] > 1, "IsAlone"] = 0

    # 3. Age Group
    df["AgeGroup"] = pd.cut(df["Age"],
                           bins=[0, 12, 20, 40, 60, 100],
                           labels=["Child", "Teen", "Adult", "MidAge", "Senior"])

    return df
