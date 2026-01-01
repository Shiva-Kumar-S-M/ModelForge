import pandas as pd

def drop_unnecessary_columns(df):
    columns_to_drop=['PassengerId', 'Name', 'Ticket', 'Cabin']
    df=df.drop(columns=columns_to_drop)
    return df



#Handling missinf values
def handle_missing_values(df):
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


#Encoding categorical variables
def encode_categorical_variables(df):
    df=pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)
    return df
