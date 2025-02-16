import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle as pickle

def carga_datos():
    df=sns.load_dataset('diamonds') 
    return df
def crear_modelo(df):
#Elegimos las variables de entrada y salida
    X=df[['carat']]
    y=df['price']
    #Escalado de características
    scaler_1v=StandardScaler()
    X=scaler_1v.fit_transform(X)
    #Split de datos
    X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=0.2,random_state=42)
    #train
    print(X_test.size)
    model_1v=LinearRegression()
    model_1v.fit(X_train,y_train)
    #Evaluación
    y_pred=model_1v.predict(X_test)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    rmse = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
    print(f"Coeficiente de determinación R^2:", model_1v.score(X, y))
    print(f"El error (MSE) de test es: {mse}")
    print(f"El error (RMSE) de test es: {rmse}")
    input_dict={
        'carat':0.2,
        }
    
    X_in=np.array(list((input_dict.values()))).reshape(1,-1)
    prediccion=model_1v.predict(X_in)
    print(prediccion)
    return model_1v,scaler_1v
def main():
    df=carga_datos()
    model_1v,scaler_1v =crear_modelo(df)

    #Aquí montamos nuestra app en streamlit

    with open('app_1v/model_1v.pkl','wb') as f:
        pickle.dump(model_1v,f)
    with open('app_1v/scaler_1v.pkl','wb') as g:
        pickle.dump(scaler_1v,g)
if __name__ == "__main__":
    main()

