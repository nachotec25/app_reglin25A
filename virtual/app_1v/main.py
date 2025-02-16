import streamlit as st
import pickle as pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

#Configuramos la página que mostrará nuestra app en streamlit
st.set_page_config(page_title="Precio diamantes",
                    page_icon="👩‍⚕️", 
                    layout="wide", 
                    initial_sidebar_state="expanded")
#!Función carga de datos en un dataframe que llamamos df.
def carga_datos():
    df=sns.load_dataset('diamonds')
    return df
#TODO Cargamos el modelo y el scaler serializados anteriormente.
scaler_1v = pickle.load(open("scaler_1v.pkl", "rb"))
model_1v=pickle.load(open("model_1v.pkl", "rb"))
#!Declaramos un sidebar donde colocaremos las características de nuestro diamante
#!El valor de cada característica será seleccionado o introducido por el usuario.
def add_sidebar():
  st.sidebar.header('Carcaterísticas del diamante')
  #declaramos datos para tener mi dataframe
  df=carga_datos()
  slider_labels=[
    ('carat','carat'),
  ]
  input_dict={}
  for label,col in slider_labels:
  #Guardamos los valores seleccionados para cada feature en un diciionario
    input_dict[col]=st.sidebar.slider(
      label,
      min_value=float(df[col].min()),
      max_value=float(df[col].max()),
      value=float(df[col].mean())   
    )
  return input_dict
#!Almacenamos los valores de las variables de entrada en la variable input_data
input_data=add_sidebar()
'''Función que nos permite visualizar la predicción en función de los valores
de las variables que seleccionamos''' 
def display_prediction(input_val,model_1v,scaler_1v):
   df=carga_datos()
   #mostrar prediccion
   input_val=np.array(list((input_data.values()))).reshape(1,-1)
   #input_data_scaled = scaler.transform(input_array)
   prediction = model_1v.predict(input_val)
   st.write('El precio del diamante es:')   
   st.error(prediction[0])
def coef_reg():
   inter=model_1v.intercept_
   coef=model_1v.coef_
   st.write('Intercepto del modelo__b0')
   st.success(inter) 
   st.write('Coeficiente del modelo__b1')
   st.warning(coef[0])
def lin_model():
   inter=model_1v.intercept_
   coef=model_1v.coef_
   df=carga_datos()
   x=df[['carat']]
   y=coef+inter*x
   st.write('Función del modelo') 
   fig = plt.figure(figsize=(10, 6))
   ax = fig.add_subplot(111)
   ax.plot(x, y, color='r')
   ax.set_xlabel('carat')
   ax.set_ylabel('price')
   plt.title('Distribución de carat')
   st.pyplot(fig)
def scatter_data():
   st.write('Distribución de descriptores')   
   df=carga_datos()
   x=df[['carat']]
   y=df['price']
   # Create a 3D scatter plot with Seaborn
   fig = plt.figure(figsize=(10, 6))
   ax = fig.add_subplot(111)
   ax.scatter(x, y, color='g')
   ax.set_xlabel('carat')
   ax.set_ylabel('price')
   plt.title('Distribución de carat')
   st.pyplot(fig)        
#TODO Esta es la función principal que construye la página de la APP
def main():
     #Mostramos en el body los valores seleccionados para las variables en el sidebar
  #st.write(input_data)
  #Establecemos el contenido del contenedor principal de la página
  with st.container(border=True):
     st.title('Predicción del precio de diamantes')
     st.write('En este caso sólo vamos a considerar una vaiable predictora del precios')
  col1,col2,col3=st.columns(3,border=True)
  with col1:
     coef_reg()
     #!Invocamos la función display_prediction() para que muestre el precio del diamante.
     display_prediction(input_data,model_1v,scaler_1v)
  with col2:
    scatter_data()
  with col3:
     lin_model()          
#Esto permite que se ejecute la función main al ejecutar el script
if __name__ == "__main__":
    main()
