import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from xgboost import XGBRegressor

# Load dataset and trained models
df = pd.read_csv('Final_dataset.csv')
model_lr = pickle.load(open('lr_base_model.pkl', 'rb'))
model_xgb = pickle.load(open('xgb_simple_best_model.pkl', 'rb'))
model_rr = pickle.load(open('rr_model.pkl','rb'))

st.title("Aplikasi Prediksi Harga Mobil :car:")
merk_mobil = st.selectbox('Merk Mobil', [merk for merk in df['Merek'].unique()])
model_mobil = st.selectbox('Model', [model for model in df[df['Merek'] == merk_mobil]['Model'].unique()])
varian_mobil = st.selectbox('Varian', [varian for varian in df[df['Model'] == model_mobil]['Varian'].unique()])
tahun_mobil = st.select_slider('Tahun Produksi', [tahun for tahun in range(1985, 2020)])
transmisi_mobil = st.selectbox('Transmisi', ['Automatic', 'Manual', 'Automatic Triptonic'])
km_mobil = st.number_input('KM Mobil')


# Model selection
select_models = {
    'Linear Regression': model_lr,
    'XGBoost': model_xgb,
    'Random Forest': model_rr
}
model_name = st.selectbox('Pilih Model', list(select_models.keys()))

if 'text' not in st.session_state:
    st.session_state['text'] = ''
price_pred = st.text_input('Harga Prediksi', st.session_state['text'])

# Prediction function
def predict():
    row = np.array([merk_mobil, model_mobil, varian_mobil, tahun_mobil, transmisi_mobil, km_mobil])
    X = pd.DataFrame([row], columns=df.columns)
    
    if model_name == 'Linear Regression':
        model = model_rr
    elif model_name == 'XGBoost':
        model = model_xgb
    elif model_name == 'Random Forest':
        model = model_rr

    
    
    
    prediction = model.predict(X)[0]
    st.session_state['text'] = f'{int(prediction)} Juta'



predict_button = st.button('Predict', on_click=predict)


