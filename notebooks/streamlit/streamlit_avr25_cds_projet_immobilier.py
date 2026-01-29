
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from streamlit_prevision_app import flat_display_exponential_predictions, flat_display_lstm_predictions, flat_display_monthly_data, flat_display_monthly_inflation_data, flat_display_prophet_inflation_predictions, flat_display_prophet_predictions, flat_merge_data_inflation, flat_plot_predictions

parquet_extension = ".parquet"
monthly_data_file= "monthly_data" + parquet_extension
monthly_inflation_data_file = "monthly_inflation_data" + parquet_extension

PROJECT_TITLE = "Projet immobilier - Modélisation des prix des maisons et appartements - France Métropolitaine"
PAGES = ["Présentation","Visualisation","Analyse ACP","Modélisation","Prédiction en temps","Prédiction du prix","Conclusion"]
FLAT_NAME = "appartement"
HOUSE_NAME = "maison"
HOUSE_FLAT_CHOICE = [HOUSE_NAME,FLAT_NAME]
REGRESSION_MODEL_LGB = "LgbRegressor"
REGRESSION_MODEL_DECISION_TREE = "DecisionTreeRegressor"
REGRESSION_MODEL_XGB = "XGBRegressor"

REGRESSION_MODEL_CHOICE = [REGRESSION_MODEL_LGB,REGRESSION_MODEL_DECISION_TREE,REGRESSION_MODEL_XGB]

current_dir = Path(__file__).parent

data_dir = current_dir / "data"

#  *****************************************************************************
#  load_appartement_file
#  *****************************************************************************

def load_appartement_file (start_path, immocv_file) :
    final_path = start_path / immocv_file
    return pd.read_parquet(final_path.as_posix())


#  *****************************************************************************
#  calculate_metrics
#  *****************************************************************************

def calculate_metrics(actual, predicted, model_name):
    mae = mean_absolute_error(actual,predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mre = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)

    
    st.write(f"\n{model_name} performance  Metrics:")
    st.write(f"  MAE :  {mae:.2f}")
    st.write(f"  RMSE: {rmse:.2f}")
    st.write(f"  MRE : {mre:.2f}%")
    st.write(f"  R2  : {r2:.4f}")

    return mae, rmse, mre, r2


#  *****************************************************************************
#  main
#  *****************************************************************************

st.title(PROJECT_TITLE)
st.sidebar.title("Sommaire")
pages = PAGES
page=st.sidebar.radio("", pages)

#  *****************************************************************************
#  Page : Présentation
#  *****************************************************************************
if page == pages[0] : 
    st.write("### Présentation")

#  *****************************************************************************
#  Page : Visualisation des data
#  *****************************************************************************
if page == pages[1] :
    title = "Visualization et traitement sur les données"
    st.write(title)

    house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
    if house_flat == HOUSE_NAME :
         st.write("")
    elif house_flat == FLAT_NAME :
         st.write("")

#  *****************************************************************************
#  Page : ACP
#  *****************************************************************************
if page == pages[2] : 
    st.write("### Analyse en Composantes Principales")

#  *****************************************************************************
#  Page : Modelisation
#  *****************************************************************************
if page == pages[3] :
    title = "Modélisation du prix de vente au m2 des appartements et des maisons"
    st.write(title)

    house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
    model_type = st.selectbox('Choix du modèle', REGRESSION_MODEL_CHOICE,index=0)


#  *****************************************************************************
#  Page : Time prediction
#  *****************************************************************************
if page == pages[4] : 

    # load data
    df = load_appartement_file(data_dir,monthly_data_file)
    inflation = load_appartement_file(data_dir,monthly_inflation_data_file)
 
    title = "Prediction en temps du prix au m2 des appartements sur la période " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
    st.write(title)

    tab1, tab2 = st.tabs(["Visualisation", "Prediction"])
    with tab1 :
        title = "Visualization des data  sur la période " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
        st.write(title)

        choices = ['prix de vente au m2', 'variables économiques']
        option = st.selectbox('Choix de la visualisation', choices,index=0)

        if option == choices[0] :
            flat_display_monthly_data(df)
        else :
            flat_display_monthly_inflation_data (inflation,df)
    with tab2 :
        title = "Prediction du prix au m2 des appartements"
        st.write(title)
        length = 18

        df_merge = flat_merge_data_inflation(df,inflation)
        test_data = df_merge[-length:]

        choices = ['Prophet', 'Prophet et variables économiques','exponential smooting predictions','LSTM prediction','Summary']
        option = st.selectbox('Choix de la prédiction', choices,index=0)
        if option == choices[0] :
            forecast = flat_display_prophet_predictions(df, length)
            st.session_state['forecast'] = forecast
        elif option == choices[1] :
            forecast_inflation = flat_display_prophet_inflation_predictions(df_merge, length)
            st.session_state['forecast_inflation'] = forecast_inflation
        elif option == choices[2] :
            forecast_exponential = flat_display_exponential_predictions(df_merge, length)
            st.session_state['forecast_exponential'] = forecast_exponential
        elif option == choices[3] :
            forecast_lstm = flat_display_lstm_predictions(df_merge, length)
            st.session_state['forecast_lstm'] = forecast_lstm
        elif option == choices[4] :
            forecasts = {"Exponential smoothing": st.session_state['forecast_exponential'], 
                            "LSTM prediction" : st.session_state['forecast_lstm'], 
                            "Prophet": st.session_state['forecast'], 
                            "Prophet with inflation" : st.session_state['forecast_inflation']}
            flat_plot_predictions(forecasts,test_data,length)

#  *****************************************************************************
#  Page : prediction du prix
#  *****************************************************************************
if page == pages[5] : 
    title = "Prediction du prix ou d'un appartemment"
    st.write(title)
    house_flat = st.selectbox('Type de bien', HOUSE_FLAT_CHOICE,index=0)
    if house_flat == HOUSE_NAME :
         st.write("")
    elif house_flat == FLAT_NAME :
         st.write("")

#  *****************************************************************************
#  Page : Conclusion
#  *****************************************************************************
if page == pages[len(pages)-1] : 
    st.write("### Conclusion")
