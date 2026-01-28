
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

parquet_extension = ".parquet"
monthly_data_file= "monthly_data" + parquet_extension
monthly_inflation_data_file = "monthly_inflation_data" + parquet_extension


#  *****************************************************************************
#  load_appartement_file
#  *****************************************************************************

def load_appartement_file (start_path, immocv_file) :
    final_path = start_path / immocv_file
    return pd.read_parquet(final_path.as_posix())

#  *****************************************************************************
#  display_pandemic
#  *****************************************************************************
def display_pandemic ():
    plt.axvline(datetime.datetime(2020, 1,9 ), color='red', linewidth=3, linestyle='-')
    bottom,top = plt.ylim()
    ypos = bottom + (top-bottom)*0.15
    plt.text (datetime.datetime(2019, 6,1 ), ypos,"start pandemic")

    plt.axvspan(datetime.datetime(2020, 3,17), datetime.datetime(2020, 5,11 ), color='red',alpha=0.4)
    plt.axvline(datetime.datetime(2020, 3,17 ), color='red', linewidth=3, linestyle='--')
    plt.axvline(datetime.datetime(2020, 5,11 ), color='red', linewidth=3, linestyle='--')

    plt.axvspan(datetime.datetime(2020, 10,30 ), datetime.datetime(2020,12,15), color='red',alpha=0.4)
    plt.axvline(datetime.datetime(2020, 10,30 ), color='red', linewidth=3, linestyle='--')
    plt.axvline(datetime.datetime(2020, 12,15 ), color='red', linewidth=3, linestyle='--')

    plt.axvspan(datetime.datetime(2021, 4,3 ), datetime.datetime(2021, 5,3 ), color='red',alpha=0.4)
    plt.axvline(datetime.datetime(2021, 4,3 ), color='red', linewidth=3, linestyle='--')
    plt.axvline(datetime.datetime(2021, 5,3 ), color='red', linewidth=3, linestyle='--')

    plt.axvline(datetime.datetime(2023, 1,1 ), color='red', linewidth=3, linestyle='-')
    plt.text (datetime.datetime(2023, 1,1 ), ypos,"end pandemic")

#  *****************************************************************************
#  display_prediction_window
#  *****************************************************************************

def display_prediction_window() :
    plt.axvspan(datetime.datetime(2024, 7,1 ), datetime.datetime(2025, 12,1 ), color='pink',alpha=0.4)

#  *****************************************************************************
#  def display_extra_data() :
#  *****************************************************************************
def display_extra_data() :
    plt.axvspan(datetime.datetime(2025, 3,1 ), datetime.datetime(2025, 12,1 ), color='orange',alpha=0.4)


#  *****************************************************************************
#  display_data
#  *****************************************************************************

def display_data(df) :
    
    scaled_annonce = df["scaled_nb_annonce"]*1000 + df.prix_m2_vente.min()

    fig = plt.figure()
    title = "prix vente m2 appartement over " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
    plt.title (title)
    plt.ylabel('prix vente m2')
    plt.xlabel('date')

    plt.plot(df.prix_m2_vente,label="prix_vente au m2")
    plt.plot(scaled_annonce,label="scaled nombre d'annonces")
    plt.legend()

    display_pandemic()

    display_extra_data()

    st.pyplot(fig)

#  *****************************************************************************
#  display_inflation_data
#  *****************************************************************************

def display_inflation_data (inflation,df) :

    prix_scaled = (df.prix_m2_vente -df.prix_m2_vente.min()) / (df.prix_m2_vente.max()-df.prix_m2_vente.min())
    prix_scaled = prix_scaled*inflation.inflation.max()
    print (prix_scaled)

    nb_annonce = df.scaled_nb_annonce*inflation.inflation.max()

    fig = plt.figure()

    plt.plot (inflation.date, inflation.inflation,label="inflation rate",linewidth=2,color="blue")
    plt.plot (inflation.date, inflation.taux_livret_A,label="taux livret A",color="purple")
    plt.plot (inflation.date, inflation.taux_BCE,label="taux BCE",color="pink")
    plt.plot (inflation.date, inflation.taux_pret_20ans,label="taux pret 20 ans",color="green")
    plt.plot (inflation.date, prix_scaled,label="scaled prix vente m2",linewidth=2,color="red")
    plt.plot (inflation.date, nb_annonce,label="scaled nb annonce",linewidth=2,color="grey")

    plt.title ("inflation rate, investment rate and loan rate")
    plt.ylabel("rate in %")

    plt.fill_between(inflation.date,prix_scaled -0.35, prix_scaled+0.35, facecolor='C0', alpha=0.4)

    display_pandemic()

    display_extra_data()
    plt.legend()

    st.pyplot(fig)


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
#  plot_predictions
#  *****************************************************************************

def plot_predictions (forecasts,test_data,length) :


    fig = plt.figure()
    plt.plot(test_data['ds'],test_data["y"],label="test")
    for key, forecast in forecasts.items() : 
        plt.plot(test_data['ds'],forecast["yhat"].iloc[-length:],label=key)

    plt.ylabel('prix vente m2')
    plt.xlabel('Months')
    plt.xticks(test_data['ds'], rotation=45)

    if 'yhat_lower' in forecast.columns :
        plt.fill_between(test_data['ds'], forecast['yhat_lower'].tail(length), forecast['yhat_upper'].tail(length), 
                        color='orange', alpha=0.15, label='Intervalle de confiance (80%)')
    plt.legend()

    st.pyplot(fig)


#  *****************************************************************************
#  display_prophet_predictions
#  *****************************************************************************

def display_prophet_predictions (df, length = 18) :

    df3 = df.rename(columns={"date" : "ds","prix_m2_vente":"y"})

    train_data = df3[:-length]
    test_data = df3[-length:]

    model = Prophet (changepoint_prior_scale=0.05)
    model.add_seasonality(name='monthly', period=12, fourier_order=5)

    model.fit (train_data)

    future = model.make_future_dataframe(periods=length,freq="MS")

    forecast = model.predict(future)

    fig, ax = plt.subplots()
    model.plot(forecast,ax=ax, ylabel = "prix vente m2")
    display_extra_data()

    st.pyplot(fig)

    y_true = test_data['y'].values
    y_pred = forecast['yhat'][-length:].values


    forecasts = {"Prophet prediction": forecast}
    plot_predictions(forecasts,test_data,length)

    mae, rmse, mre, r2 = calculate_metrics(y_true,y_pred,"Prophet")

    return forecast


#  *****************************************************************************
#  display_prophet_predictions
#  *****************************************************************************

def display_prophet_inflation_predictions (df, length=18) :

    train_data = df[:-length]
    test_data = df[-length:]


    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )

    model.add_seasonality(name='monthly', period=12, fourier_order=5)

    model.add_regressor('inflation')
    model.add_regressor('taux_livret_A')
    model.add_regressor('taux_BCE')
    model.add_regressor('taux_pret_20ans')

    attributes = ['ds', 'y', 'inflation','taux_livret_A','taux_BCE','taux_pret_20ans']
    local_frame = train_data[attributes]
    model.fit (local_frame)

    future = model.make_future_dataframe(periods=length,freq="MS")
    future = test_data[attributes].copy()

    forecast1 = model.predict(future)

    y_true = test_data['y'].values
    y_pred = forecast1['yhat'][-length:].values


    fig, ax = plt.subplots()
    model.plot(forecast1, ax=ax, ylabel = "prix vente m2",plot_cap=True)

    plt.legend()

    st.pyplot(fig)


    forecasts = {"Prophet prediction with inflation": forecast1}
    plot_predictions(forecasts,test_data,length)

    regressor_coefficients = {
        'inflation': model.extra_regressors['inflation']['prior_scale'],
        'taux livret A': model.extra_regressors['taux_livret_A']['prior_scale'],
        'taux BCE': model.extra_regressors['taux_BCE']['prior_scale'],
        'taux pret 20 ans': model.extra_regressors['taux_pret_20ans']['prior_scale']
    }
    fig = plt.figure()

    plt.bar(regressor_coefficients.keys(), regressor_coefficients.values())
    plt.ylabel('Prior Scale')
    plt.title('Regressor Prior Scales')
    plt.tick_params(axis='x', rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    mae_rates, rmse_rates, mre_rates, r2_rates = calculate_metrics(y_true,y_pred,"Prophet with inflation and loan rates")

    st.pyplot(fig)

    fig = model.plot_components(forecast1)
    st.pyplot(fig)

    
    return forecast1

#  *****************************************************************************
#  main
#  *****************************************************************************
def merge_data_inflation (df,inflation) :
    df3 = df.rename(columns={"date" : "ds","prix_m2_vente":"y"})
    df3 = pd.merge(df3,inflation,left_index=True,right_on="date")
    df3.set_index(df.index,inplace=True)
    df3.drop("date",axis=1,inplace=True)

    return df3

#  *****************************************************************************
#  display_data
#  *****************************************************************************
def display_exponential_predictions(df,length) :


    train_data = df[:-length]
    test_data = df[-length:]

    y_true = test_data['y']

    hw_model = ExponentialSmoothing(
        train_data['y'],
        trend='add',
        seasonal='add'
    )
    hw_fit = hw_model.fit()
    hw_pred = hw_fit.forecast(steps=len(test_data))

    hw_pred = hw_pred.to_frame(name='yhat')

    forecasts = {"ExponentialSmoothing": hw_pred}
    plot_predictions(forecasts,test_data,length)

    mae_exp, rmse_exp, mre_exp, r2_exp = calculate_metrics(y_true,hw_pred,"ExponentialSmoothing with inflation and loan rates")

    return hw_pred

#  *****************************************************************************
#  LSTM create_sequences
#  *****************************************************************************

def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, :])
        y.append(data[i+lookback, 0])  # target = y
    return np.array(X), np.array(y)

#  *****************************************************************************
#  LSTM prepare_lstm_data
#  *****************************************************************************

def prepare_lstm_data(df, features, lookback=30):
    """Prepare data for LSTM with multiple features"""
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # create the data
    X,y = create_sequences(scaled_data,lookback)

    return X, y, scaler


#  *****************************************************************************
#  LSTM prepare_lstm_data
#  *****************************************************************************

from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def lstm_forecast(df, features, length= 0, lookback=30):
    """Forecast using LSTM neural network"""
    st.write ("\n" + "=" * 50)
    st.write ("LSTM NEURAL NETWORK FORECASTING")
    st.write ("=" * 50)
    
    # Prepare data
    X, y, scaler = prepare_lstm_data(df, features, lookback)

    with tf.device('/cpu:0'):
        X = tf.convert_to_tensor(X, np.float32)
        y = tf.convert_to_tensor(y, np.float32)
        
    # Split data
    if length == 0 :
       train_size  = int(len(X) * 0.8)
    else :
        train_size = -length
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
 
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, 
             input_shape=(lookback, X.shape[2])),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics =['mae']
    )

    st.write (model.summary())

    
    # Train model
    st.write("Training LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    # Make predictions on test set
    pred_scaled = model.predict(X_test)

    # Re-scale predictions
    dummy = np.zeros((len(pred_scaled), len(features)))
    dummy[:, 0] = pred_scaled[:, 0]

    y_pred = scaler.inverse_transform(dummy)[:, 0]

    
    return model, X_test, y_test, y_pred, history


#  *****************************************************************************
#  LSTM prepare_lstm_data
#  *****************************************************************************
def display_lstm_predictions (df, length) :

    features = ['y', 'inflation', 'taux_livret_A', 'taux_BCE', 'taux_pret_20ans']

    tf.config.set_visible_devices([], '')

    with tf.device('/CPU:0'):

        model, X_test, y_test, y_pred, history = lstm_forecast(df=df,features=features,length=length,lookback=30)

        data = {
            'time' :  df.index[-len(y_pred)],
            'yhat' : y_pred
        }

        test_data = df[-length:]
        y_true = test_data['y']
        y_pred_lstm = pd.DataFrame(data)

        forecasts = {"LSTM prediction": y_pred_lstm}
        plot_predictions(forecasts,test_data[-len(y_pred_lstm):],length)

        # mae_lstm, rmse_lstm, mre_lstm, r2_lstm = calculate_metrics(y_true,y_pred_lstm,"LSTM prediction model")

#  *****************************************************************************
#  main
#  *****************************************************************************

current_dir = Path(__file__).parent

df = load_appartement_file(current_dir,monthly_data_file)

inflation = load_appartement_file(current_dir,monthly_inflation_data_file)


st.title("Prédiction du prix de vente au m2 des appartements")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Prédiction"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.write("### Introduction")

    st.dataframe(df.head(10))
    st.dataframe(inflation.head(10))

    if st.checkbox("Afficher les NA") :
        st.dataframe(df.isna().sum())

if page == pages[1] : 
    title = "Visualization des data  sur la période " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
    st.write(title)

    choices = ['prix de vente au m2', 'variables économiques']
    option = st.selectbox('Choix de la visualisation', choices,index=0)

    if option == choices[0] :
        display_data(df)
    else :
        display_inflation_data (inflation,df)



if page == pages[2] : 
    title = "Prediction du prix au m2 des appartements " + df.index[0].strftime('%Y-%m') + " - " + df.index[0-1].strftime('%Y-%m')
    st.write(title)
    length = 18

    test_data = test_data = df[-length:]
    df_merge = merge_data_inflation(df,inflation)

    choices = ['Prophet', 'Prophet et variables économiques','exponential smooting predictions','LSTM prediction']

    option = st.selectbox('Choix de la prédiction', choices,index=0)
    if option == choices[0] :
        foecast = display_prophet_predictions(df, length)
    elif option == choices[1] :
        foecast_inflation = display_prophet_inflation_predictions(df_merge, length)
    elif option == choices[2] :
        forecast_exponential = display_exponential_predictions(df_merge, length)
    elif option == choices[3] :
        forecast_lstm = display_lstm_predictions(df_merge, length)