import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import mplfinance as mpf
from data_preprocessing import TimeSeriesDataset
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
import os
from sklearn import metrics
version = "1.0"
st.set_page_config(
    page_title="STONKS",
    page_icon=":chart_with_upwards_trend:"   
)

with st.sidebar:
    #st.title("Sidebar")
    #st.radio("Navigation",["Visualise", "Train","About"])
    st.sidebar.success("We did it reddit!")

if 'started_analyis' not in st.session_state:
    st.session_state['plot_button'] = False
    st.session_state['ticker_button'] = False
    st.session_state['started_analysis'] = True

st.title("Stock direction prediction "+version)
st.write("Select ticker")
symbol = st.selectbox("SYMBOL:",('AAPL','AMZN','MSFT','GOOGL','TSLA','^GSPC','^DJI'))
ticker_cached = os.path.exists(symbol+'.pkl')
period = 'max'
interval = '1d'
if st.button("Reload"):
    st.session_state['ticker_button'] = True
    ticker_data = yf.download(tickers=symbol, period=period, interval=interval, prepost=True, rounding=False)
    ticker_data.to_pickle(symbol+'.pkl')
if ticker_cached:
    ticker_data = pd.read_pickle(symbol+'.pkl')
    window_size = st.number_input("Window size:",min_value = 1,max_value = len(ticker_data),value = 5)
    sample_n = st.number_input("Intervals to extract:",min_value = window_size+2,max_value = len(ticker_data),value=len(ticker_data))
    data = TimeSeriesDataset(ticker_data).sample(sample_n,window_size=window_size)
    c1, c2 = st.columns([1, 1])
    c1.write("Raw ticker data: "+str(ticker_data.shape))
    c1.dataframe(ticker_data.head(5))
    c2.write("Processed train set: "+str(data.train.shape))
    c2.dataframe(pd.DataFrame(data.train).head(5))
    ## PLOT CANDLES##
    mc = mpf.make_marketcolors(up='g',down='r')
    s  = mpf.make_mpf_style(marketcolors=mc)
    fig, ax = mpf.plot(ticker_data,type='line',style=s,show_nontrading=False, figscale = 0.6,figratio = (12,4),returnfig=True)
    st.pyplot(fig)
    ##HYPERPARAMETERS
    hp_c = st.number_input("C:",min_value = 0.0,value = 1.0)
    ##SETUP, TRAIN, EVALUATE##
    scaler = StandardScaler()   
    scaler.fit(data.train)
    c_weights = 1/(data.occurences/np.sum(data.occurences))
    c_weights = c_weights/c_weights.min()
    weight_dict = {i-1:c_weights[i] for i in range(len(c_weights)) }
    classifier = SVC(kernel = 'rbf',C=hp_c,class_weight=weight_dict)
    model = classifier.fit(scaler.transform(data.train), data.y_train)
    score_train = model.score(scaler.transform(data.train), data.y_train)
    score_val = model.score(scaler.transform(data.val), data.y_val)
    st.write("R2 Train:",score_train,"R2 Val:",score_val)
    prediction = model.predict(scaler.transform(data.val))
    alt_accuracy = np.array([1 if (prediction[i] == data.y_val[i]) or (prediction[i] == data.y_val_pre[i]) else 0 for i in range(len(prediction))])
    class_acc = metrics.accuracy_score(data.y_val, prediction, normalize=True, sample_weight=None)
    st.write("Classification accuracy:","{:.2f}%".format(100*class_acc))
    st.info("Classification accuracy is the percentage of correctly matching target labels guessed",icon="ℹ️")
    effec_acc = np.sum(alt_accuracy)/len(alt_accuracy)
    st.write("Effective accuracy: {:.2f}%".format(100*effec_acc))
    st.info("Effective accuracy counts cases where the classifier wrongfully guessed a direction when the magnitude was too low but the actual direction matched", icon="ℹ️")
    error = prediction != data.y_val