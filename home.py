import streamlit as st
import yfinance as yf
import pandas as pd 
import plotly.express as px
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import os 
import matplotlib.pyplot as plt
import time
import numpy as np

info_s = """ebitdaMargins
profitMargins
grossMargins
revenueGrowth
operatingMargins
sector
earningsGrowth
currentRatio
returnOnAssets
numberOfAnalystOpinions
debtToEquity
returnOnEquity
totalCashPerShare
revenuePerShare
quickRatio
enterpriseToRevenue
enterpriseToEbitda
forwardEps
heldPercentInstitutions
priceToBook
shortRatio
beta
earningsQuarterlyGrowth
priceToSalesTrailing12Months
pegRatio
forwardPE
trailingPE
dividendYield
trailingPegRatio""".split('\n')

def get_data(tick_str,info_s):
    ticker = yf.Ticker(tick_str)
    # from info 
    info_dict = ticker.info

    #check if exists 
    if not info_dict.get('exchange'):
        print(tick_str + " Does Not Exist")
        return

    new_dict = {k:v for k,v in info_dict.items() if k in info_s}
    df = pd.DataFrame(new_dict,index=[0])
    
    #from financials 
    financials = pd.DataFrame(ticker.financials.iloc[:,0]).T.reset_index(drop=True)
    try:
        r_and_d = financials['Research Development']/financials['Total Revenue']
        sga = financials['Selling General Administrative'] / financials['Total Revenue']
    except:
        r_and_d = 0
        sga=0
    df["R&D/Revenue"] = r_and_d
    df["SGA/Revenue"] = sga

    #from major holders
    holders = ticker.get_major_holders()
    holders["Number of Institutions Holding Shares"] = int(holders.iloc[3][0])
    
    # insert ticker string 
    df['Ticker'] = tick_str
    return df 

def update_data(stocks):
    main = pd.DataFrame()
    for stock in stocks:
        df = get_data(stock,info_s)
        try:
            main = pd.concat([main,df],axis=0)
        except:
            main.to_csv('data.csv',mode='a',index=False)
            return
    #write to csv
    main.to_csv('data.csv',mode='a',index=False) 
    return main

## Function that plots daily high price over time
def plot_graph_over_time(dataframe):
    fig,ax=plt.subplots()
    ax.plot("High", data = dataframe)
    ax.set_title("Daily High and Volume over Time")
    ax.tick_params(axis = 'x', labelrotation = 45)
    ax.set_ylabel("Daily High (USD)", color = "blue")
    ax2 = ax.twinx()
    ax2.plot("Volume", data = dataframe, color = "red")
    ax2.set_ylabel("Volume", color = "red")
    ax.legend()
    ax2.legend()
    st.pyplot(fig=fig)

## Function that plots the net income over time
def plot_net_income(dataframe):
    fig,ax = plt.subplots()
    ax.plot("Net Income", data = dataframe)
    ax.set_title("Net Income over Time")
    ax.set_xticks(rotation = 45)
    st.pyplot(fig=fig)

def get_recco_cf(ticker):
    tick = yf.Ticker(ticker)
    recommendations = tick.recommendations
    df_recco = pd.DataFrame(recommendations)
    cashflow = tick.cashflow
    df_cashflow = pd.DataFrame(cashflow).T
    historical = pd.DataFrame(tick.history(start = "2021-06-01", end = "2022-02-26"))
    return df_recco, df_cashflow, historical


st.set_page_config(layout="wide")

st.title('Stock Selection Dashboard')
st.markdown('''Hello! Welcome to our Stock Dashboard! Thanks to the [yfinance package](https://pypi.org/project/yfinance/)
and the Yahoo Finance API, you will be able to find the following features here:
1. Find similar stocks: our algorithm uses various financial ratios to determine similarity of business model. This can be used during the process of portfolio planning. It can also paint a clearer picture of a business other than its sector and industry.
2. Get latest price data and analysts recommendations
3. Various visualisations of a stock's performance ''')

st.sidebar.header('User Input')
type_select = st.sidebar.selectbox('Select Universe of Stocks',['S&P 100','S&P 500'])
path = type_select.replace(' ','') +'data.csv'
#get list of stocks 
with open(type_select.replace(' ','')+'.txt') as f:
    stocks = f.readlines()
    stocks = [x.replace('\n','') for x in stocks]

selection = st.sidebar.selectbox("Please Select a Ticker",stocks)

st.sidebar.write('A higher learning rate would lead to the data being more spread out')
learning_rate = st.sidebar.selectbox('Choose Desired Learning Rate',np.arange(2,450,10))


if st.sidebar.button('Find'):
    #data preprocessing 
    data = pd.read_csv(path)
    data.fillna(value=0,inplace=True)
    train = data.drop(['Ticker','sector'],axis=1)
    tickers = data['Ticker']
    sectors = data['sector']

    #model
    # Create a normalizer: normalizer
    normalizer = Normalizer()

    tsne = TSNE(learning_rate=learning_rate)

    # Make a pipeline chaining normalizer and kmeans: pipeline
    pipeline = make_pipeline(normalizer,tsne)

    # Fit pipeline to the daily price movements
    tsne_features = pipeline.fit_transform(train)

    # Select the 0th feature: xs
    xs = tsne_features[:,0]

    # Select the 1th feature: ys
    ys = tsne_features[:,1]

    to_plot = pd.DataFrame({'x':xs,'y':ys,'Ticker':tickers,'Sector':sectors})
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=to_plot['x'],y=to_plot['y'],text=to_plot['Ticker'],mode='markers',marker=dict(size=12)))
    selection = to_plot[to_plot['Ticker'] == selection ]
    fig.add_trace(go.Scatter(x=selection['x'],y=selection['y'],text = selection['Ticker'],mode='markers',marker=dict(size=45,color='rgba(0, 0, 0, 0)',line=dict(color='red',width=3))))
    fig.update_layout(template="simple_white",width=2400, height=900,title = "Universe of Stocks", font=dict(
        family="Arial",
        size=18,
        color="Black"
    ) )
    fig.update_layout(
    title={
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_traces(showlegend=False)
    fig.update_xaxes(visible=False)   
    fig.update_yaxes(visible=False)

    st.plotly_chart(fig, use_container_width=True)

    df_recco, df_cashflow, historical = get_recco_cf(str(selection['Ticker'].iloc[0]))

    latest = historical.iloc[[-1]].iloc[: , :5].round(decimals=2)
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(latest.columns),
                fill_color='black',
                align='left', font=dict(color='white', size=20)),
    cells=dict(values=latest.transpose().values.tolist(),
               fill_color='white',
               align='left',font_size=20)
    )])
    st.write(fig)

    df_recco = df_recco.tail(10)
    fig = go.Figure(data=[go.Table(
    header=dict(values=list(df_recco.columns),
                fill_color='black',
                align='left', font=dict(color='white', size=20)),
    cells=dict(values=df_recco.transpose().values.tolist(),
               fill_color='white',
               align='left',font_size=20,height=70)
    )])
    st.write(fig)

    #historical
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.plot("High", data = historical)
    ax.set_title("Daily High and Volume over Time")
    ax.tick_params(axis = 'x', labelrotation = 45)
    ax.set_ylabel("Daily High (USD)", color = "blue")
    ax2 = ax.twinx()
    ax2.plot("Volume", data = historical, color = "red")
    ax2.set_ylabel("Volume", color = "red")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    st.pyplot(fig)
    #cash flow
    fig1,ax1 = plt.subplots(figsize=(10,5.5))
    ax1.plot("Net Income", data = df_cashflow)
    ax1.set_title("Net Income over Time")
    ax1.tick_params(axis = 'x', labelrotation = 45)
    st.pyplot(fig1)


st.markdown('''This app was created by Drama Club (Lim Wei Hern, Poon Zhe Xuan, Chow Shi Kai, Yeo Zong Yao) ''')
st.markdown('*This app is purely for educational and entertainment purposes only and should not be taken as financial advice.*')
modified = os.path.getctime(path)
modified = time.ctime(modified)
st.markdown(f'*Data was last updated {modified}*')
if st.button('Update Data'):
    main = update_data(stocks)
