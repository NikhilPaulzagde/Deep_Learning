import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

import yfinance as yfin
yfin.pdr_override()


# Set page title and app wide style
st.set_page_config(
    page_title="Stock Trend Prediction",
    page_icon="📈",
    layout="wide"
)

# Title and Header
st.title("Stock Trend Prediction")
st.header("An Interactive Stock Price Prediction App")

# Sidebar - User Inputs
st.sidebar.header("User Inputs")
start = st.sidebar.date_input("Enter Starting Date")
end = st.sidebar.date_input("Enter End Date")
user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL', key='stock_ticker')

# Fetch Stock Data
try:
    df = pdr.get_data_yahoo(user_input, start=start, end=end)
    st.sidebar.subheader(f"Date Range: {start} to {end}")
    
    # Format describe output to three decimal places
    description = df.describe().applymap(lambda x: f"{x:.2f}")
    
    # Display describe output with custom CSS style
    st.sidebar.markdown(f"<style>.sidebar-table {{ width: auto; }}</style>", unsafe_allow_html=True)
    st.sidebar.write(description)
except:
    st.sidebar.error("Invalid stock symbol or date range")






st.subheader("Closing Price vs Time Chart")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df.Close, label='Closing Price')
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.set_title("Closing Price vs Time")
# Custom Styling
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, linestyle='--', alpha=0.7, color='gray')  # Add grid lines with gray color
ax1.set_facecolor('#f9f9f9')
ax1.legend(frameon=False)
# Add Interactivity
st.pyplot(fig1, use_container_width=True)


# Closing Price with Moving Averages and Interactive Features
st.subheader("Closing Price with Moving Averages")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

ax2.plot(df.Close, label='Closing Price')
ax2.plot(ma100, label='MA100', color='red')
ax2.plot(ma200, label='MA200', color='green')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.set_title("Closing Price with Moving Averages")

# Custom Styling
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, linestyle='--', alpha=0.7, color='gray')  # Add grid lines with gray color
ax2.set_facecolor('#f9f9f9')
ax2.legend(frameon=False)

# Add Interactivity
st.pyplot(fig2, use_container_width=True)

## Spliting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)



#Load Model
model=load_model("keras_model.h5")


#Testing part

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)

input_data=scaler.fit_transform(final_df)


X_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    


X_test,y_test=np.array(X_test),np.array(y_test)

y_pred=model.predict(X_test)

scal=scaler.scale_

scale_factor=1/scal
y_pred=y_pred*scale_factor
y_test=y_test*scale_factor

# Prediction Chart with Interactive Features
st.subheader("Predicted vs Original Price")
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(y_test, label='Original Price', color='blue')
ax3.plot(y_pred, label='Predicted Price', color='red')
ax3.set_xlabel("Time")
ax3.set_ylabel("Price")
ax3.set_title("Predicted vs Original Price")
ax3.legend()


# Custom Styling
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(True, linestyle='--', alpha=0.7, color='gray')  # Add grid lines with gray color
ax3.set_facecolor('#f9f9f9')
ax3.legend(frameon=False)
ax3.set_xticks(range(0, len(y_test), 50))


# Add Interactivity
st.pyplot(fig3, use_container_width=True)



# Custom Styling with Grid and Gray Background
def apply_custom_styling(ax, title):
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')  # Add grid lines with gray color
    ax.set_facecolor('#f9f9f9')
    ax.legend(frameon=False)
    ax.set_xticks(range(0, len(y_test), 50))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")


