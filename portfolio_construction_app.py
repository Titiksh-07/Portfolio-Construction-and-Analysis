import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import sys
sys.path.insert(0, r'C:\Users\user\Documents\GitHub\Portfolio-Construction-and-Analysis\python_files')
import functions_1 as fnc
import portfolio_construction_toolkit.all as pc
from datetime import datetime, timedelta

st.title('Portfolio Construction and Analysis Dashboard')

selected_assets = st.text_input('Assets (separated by ", "). Press "enter" to run.')

show_filters = st.button('Show Filters')

if show_filters:
    # Define filter options
    interval_options = ['1d', '1wk', '1mo']
    dividends_options = [True, False]
    time_period_options = ['Maximum', 'Common Maximum', 'Random Time Period']

    # Create filter widgets
    select_data_frequency = st.selectbox('Select Data Frequency:', interval_options)
    selected_dividends = st.selectbox('Include Dividends:', dividends_options)
    selected_time_period = st.selectbox('Select Time Period: ', time_period_options)

    if selected_time_period == 'Random Time Period':
        start_date = st.date_input('Start Date')
        end_date = st.date_input('End Date')
        if start_date > end_date:
            st.error('Error: End date must be after start date.')

    # Filter button
    if st.button('Apply Filters'):
        asset_list = selected_assets.split(', ')
        if asset_list:
            df = fnc.get_returns_data(asset_list, max_period=True, interval=select_data_frequency,
                                       replace_tickers=False, dividends=selected_dividends)
            st.write(df)

    hide_filters = st.button('Hide Filters')
    if hide_filters:
        show_filters = False
else:
    if selected_assets:
        asset_list = selected_assets.split(', ')
        if asset_list:
            df = fnc.get_returns_data(asset_list, max_period=True, interval='1d',
                                   replace_tickers=False, dividends=True)
            st.write(df)

