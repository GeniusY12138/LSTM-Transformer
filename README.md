# Stock Volatility Prediction with Google Search Trends

## Introduction

## Data Collection
Data was collected through pytrends 4.9.2, which is an unofficial API for Google Trends. Google Trends provides daily search data for up to the last 90 days. For any longer periods, the data is only available weekly. Due to Google backend's rate limit, we were able to retrieve at most the weekly search volume over the past 5 years for 5 keywords at a time. The weekly Google Trends of a total of 98 keywords were collected over the past 5 years, which gave 260 time points in total.

The key words list includes:
['economics', 'debt', 'inflation', 'metals', 'sell', 'bonds', 'risk', 'car', 'leverage', 'color',
'chance', 'unemployment', 'nasdaq', 'money', 'society', 'war', 'transaction', 'cash', 'economy',
'stocks', 'forex', 'finance', 'fed', 'growth', 'culture', 'banking', 'markets', 'marriage', 'office',
'stock market', 'revenue', 'house', 'dow jones', 'portfolio', 'fond', 'travel', 'ore', 'fine', 'religion',
'gains', 'restaurant', 'consumption', 'loss', 'credit', 'default', 'crisis', 'hedge', 'headlines', 'cancer',
'rich', 'trader', 'garden', 'housing', 'gain', 'return', 'tourism', 'investment', 'derivatives', 'oil',
'politics', 'invest', 'food', 'crash', 'returns', 'greed', 'movie', 'health', 'nyse', 'rare earths', 'success',
'water', 'short sell', 'consume', 'gold', 'bubble', 'energy', 'lifestyle', 'home', 'freedom', 'world',
'opportunity', 'happy', 'dividend', 'arts', 'present', 'labor', 'environment', 'buy', 'financial markets',
'fun', 'short selling', 'earnings', 'holiday', 'profit', 'kitchen', 'train', 'ring', 'conflict']

## Data Preprocessing

## Scheme Selection

## OLS model

## Ridge Regression

## Lasso Regression

## PLS model

## LSTM model
