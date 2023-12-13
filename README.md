# Stock Volatility Prediction with Google Search Trends

## Overview
This repository is a group project for QTM 347: Machine Learning at Emory University, in which we present a prediction framework for stock market volatility, specifically on the Volatility Index (VIX). Our analysis is based on 98 Google Search trends (Preis, 2013), to capture the collective sentiment and interest of market participants. The models employed include Ordinary Least Squares (OLS), Partial Least Squares (PLS), Ridge Regression, Lasso Regression, and Long Short-Term Memory (LSTM) networks. 

## Data Collection
Google Trends was collected through pytrends 4.9.2, which is an unofficial API for Google Trends. Google Trends provides daily search data for up to the last 90 days. For any longer periods, the data is only available weekly. Due to Google backend's rate limit, we were able to retrieve at most the weekly search volume over the past 5 years for 5 keywords at a time. The weekly Google Trends of a total of 98 keywords were collected over the past 5 years.

![keyword_wordcloud](https://github.com/GeniusY12138/LSTM-Transformer/assets/110353222/418d1629-8b18-422e-aa6e-f0e57d0b04ac)

Stock volatility is measured through the Chicago Board Options Exchange's CBOE Volatility Index (VIX), a popular measure of the stock market's expectation of volatility based on S&P 500 index options. We also calculated the volatility based on the following equations (Garman, 1980):

$u = \log\left(\frac{\text{Hit}}{\text{Opt}}\right)$, $d = \log\left(\frac{\text{Lot}}{\text{Opt}}\right)$, $c = \log\left(\frac{\text{Clt}}{\text{Opt}}\right)$ (1)

$\sigma_t = 0.511 (u - d)^2 - 0.019 \left[c(u + d) - 2ud\right] - 0.383c^2$ (2)

We obtained a Spearman's rank correlation of 0.7 from the two measures, suggesting a strong positive monotonic relationship. We used VIX as the measure for volatility in the following analysis

## Data Preprocessing
Google Trends shows relative search frequency with a range between 0 to 100. Min-Max scaling is applied to the features to ensure that all values are within the same scale, typically between 0 and 1. We then create sequences for LSTM training. Each input sequence is a concatenation of feature values and VIX from the past n samples, with the corresponding VIX value from the next sample.

The data is split into training and testing sets using the train_test_split function from scikit-learn. The shuffle=False parameter ensures that the temporal order of the data is maintained, which is crucial for time series data.

![image](https://github.com/GeniusY12138/LSTM-Transformer/assets/110353222/a6e1fe26-62f7-4f03-aa1e-38f1acd0dbe0)


## Scheme Selection

## OLS model

## Ridge Regression

## Lasso Regression

## PLS model

## LSTM model
This section introduces the Long Short-Term Memory (LSTM) model, implemented using TensorFlow and Keras in Python. Long Short-Term Memory (LSTM) is a variant of recurrent neural networks (RNNs), which excels at capturing temporal dependencies in sequential data—making them particularly well-suited for time series analysis.

We used a baseline LSTM model architecture as follows:

```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 64)                41984     
_________________________________________________________________
dense (Dense)                (None, 1)                 65        
=================================================================
Total params: 42049
Trainable params: 42049
Non-trainable params: 0
```
The model is compiled using the Adam optimizer and Mean Squared Error (MSE) as the loss function. Additionally, the Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) are used as the evaluation metrics. The training process includes early stopping to prevent overfitting. The training history is captured to monitor the model's performance over epochs.

![image](https://github.com/GeniusY12138/LSTM-Transformer/assets/110353222/d864b740-d058-4616-9ffb-ae8dabd6d1aa)

![image](https://github.com/GeniusY12138/LSTM-Transformer/assets/110353222/80c13b1a-8315-42e7-bceb-5968f31a6c5f)

**Model Evaluation:**
```
loss: 7.0208 - root_mean_squared_error: 2.6497 - mape: 13.7877
Test MSE: [7.020784378051758, 2.6496763229370117, 13.78770923614502]
```

## PCA + LSTM
To refine the prediction model, we incorporated Principal Component Analysis (PCA) to identify and select the most influential features. After applying PCA, we selected the top 20 principal components (PCs) that exhibit the highest variance, providing a more concise representation of the input data.

![image](https://github.com/GeniusY12138/LSTM-Transformer/assets/110353222/8acccad6-4a81-4380-bd21-7f4b2fe56577)

![image](https://github.com/GeniusY12138/LSTM-Transformer/assets/110353222/f8a36dc1-a4c6-4e77-b7ef-d51131f636fa)

![image](https://github.com/GeniusY12138/LSTM-Transformer/assets/110353222/ab90c158-e773-4745-9129-a72ddb91d0f9)

**Model Evaluation:**
```
loss: 8.8712 - root_mean_squared_error: 2.9785 - mape: 14.1179
Test MSE: [8.871188163757324, 2.9784538745880127, 14.117850303649902]
```

## Discussion

## References
Garman, M. B., & Klass, M. J. (1980). On the Estimation of Security Price Volatilities from Historical Data. The Journal of Business, 53(1), 67–78. http://www.jstor.org/stable/2352358

Preis, T., Helen Susannah Moat, & H. Eugene Stanley. (2013). Quantifying Trading Behavior in Financial Markets Using Google Trends. Scientific Reports, 3(1). https://doi.org/10.1038/srep01684

Xiong, R., Nichols, E., & Shen, Y. (n.d.). Deep Learning Stock Volatility with Google Domestic Trends. https://arxiv.org/pdf/1512.04916.pdf

## Contributors
Max Cao, Zoe Ji, Kristen Li, Bowen You 
