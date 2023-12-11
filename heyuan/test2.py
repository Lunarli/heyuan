import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Define the array data
data = np.array([0.25421766, 0.24858746, 0.24437478, 0.24297722, 0.24579232, 0.24858746,
                 0.24437478, 0.24157966, 0.23736698, 0.23455188, 0.19382274, 0.22191387,
                 0.17836964, 0.2148861, 0.20927586, 0.21628366, 0.15449118, 0.15449118,
                 0.16431409, 0.17415697, 0.20083056, 0.16854673, 0.17415697, 0.19803542,
                 0.19943298, 0.20083056, 0.19803542, 0.19803542, 0.18819255, 0.18961007,
                 0.18961007, 0.19661789, 0.19943298, 0.20646076, 0.2148861, 0.21067343,
                 0.20927586, 0.22049634, 0.23173678, 0.23594945, 0.23736698, 0.23455188,
                 0.24437478, 0.23736698, 0.24437478, 0.23736698, 0.24718988, 0.24157966,
                 0.23455188, 0.22892168, 0.22752411, 0.22470901, 0.21909877, 0.21348853,
                 0.20927586, 0.21207099, 0.22049634, 0.21909877, 0.2303392, 0.2303392,
                 0.250005, 0.24016211, 0.25140256, 0.26122546, 0.26685568, 0.26543814,
                 0.26685568, 0.26825324, 0.27106833, 0.26543814, 0.26404058, 0.262643,
                 0.25843034, 0.24858746, 0.24858746, 0.25140256, 0.25561522, 0.26122546,
                 0.27246592, 0.27246592, 0.25140256, 0.25421766, 0.2598279, 0.27246592,
                 0.30055703, 0.3103999, 0.30476971, 0.29494679, 0.28652145, 0.29352925,
                 0.30618722, 0.31319505, 0.32864814, 0.3103999, 0.30476971, 0.29213169,
                 0.28089125, 0.28089125, 0.27246592, 0.2696508, 0.27528101, 0.27388343,
                 0.27667858, 0.27246592, 0.2696508, 0.26685568, 0.26685568, 0.27388343,
                 0.2696508, 0.262643, 0.2598279, 0.26543814, 0.26543814, 0.2696508,
                 0.28510391, 0.29915947, 0.30195459, 0.28652145, 0.27667858, 0.27246592])

# Split the data into training and testing sets
train_data = data[:96]
test_data = data[96:]

# Define the parameter ranges
seasonal_periods = [12, 24]  # Seasonal periods
trend_options = ["add", "mul", None]  # Trend models
seasonal_options = ["add", "mul", None]  # Seasonal models
damped_trend_options = [True, False]  # Damped trend options

# Save the best model parameters and corresponding AIC value
best_aic = float("inf")
best_params = None

# Iterate over parameter combinations to find the best model based on AIC
for seasonal_period in seasonal_periods:
    for trend_option in trend_options:
        for seasonal_option in seasonal_options:
            for damped_trend_option in damped_trend_options:
                try:
                    model = ExponentialSmoothing(train_data,
                                                 seasonal_periods=seasonal_period,
                                                 trend=trend_option,
                                                 seasonal=seasonal_option,
                                                 damped_trend=damped_trend_option)
                    model_fit = model.fit()

                    rmse = np.sqrt(np.mean((model_fit.forecast(24) - test_data) ** 2))
                    AIC = model_fit.aic
                    BIC = model_fit.bic

                    print('rmse ' + str(rmse) ,
                          'AIC ' + str(AIC),
                          'BIC ' + str(BIC) )
                    print('seasonal_periods :' + str(seasonal_period),
                          'trend :' + str(trend_option),
                          'seasonal :' + str(seasonal_option),
                          'damped_trend :' + str(damped_trend_option)
                          )
                    # plt.plot(model_fit.forecast(24), label='predict')
                    # plt.plot(test_data, label='real')
                    # plt.legend()
                    # plt.show()
                    #
                    # plt.plot(model_fit.fittedvalues, label='predicted_value')
                    # plt.plot(train_data.ravel(), label='real_value')
                    # plt.title('拟合结果')
                    # plt.legend(loc=1)
                    # plt.show()


                    aic = model_fit.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (seasonal_period, trend_option, seasonal_option, damped_trend_option)
                except:
                    continue

# Fit the best model and forecast the next 24 points
model = ExponentialSmoothing(data,
                             seasonal_periods=best_params[0],
                             trend=best_params[1],
                             seasonal=best_params[2],
                             damped_trend=best_params[3])
model_fit = model.fit()
forecast = model_fit.forecast(steps=24)



# Plot the predicted values and the true values
plt.plot(np.arange(97, 121), test_data, label='True Values')
plt.plot(np.arange(97, 121), forecast, label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Exponential Smoothing Model - Forecast')
plt.legend()
plt.show()

print()