# Electrical Load Forecasting using Machine Learning
## 1. Introduction and Data Preprocessing

The objective of this assignment was to predict the electrical load (Power Factor) using historical data. The first critical step was restructuring the raw data: the original matrix format (days as rows, hours as columns) was flattened into a continuous hourly time-series.

Since electrical load is highly dependent on human behavior and weather, exogenous variables were integrated. Temperature data was retrieved via Open-Meteo (noting that real-world deployment would rely on the accuracy of weather forecasts). Furthermore, the calendar data was heavily engineered:

    A weekend flag and day_of_week were added.

    Cyclic Encoding: Time variables (hours and months) were transformed using sine and cosine functions. This mathematical representation is crucial to teach the network the continuous nature of time (e.g., ensuring the model understands that 23:00 and 00:00 are consecutive, not opposite ends of a scale).

## 2. Model Selection: Classical Methods vs. Machine Learning

Initially, traditional statistical models like SARIMA were considered. However, SARIMA models present significant limitations when dealing with multiple seasonalities (daily and weekly cycles) over large datasets (3 years of hourly data) due to immense computational costs. Furthermore, SARIMA is inherently linear. Electrical load has a non-linear "U-shaped" relationship with temperature (load peaks during extreme cold for heating and extreme heat for cooling). Consequently, the project pivoted to Machine Learning models, which can autonomously learn non-linear relationships.
## 3. Baseline Model: Random Forest

To establish a baseline, a Random Forest Regressor was trained using only weather and calendar features. Random Forest is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

Validation Strategy: The model was trained on the first two years (2000-2001) and validated on unseen data from the entire third year (2002) to test long-term generalization.

    Results: MAE = 0.0266 | R² = 0.8783
    (Note: R² or Coefficient of Determination indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. A score of 0.87 is a strong baseline).

While the Random Forest performed well, it acts as a "stateless" model: it predicts the load based purely on the current temperature and time, completely ignoring the state of the power grid in the previous hours.
## 4. Advanced Model: Multi-Layer Perceptron (MLP) and Autocorrelation

Electrical grids exhibit strong Autocorrelation (inertia), meaning the current load is heavily dependent on the load of the previous hour, the previous day, and the previous week. To capture this, "Lag features" (lag_1h, lag_24h, lag_168h) were introduced.

To process these dynamic features, the architecture was upgraded to a Multi-Layer Perceptron (MLP), a feed-forward Artificial Neural Network. Unlike Random Forest, an MLP routes data through hidden layers of artificial neurons using non-linear activation functions (ReLU), allowing it to map highly complex interactions.
Important: Before feeding data to the MLP, a StandardScaler was applied. Neural networks are highly sensitive to unscaled data (e.g., mixing temperatures around 30 with sine values between -1 and 1 can distort the network's weights).

    Results: MAE = 0.0070 | R² = 0.9922

## 5. Feature Importance and Model Explainability

Unlike Random Forest, Neural Networks are "black boxes" and do not have a built-in feature importance attribute. To explain the model's decisions, Permutation Importance was utilized. This technique works by randomly shuffling a single feature's column and measuring the subsequent drop in the model's R² score. If the score collapses, the feature is critical.

The Permutation Importance results on the 2002 validation set showed that pf_lag_1h overwhelmingly dominated the predictions (Importance: 1.635). This proves that the network correctly identified grid inertia as the primary driver. Interestingly, the raw temperature feature dropped in importance because its effect is "absorbed" by the lag features (if it was hot an hour ago, the previous load already reflects the cooling demand).
6. Visual Analysis: The Holiday Effect

To visually compare the models, predictions for the last 14 days of 2002 were plotted. This specific timeframe was chosen to observe the Holiday Effect (Christmas and New Year).

During the holidays, standard calendar patterns break down (e.g., a Wednesday acts like a Sunday). The Random Forest, lacking memory, severely overestimated the load on these days because it blindly followed the "weekday" logic. The MLP, relying heavily on its autoregressive lag features, detected the drop in consumption in real-time and dynamically adjusted its forecast, maintaining extreme accuracy despite the calendar anomalies.
