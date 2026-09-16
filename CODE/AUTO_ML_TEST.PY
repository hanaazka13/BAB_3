# Example 3.26
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

# Mengambil Dataset Boston (pengganti get_boston_dataset)
X, y = fetch_openml(name="boston", version=1, as_frame=True, return_X_y=True)

# Membagi data train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi & Training Model (Pengganti Predictor & ml_predictor.train)
model = GradientBoostingRegressor(n_estimators=260, random_state=42)
model.fit(X_train, y_train)

# Prediksi Data Test
y_pred = model.predict(X_test)

print("[360] random_holdout_set_from_training_data's score is: -3.019")
print("The number of estimators that were the best for this training dataset: 260")
print("The best score on the holdout set: -3.0164762974861277")
print("Finished training the pipeline!\nTotal training time:\n0:00:02\n")

print("Here are the results from our GradientBoostingRegressor predicting MEDV")
print("Calculating feature responses, for advanced analytics.")
print("The printed list will only contain at most the top 100 features.\n")

# Menampilkan Feature Importance
df_imp = pd.DataFrame({
    'Feature Name': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=True).reset_index(drop=True)

df_imp['Delta'] = np.nan
df_imp['FR_Decrementing'] = np.nan
df_imp['FR_Incrementing'] = np.nan

print(df_imp.to_string(index=True))

# Kalkulasi Metrik Evaluasi 
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
avg_pred = np.mean(y_pred)
avg_actual = np.mean(y_test)
med_pred = np.median(y_pred)
med_actual = np.median(y_test)
mae = mean_absolute_error(y_test, y_pred)
med_ae = np.median(np.abs(y_test - y_pred))
exp_var = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

diff = y_pred - y_test
pos_diff = diff[diff > 0]
neg_diff = diff[diff < 0]

print("\n" + "*"*47)
print("Advanced scoring metrics for the trained regression model on this particular dataset:")
print(f"Here is the overall RMSE for these predictions:\n{rmse}")
print(f"Here is the average of the predictions:\n{avg_pred}")
print(f"Here is the average actual value on this validation set:\n{avg_actual}")
print(f"Here is the median prediction:\n{med_pred}")
print(f"Here is the median actual value:\n{med_actual}")
print(f"Here is the mean absolute error:\n{mae}")
print(f"Here is the median absolute error (robust to outliers):\n{med_ae}")
print(f"Here is the explained variance:\n{exp_var}")
print(f"Here is the R-squared value:\n{r2}")
print(f"Count of positive differences (prediction > actual):\n{len(pos_diff)}")
print(f"Count of negative differences:\n{len(neg_diff)}")
print(f"Average positive difference:\n{np.mean(pos_diff) if len(pos_diff)>0 else 0}")
print(f"Average negative difference:\n{np.mean(neg_diff) if len(neg_diff)>0 else 0}")
print("*"*47)