import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("hour.csv")

def preprocess_bike_data(df):
    df.drop(["instant", "dteday", "yr", "atemp", "casual", "registered"], axis=1, inplace=True)

    sincos = ["hr", "mnth", "weekday"]
    df["weekday"] += 1

    for col in sincos:
        max_val = df[col].max()
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    df.drop(["hr", "mnth", "weekday"], axis=1, inplace=True)

    scaler = MinMaxScaler()
    cols = ["temp", "hum", "windspeed", "hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos"]
    df[cols] = scaler.fit_transform(df[cols])

    return df

df = preprocess_bike_data(df)
y = df['cnt']
X = df.drop(['cnt'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

model = LGBMRegressor(n_estimators=500, learning_rate=0.1, colsample_bytree=0.7, random_state=17)
model.fit(X_train, y_train)

joblib.dump(model, 'bike_rentals_model.pkl')
print("Model başarıyla kaydedildi!")
