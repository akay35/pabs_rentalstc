import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import streamlit as st

# API Anahtarları
WEATHER_API_KEY = "9575db1d95635dbdb1892012e72aa716"
HOLIDAY_API_KEY = "zvpar9p0MNt7KR7cLsyjAXIOJhZtp7e9"

# LightGBM Modeli Yükle
final_model = joblib.load('bike_rentals_model.pkl')

# Streamlit Arayüzü
st.title("Bisiklet Kiralama Tahmin Uygulaması")
city = st.text_input("Şehir Adı", "Istanbul")

def get_season(month):
    if month in [12, 1, 2]:
        return 1  # Kış
    elif month in [3, 4, 5]:
        return 2  # İlkbahar
    elif month in [6, 7, 8]:
        return 3  # Yaz
    else:
        return 4  # Sonbahar

def get_weathersit1(weather_condition):
    if weather_condition in ['clear sky', 'few clouds', 'partly cloudy', 'mostly clear', 'light clouds']:
        return 1
    elif weather_condition in ['mist', 'overcast clouds', 'broken clouds', 'cloudy', 'fog', 'haze']:
        return 2
    elif weather_condition in ['light rain', 'scattered clouds', 'light snow', 'moderate rain']:
        return 3
    else:
        return 4

def sin_cos_encoding(df, columns):
    for col in columns:
        max_val = df[col].max()
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def preprocess_live_data(live_data):


    sincos = ["hr", "mnth", "weekday"]
    live_data = sin_cos_encoding(live_data, sincos)
    # live_data.drop(["hr", "mnth", "weekday"], axis=1, inplace=True)

    scaler = MinMaxScaler()
    y_num_cols = ["temp", "hum", "windspeed", "hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos"]
    live_data[y_num_cols] = scaler.fit_transform(live_data[y_num_cols])

    new_column_order = [
        "temp", "hum", "windspeed", "hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos",
        "season", "holiday", "workingday", "weathersit"
    ]
    return live_data[new_column_order]

def get_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None

    data = response.json()
    forecast_list = data['list']
    weather_data = []

    # Weekday mapping (Burada doğru girintilemeyi sağlıyoruz)
    weekday_mapping = {
        0: 2,  # Pazartesi -> 2
        1: 3,  # Salı -> 3
        2: 4,  # Çarşamba -> 4
        3: 5,  # Perşembe -> 5
        4: 6,  # Cuma -> 6
        5: 7,  # Cumartesi -> 7
        6: 1   # Pazar -> 1
    }

    for entry in forecast_list:
        forecast_datetime = datetime.strptime(entry['dt_txt'], '%Y-%m-%d %H:%M:%S')
        hour = forecast_datetime.hour
        weekday = forecast_datetime.weekday()
        month = forecast_datetime.month
        date_str = forecast_datetime.strftime('%d/%m/%Y')  # Gün/Ay/Yıl formatında tarih

        weather_condition = entry['weather'][0]['description']
        weathersit = get_weathersit1(weather_condition)
        sic = entry['main']['temp']          #gerçek sıcaklık
        temp = (sic - (-8)) / (39 - (-8))    #modele uygun ismi ile normalize edilmiş sıcaklık, minmax sonrası modelimize sokulacak
        humidity = entry['main']['humidity'] #gerçek nem
        wind_speed = entry['wind']['speed']
        windspeed_scaled = wind_speed / 67

        # Burada datetime'ı sadece gösterim için ekliyoruz
        datetime_str = f"{date_str} {hour}:00"  # Tarih ve saat bilgisini birleştiriyoruz

        weather_data.append({
            "Tarih saat": datetime_str,
            "temp": temp,                    #model sıcaklığı
            "Sıcaklık": sic,
            "hum": humidity,                 #model nemi
            "Nem": humidity.apply(lambda x: f"{x}%"),
            "windspeed": windspeed_scaled,
            "hr": hour,
            "mnth": month,
            "weekday": weekday_mapping[weekday],
            "season": get_season(month),
            "holiday": 0,
            "workingday": 1 if humidity >= 50 else 0,
            "weathersit": weathersit
        })
    return pd.DataFrame(weather_data)

def make_predictions(city):
    live_data = get_weather_data(city)
    if live_data is None:
        st.error("Veri alınamadı!")
        return None

    processed_data = preprocess_live_data(live_data)
    predictions = final_model.predict(processed_data)
    live_data['predicted_rentals'] = predictions
    return live_data
####################################################################################################
########## YAZDIRMA AŞAMASI

weekday_str = {
    1: "Paz",  # Pazar
    2: "Pzt",  # Pazartesi
    3: "Sal",  # Salı
    4: "Çar",  # Çarşamba
    5: "Per",  # Perşembe
    6: "Cum",  # Cuma
    7: "Cmt"   # Cumartesi
}

season_str = {
1: "Kış",
2: "İlkbahar",
3: "Yaz",
4: "Sonbahar"
}

# Streamlit butonu ve sonuç gösterimi
if st.button("Tahmin Yap"):
    result = make_predictions(city)
    if result is not None:
        # 'weekday' sütununu kısaltmalarla değiştiriyoruz ve yeni bir 'weekdays' sütunu oluşturuyoruz
        result['Gün'] = result['weekday'].map(weekday_str)
        result['Mevsim'] = result["season"].map(season_str)
        # Dataframe'i Streamlit ile yazdırıyoruz
        st.dataframe(result[['Tarih saat', "Mevsim", "Gün", 'Sıcaklık', 'Nem', 'windspeed', 'predicted_rentals']].rename(columns={'predicted_rentals': 'Tahmini Kiralama Sayısı'}))
