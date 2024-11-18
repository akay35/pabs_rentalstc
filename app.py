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

def preprocess_live_data(live_data, holiday_data):
    live_data['Tarih saat'] = pd.to_datetime(live_data['Tarih saat'], format='%d/%m/%Y %H:%M')
    live_data['Tarih'] = live_data['Tarih saat'].dt.strftime('%Y-%m-%d')
    
    # Resmi tatil bilgilerini holiday_data listesi ile güncelliyoruz
    live_data['holiday'] = live_data['Tarih'].apply(lambda x: 1 if x in holiday_data else 0)
    
    # Haftasonları (Cumartesi ve Pazar) için çalışma günü kontrolü
    # live_data['weekday'] = live_data['Tarih saat'].dt.weekday
    # live_data['workingday'] = live_data.apply(
    #     lambda row: 1 if row['holiday'] == 0 and row['weekday'] < 5 else 0, axis=1
    # )

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
        weekday1 = forecast_datetime.weekday()
        month = forecast_datetime.month
        date_str = forecast_datetime.strftime('%d/%m/%Y')  # Gün/Ay/Yıl formatında tarih

        weather_condition = entry['weather'][0]['description']
        weathersit = get_weathersit1(weather_condition)
        sic = entry['main']['temp']          #gerçek sıcaklık
        temp = (sic - (-8)) / (39 - (-8))    #modele uygun ismi ile normalize edilmiş sıcaklık, minmax sonrası modelimize sokulacak
        humidity = entry['main']['humidity'] #gerçek nem
        ruzgar = entry['wind']['speed']
        ruzgarg = ruzgar * 3.6
        windspeed = ruzgar / 67

        # Burada datetime'ı sadece gösterim için ekliyoruz
        datetime_str = f"{date_str} {hour}:00"  # Tarih ve saat bilgisini birleştiriyoruz
        # date_str = forecast_datetime.strftime('%Y-%m-%d')

        weather_data.append({
            # "Tarih_": forecast_datetime.strftime('%Y-%m-%d'),
            "Tarih saat": datetime_str,
            # "Tarih saat": date_str,
            "temp": temp,                    #model sıcaklığı
            "Sıcaklık": sic,
            "hum": humidity,                 #model nemi
            "Nem": humidity,
            "windspeed": windspeed,
            "Rüzgar": ruzgarg,
            "hr": hour,
            "Saat": hour,
            "mnth": month,
            "weekday": weekday_mapping[weekday1],
            "season": get_season(month),
            "weathersit": weathersit
        })
    return pd.DataFrame(weather_data)
###################################################################################

def get_holiday_data():
    today = datetime.now()
    url = f"https://calendarific.com/api/v2/holidays?api_key={HOLIDAY_API_KEY}&country=TR&year={today.year}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"API Hatası: {response.status_code}")
        return []

    data = response.json()
    holidays = [holiday['date']['iso'][:10] for holiday in data['response']['holidays']]  # YYYY-MM-DD formatında tarih
    return holidays
    
###################################################################################
def make_predictions(city):
    live_data = get_weather_data(city)
    if live_data is None:
        st.error("Veri alınamadı!")
        return None

    holiday_data = get_holiday_data()
    
    processed_data = preprocess_live_data(live_data, holiday_data)
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
        result['Kiralama tahmini'] = np.round(result['predicted_rentals']).astype(int)
        result['Çalışma'] = result['workingday'].map({1: 'Evet', 0: 'Hayır'})
        result['Tatil'] = result['holiday'].map({1: 'Evet', 0: 'Hayır'})

        # Dataframe'i Streamlit ile yazdırıyoruz
# st.table(result[["Tarih saat", "Saat", "Mevsim", "Gün", "weekday", "Çalışma", "Tatil", 'Sıcaklık', 'Nem', 'Rüzgar', 'Kiralama tahmini']])

st.dataframe(result[["Tarih saat", "Saat", "Mevsim", "Gün", "weekday", "Çalışma", "Tatil", 'Sıcaklık', 'Nem', 'Rüzgar', 'Kiralama tahmini']], use_container_width=True)        
