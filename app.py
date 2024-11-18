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
# st.title("Bisiklet Kiralama Tahmin Uygulaması")
# city = st.text_input("Şehir Adı", "Izmir")


st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #3E8E41;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }
        .subtitle {
            font-size: 20px;
            color: #6A9E3F;
            text-align: center;
            font-style: italic;
            font-family: 'Arial', sans-serif;
        }
        /* Dinamik Arka Plan Animasyonu */
        body {
            animation: changeBackground 20s infinite;
            background-size: cover;
            background-position: center;
            font-family: 'Arial', sans-serif;
            text-align: center;
            padding-top: 20%;
            color: white;
        }

        @keyframes changeBackground {
            0% {
                background-image: url('https://www.w3schools.com/w3images/fjords.jpg'); /* Güneşli hava */
            }
            25% {
                background-image: url('https://www.w3schools.com/w3images/mountains.jpg'); /* Bulutlu hava */
            }
            50% {
                background-image: url('https://www.w3schools.com/w3images/forest.jpg'); /* Yağmurlu hava */
            }
            75% {
                background-image: url('https://www.w3schools.com/w3images/forest.jpg'); /* Sisli hava */
            }
            100% {
                background-image: url('https://www.w3schools.com/w3images/fjords.jpg'); /* Güneşli hava */
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #3E8E41;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .subtitle {
        font-size: 20px;
        color: #6A9E3F;
        text-align: center;
        font-style: italic;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Kullanıcıdan şehir adı al
city = st.text_input("Şehir Adı", "Izmir")

# Dinamik başlık ve alt başlık
st.markdown(f"""
    <div class="title">
        🌳 Bisiklet Kiralama Tahmin Uygulaması 🌿
    </div>
    <div class="subtitle">
        Şehir: {city}
    </div>
""", unsafe_allow_html=True)

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

# def get_weathersit1(weather_condition):
#     if weather_condition in ['clear sky', 'few clouds', 'partly cloudy', 'mostly clear', 'light clouds']:
#         return 1, "Açık Hava"  # clear sky, few clouds vb. -> Açık Hava
#     elif weather_condition in ['mist', 'overcast clouds', 'broken clouds', 'cloudy', 'fog', 'haze']:
#         return 2, "Sisli ve Bulutlu"  # mist, overcast clouds vb. -> Sisli ve Bulutlu
#     elif weather_condition in ['light rain', 'scattered clouds', 'light snow', 'moderate rain']:
#         return 3, "Hafif Yağış"  # light rain, scattered clouds vb. -> Hafif Yağış
#     else:
#         return 4, "Şiddetli Yağış"  # diğer durumlar -> Şiddetli Yağış

def sin_cos_encoding(df, columns):
    for col in columns:
        max_val = df[col].max()
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def preprocess_live_data(live_data, holiday_data):
    # Resmi tatil bilgilerini holiday_data listesi ile güncelliyoruz
    live_data['holiday'] = live_data['Tarih saat'].apply(lambda x: 1 if x[:10] in holiday_data else 0)

    
    # Haftasonları (Cumartesi ve Pazar) için çalışma günü kontrolü
    live_data['weekday'] = live_data['weekday']  # get_weather_data'dan alınan 'weekday' kullanılıyor
    live_data['workingday'] = live_data.apply(
        lambda row: 1 if row['holiday'] == 0 and 2 <= row['weekday'] <= 6 else 0, axis=1
    )

    # Sinüs ve kosinüs dönüşümü için saat, ay, hafta günü gibi kolonları kullanıyoruz
    sincos = ["hr", "mnth", "weekday"]
    live_data = sin_cos_encoding(live_data, sincos)

    # Veriyi normalleştiriyoruz
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
            "Hava": weather_condition,
            "hr": hour,
            "Saat": hour,
            "mnth": month,
            "weekday": weekday_mapping[weekday],
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
    # live_data['Hava'] = live_data['Hava'].apply(lambda x: get_weathersit1(x)[1])
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
2: "İlkbhr",
3: "Yaz",
4: "Snbhr"
}

# Streamlit butonu ve sonuç gösterimi
if st.button("🚴‍♂️ Tahmin Yap"):
    result = make_predictions(city)
    if result is not None:
        # 'weekday' sütununu kısaltmalarla değiştiriyoruz ve yeni bir 'weekdays' sütunu oluşturuyoruz
        result['Gün'] = result['weekday'].map(weekday_str)
        result['Mevsim'] = result["season"].map(season_str)
        result['Tahmin'] = np.round(result['predicted_rentals']).astype(int)
        result['Çalışma'] = result['workingday'].map({1: 'Evet', 0: 'Hayır'})
        result['Tatil'] = result['holiday'].map({1: 'Evet', 0: 'Hayır'})

        # Dataframe'i Streamlit ile yazdırıyoruz
        st.dataframe(result[["Tarih saat", "Saat", "Mevsim", "Gün", "Çalışma", "Tatil", 'Sıcaklık', 'Nem', 'Rüzgar', "Hava", 'Tahmin']], use_container_width=True)
      
# Kullanıcıya hava durumu bilgisi ve animasyon önerisi
# Streamlit ile hava durumu ve saat bilgisi gösterme
if result is not None:
    st.markdown(f"""
        <div style="text-align: center; font-size: 24px;">
            <strong>Hava Durumu:</strong> {result['Hava'].iloc[0]}<br>
            <strong>Güncel Saat:</strong> {result['Saat'].iloc[0]}<br>
        </div>
    """, unsafe_allow_html=True)


# Add background image styling at the end
background_image = "path_to_your_image.jpg"  # Replace with local file path
st.markdown(f"""
    <style>
        body {{
            background-image: url('{background_image}');
            background-size: cover;
            background-position: center;
        }}
    </style>
""", unsafe_allow_html=True)
