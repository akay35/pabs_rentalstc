import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# API Anahtarları
WEATHER_API_KEY = "9575db1d95635dbdb1892012e72aa716"
HOLIDAY_API_KEY = "zvpar9p0MNt7KR7cLsyjAXIOJhZtp7e9"

# LightGBM Modeli Yükle
final_model = joblib.load('bike_rentals_model.pkl')

# Sayaç Fonksiyonu
def update_visitor_count(file_path='visitor_count.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame({'count': [0]})
    df['count'] += 1
    df.to_csv(file_path, index=False)
    return df['count'].iloc[0]

# Sayaç güncelleniyor
visitor_count = update_visitor_count()

# Streamlit Arayüzü
st.title("Ziyaretçi Sayacı")
st.write(f"Bu sayfa {visitor_count} kez ziyaret edildi.")
##########################################################################################
########## Sayfa Düzeni
st.set_page_config(layout="wide", page_title="pabsrentalsaky", page_icon="🚲")
text_col, image_col = st.columns([0.6, 0.4])   #sayfayı ikiye böldük ;)

##########################################################################################
########## Müzik

# # SoundCloud müzik linki
# soundcloud_iframe = """
# <iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/226534207&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/edgar-poe-665232219" title="edgar poe, wsea" target="_blank" style="color: #cccccc; text-decoration: none;">edgar poe, wsea</a> · <a href="https://soundcloud.com/edgar-poe-665232219/its-a-beautiful-day-white-bird" title="It&#x27;s A Beautiful Day - White Bird" target="_blank" style="color: #cccccc; text-decoration: none;">It&#x27;s A Beautiful Day - White Bird</a></div>
# """

# st.title("🎵 Sürüş için Rahatlatıcı Müzik")
# st.markdown("Bisiklet sürüşünüze eşlik edecek huzurlu bir gitar melodisi.")

# # SoundCloud embed müzik
# st.components.v1.html(soundcloud_iframe, height=200)

# SoundCloud embed kodu
# soundcloud_iframe = """<iframe width="100%" height="200" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/226534207&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/edgar-poe-665232219" title="edgar poe, wsea" target="_blank" style="color: #cccccc; text-decoration: none;">edgar poe, wsea</a> · <a href="https://soundcloud.com/edgar-poe-665232219/its-a-beautiful-day-white-bird" title="It&#x27;s A Beautiful Day - White Bird" target="_blank" style="color: #cccccc; text-decoration: none;">It&#x27;s A Beautiful Day - White Bird</a></div>"""


def add_music_player():
    # SoundCloud iframe kodu
    soundcloud_iframe = """<iframe width="100%" height="200" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/226534207&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/edgar-poe-665232219" title="edgar poe, wsea" target="_blank" style="color: #cccccc; text-decoration: none;">edgar poe, wsea</a> · <a href="https://soundcloud.com/edgar-poe-665232219/its-a-beautiful-day-white-bird" title="It&#x27;s A Beautiful Day - White Bird" target="_blank" style="color: #cccccc; text-decoration: none;">It&#x27;s A Beautiful Day - White Bird</a></div>"""

    # HTML ve CSS için bir stil bloğu
    html_code = """
    <div style="position: fixed; width: 500px; height: 300px; margin-top: 20px">
        {iframe}
    </div>
    """.format(iframe=soundcloud_iframe)

    # Streamlit uygulamasında HTML'yi ekleyin
    st.components.v1.html(html_code, height=300)

with image_col:
    # Başlık
    image_col.markdown("<h1 style='font-size: 28px;'>🎵 Every Ride Tells a Story!</h1>", unsafe_allow_html=True)
    
    # Müzik çalar fonksiyonunu çağırıyoruz
    add_music_player() 
    
    # Model hakkında bilgi metni
    image_col.markdown("<p style='font-size: 18px; color: #8B4513; font-style: italic;'>Model tahmininde kullanılan veri seti 2011 ve 2012 yılları arasında Capital bisiklet paylaşım sistemindeki saatlik kiralık bisiklet sayısı, hava durumu, mevsimsel bilgiler ve haftanın günlerine göre çalışma ve tatil durumu bilgilerini içerir. Bu veri seti 17379 satır ve 17 sütun özellik içermektedir. Doğrulama, overfitting, hata analizi ve genellenebilirlik kontrolleri sonrası modelin başarı oranı %96 olarak ölçülmüştür.</p>", unsafe_allow_html=True)
    
    # Görsel
    image_col.image("cyclist-pedaling-through-tunnel-autumn-trees_693425-33036.jpg", width=600)

    # Yine ek metin
    image_col.markdown("<p style='font-size: 18px; color: #808080; font-style: italic;'>As you pedal through life, each ride becomes part of your story. 🎶</p>", unsafe_allow_html=True)
    image_col.markdown("<p style='font-size: 15px; color: #808080; font-style: italic;'>Akay AYDIN</p>", unsafe_allow_html=True)
    image_col.markdown("<p style='font-size: 15px; color: #808080; font-style: italic;'>Berna ERSARI</p>", unsafe_allow_html=True)
    image_col.markdown("<p style='font-size: 15px; color: #808080; font-style: italic;'>Pelin SAYAR</p>", unsafe_allow_html=True)
    image_col.markdown("<p style='font-size: 15px; color: #808080; font-style: italic;'>Semra Konancı KELEŞ</p>", unsafe_allow_html=True)

##########################################################################################
########## Sidebar
st.logo("pngwing.com.png", icon_image="pngwing.com.png", size="large")

# st.sidebar.image("Nature-autumn-background-with-red-trees-and-bike-vector-02.jpg", use_column_width=True)
# st.sidebar.write("Amaç:")
# st.sidebar.write("Bu modelin amacı, çeşitli faktörler (saat, ay, mevsim, sıcaklık, rüzgar, nem, hava durumu vb.) göz önünde bulundurularak bisiklet kiralama sayılarını tahmin eden bir model geliştirmektir.")

# st.sidebar.write("Trafik Yönetimi ve Planlama: Bisiklet kiralama talebinin tahmin edilmesi, bisiklet paylaşım sistemleri için daha verimli bisiklet dağıtımı ve park alanı planlaması sağlar. Bu, şehirlerdeki ulaşım ağlarının daha verimli yönetilmesine katkı sağlar.")

# st.sidebar.write("İklim ve Mevsimsel Etkilerin Anlaşılması: Sıcaklık, yağmur, rüzgar gibi hava durumu koşullarının bisiklet kiralama talebine etkilerini anlamak, bu faktörlerin etkilerini önceden tahmin edebilmek için veri odaklı bir yaklaşım sunar.")

# st.sidebar.write("Kullanıcı Deneyiminin İyileştirilmesi: Kullanıcılar için daha doğru bisiklet erişimi sağlamak, yoğun talep zamanlarına göre bisiklet tedarikini optimize etmek ve dolayısıyla müşteri memnuniyetini artırmak mümkündür.")

# st.sidebar.write("Pazar Stratejilerinin Geliştirilmesi: Bisiklet kiralama hizmeti sağlayıcıları, mevsimsel ve hava koşullarına dayalı tahminlerle kiralama fiyatlarını ve promosyonlarını daha iyi belirleyebilir, kâr marjlarını optimize edebilir.")

# st.sidebar.write("Sürdürülebilir Ulaşımın Teşviki: Bisiklet kullanımı, çevre dostu ulaşım seçenekleri sunarak karbon ayak izinin azaltılmasına yardımcı olabilir. Bu model, şehir içi ulaşımda bisiklet kullanımını teşvik etmek amacıyla stratejik kararlar almada yardımcı olacaktır.")

# Sidebar resmini ekleyin
st.sidebar.image("Summer-background-with-green-tree-and-bike-vector.jpg", use_column_width=True)
st.sidebar.markdown(
    """
    <style>
    .sidebar-content {
        background-color: #f0f4f1; /* Gri beyaz bir doğa temalı arka plan rengi */
        padding: 20px;
        border-radius: 10px;
        font-family: 'Arial', sans-serif;
    }
    .sidebar-content h2 {
        color: #3b7631; /* Doğal bir yeşil */
        font-size: 22px;
        margin-bottom: 10px;
    }
    .sidebar-content p {
        font-size: 14px;
        color: #555555;
        line-height: 1.6;
    }
    .sidebar-content .emoji {
        font-size: 18px;
        padding-right: 5px;
    }
    </style>
    <div class="sidebar-content">
        <h2>🚲 Amaç:</h2>
        <p>
            Bu modelin amacı, çeşitli faktörler (saat, ay, mevsim, sıcaklık, rüzgar, nem, hava durumu vb.) göz önünde bulundurularak bisiklet kiralama sayılarını tahmin eden bir model geliştirmektir.
        </p>
        <p>🌳 <strong>Trafik Yönetimi ve Planlama:</strong> Bisiklet kiralama talebinin tahmin edilmesi, bisiklet paylaşım sistemleri için daha verimli bisiklet dağıtımı ve park alanı planlaması sağlar. Bu, şehirlerdeki ulaşım ağlarının daha verimli yönetilmesine katkı sağlar.</p>
        <p>☀️ <strong>İklim ve Mevsimsel Etkilerin Anlaşılması:</strong> Sıcaklık, yağmur, rüzgar gibi hava durumu koşullarının bisiklet kiralama talebine etkilerini anlamak, bu faktörlerin etkilerini önceden tahmin edebilmek için veri odaklı bir yaklaşım sunar.</p>
        <p>😊 <strong>Kullanıcı Deneyiminin İyileştirilmesi:</strong> Kullanıcılar için daha doğru bisiklet erişimi sağlamak, yoğun talep zamanlarına göre bisiklet tedarikini optimize etmek ve dolayısıyla müşteri memnuniyetini artırmak mümkündür.</p>
        <p>📈 <strong>Pazar Stratejilerinin Geliştirilmesi:</strong> Bisiklet kiralama hizmeti sağlayıcıları, mevsimsel ve hava koşullarına dayalı tahminlerle kiralama fiyatlarını ve promosyonlarını daha iyi belirleyebilir, kâr marjlarını optimize edebilir.</p>
        <p>🌍 <strong>Sürdürülebilir Ulaşımın Teşviki:</strong> Bisiklet kullanımı, çevre dostu ulaşım seçenekleri sunarak karbon ayak izinin azaltılmasına yardımcı olabilir. Bu model, şehir içi ulaşımda bisiklet kullanımını teşvik etmek amacıyla stratejik kararlar almada yardımcı olacaktır.</p>
    </div>
    """, unsafe_allow_html=True
)
st.sidebar.image("Nature-summer-background-with-green-grass-and-bike-vector-02.jpg", use_column_width=True)
##########################################################################################

st.markdown("""
    <style>
    /* Ana sayfanın arka plan rengini değiştir */
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #b2ebf2); /* Açık mavi ve buz beyazı */
        color: #ffffff; /* Yazılar için beyaz */
    }

    /* Başlıklar için stil */
    h1, h2, h3, h4, h5, h6 {
        color: #4caf50; /* Doğal yeşil */
    }

    /* Butonların stilini özelleştirme */
    .stButton>button {
        background-color: #81c784;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        background-color: #66bb6a;
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
city = text_col.text_input("Şehir adını giriniz", "Izmir")

# Dinamik başlık ve alt başlık
text_col.markdown(f"""
    <div class="title">
        🌳 Bisiklet Kiralama Tahmin Uygulaması 🌿
    </div>
    <div class="subtitle">
        Şehir: {city}
    </div>
""", unsafe_allow_html=True)

##########################################################################################
def plot_predictions_by_hour(result):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=result, x='Saat', y='Tahmin', marker='o', color='b', label='Tahmin Edilen Kiralama Sayısı')
    
    plt.title('Saatlere Göre Tahmin Edilen Bisiklet Kiralama Sayıları')
    plt.xlabel('Saat')
    plt.ylabel('Tahmin Edilen Kiralama Sayısı')
    plt.grid(True)
    
    # Grafiği Streamlit'te göstermek
    text_col.pyplot(plt)

##########################################################################################

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
if text_col.button("🚴‍♂️ Tahmin Yap"):
    result = make_predictions(city)
    if result is not None:
        # 'weekday' sütununu kısaltmalarla değiştiriyoruz ve yeni bir 'weekdays' sütunu oluşturuyoruz
        result['Gün'] = result['weekday'].map(weekday_str)
        result['Mevsim'] = result["season"].map(season_str)
        result['Tahmin'] = np.round(result['predicted_rentals']).astype(int)
        result['Çalışma'] = result['workingday'].map({1: 'Evet', 0: 'Hayır'})
        result['Tatil'] = result['holiday'].map({1: 'Evet', 0: 'Hayır'})

        # Tabloyu uygun şekilde Streamlit ile yazdırıyoruz
        result_display = result[["Tarih saat", "Saat", "Mevsim", "Gün", "Çalışma", "Tatil", 'Sıcaklık', 'Nem', 'Rüzgar', "Hava", 'Tahmin']]
        
        # Text column içinde tabloyu doğru şekilde göstermek
        # text_col.write("🔮 **Tahmin Edilen Bisiklet Kiralama Sayıları**")
        text_col.markdown("<h3 style='color: #D1B19D;'>🔮 Tahmin Edilen Bisiklet Kiralama Sayıları</h3>", unsafe_allow_html=True)
        text_col.markdown("<p style='font-size: 15px; color: #808080; font-style: italic;'>Aşağıdaki tablodaki özellikler ve değerler, butona basıldığında güncellenir ve bu verilere dayanarak bisiklet kiralama tahmin sayısı hesaplanır.</p>", unsafe_allow_html=True)
        text_col.dataframe(result_display)  # Burada tabloyu Streamlit ile görselleştiriyoruz
        
plot_predictions_by_hour(result)
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
