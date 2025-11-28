import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import requests
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import concurrent.futures
from PIL import Image # Logo işleme için gerekli

# --- 1. LOGO YÜKLEME VE SAYFA AYARLARI ---
# Logoyu önce yüklemeye çalışıyoruz, yoksa standart ikon kullanıyoruz
try:
    logo_img = Image.open("logo.png")
    page_icon_img = logo_img
except:
    page_icon_img = "🛡️"

st.set_page_config(
    page_title="MERTT AI", 
    layout="wide", 
    page_icon=page_icon_img # Sekmedeki küçük ikon artık senin logon!
)

# --- PWA MODU (MOBİL GÖRÜNÜM) ---
def pwa_kodlari():
    pwa_html = """
    <meta name="theme-color" content="#0e1117">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="MERTT AI">
    """
    components.html(f"<html><head>{pwa_html}</head></html>", height=0, width=0)
pwa_kodlari()

# --- GÜVENLİK DUVARI ---
def guvenlik_kontrolu():
    if 'giris_yapildi' not in st.session_state: st.session_state['giris_yapildi'] = False
    
    if not st.session_state['giris_yapildi']:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            # Giriş Ekranında Büyük Logo
            try: st.image("logo.png", use_container_width=True)
            except: st.header("MERTT AI")
            
            st.markdown("<h4 style='text-align: center;'>Gelecek İçin Bilgi ve Teknoloji</h4>", unsafe_allow_html=True)
            st.divider()
            
            sifre = st.text_input("Erişim Anahtarı:", type="password")
            if st.button("Sisteme Giriş Yap", type="primary", use_container_width=True):
                try:
                    if sifre == st.secrets["GIRIS_SIFRESI"]: 
                        st.session_state['giris_yapildi'] = True
                        st.rerun()
                    else: st.error("⛔ Yetkisiz Erişim!")
                except: st.error("Ayar Hatası: Şifre tanımlanmamış.")
        return False
    return True

if not guvenlik_kontrolu(): st.stop()

# --- ANALİZ MOTORU ---
class TradingEngine:
    def __init__(self):
        # Scikit-learn hatasını önlemek için import kontrolü
        try:
            from sklearn.preprocessing import StandardScaler
        except:
            pass
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    
    def get_live_price(self, ticker):
        try:
            url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker.replace('.IS','')}-detay/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=2)
            soup = BeautifulSoup(resp.content, "html.parser")
            price = soup.find("span", {"class": "text-2"}).text.strip().replace(',', '.')
            return float(price)
        except: return None

    def get_data(self, ticker):
        if not ticker.endswith('.IS'): ticker += '.IS'
        try:
            df = yf.download(ticker, period="5d", interval="15m", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            return df
        except: return None

    def analyze(self, ticker):
        df = self.get_data(ticker)
        if df is None or len(df) < 30: return None
        
        # Canlı fiyat güncelleme
        live_price = self.get_live_price(ticker)
        if live_price: df.iloc[-1, df.columns.get_loc('Close')] = live_price
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        clean_df = df.dropna()
        features = ['RSI', 'VWAP', 'ATR']
        
        self.model.fit(clean_df.iloc[:-1][features], clean_df.iloc[:-1]['Target'])
        prob = self.model.predict_proba(clean_df.iloc[[-1]][features])[0][1] * 100
        
        last = df.iloc[-1]
        
        # Karar Mekanizması
        signal = "NÖTR / İZLE"
        color = "gray"
        stop_loss = last['Close'] - (last['ATR'] * 1.5)
        target_price = last['Close'] + (last['ATR'] * 3.0)

        if prob > 60 and last['Close'] > last['VWAP']:
            signal = "GÜÇLÜ AL 🚀"
            color = "green"
        elif prob < 40 and last['Close'] < last['VWAP']:
            signal = "SAT / DÜŞÜŞ 🔻"
            color = "red"
            
        return {
            "Hisse": ticker.replace('.IS',''), 
            "Fiyat": last['Close'], 
            "Skor": prob, 
            "RSI": last['RSI'], 
            "Sinyal": signal,
            "Renk": color,
            "Stop": stop_loss,
            "Hedef": target_price,
            "Data": df
        }

# --- ARAYÜZ ---
def main():
    # Yan Menü (Sidebar) Tasarımı
    with st.sidebar:
        try:
            # Yan menüde logo gösterimi
            st.image("logo.png", use_container_width=True)
        except:
            st.header("MERTT")
            
        st.markdown("<h3 style='text-align: center;'>Yapay Zeka Üssü</h3>", unsafe_allow_html=True)
        st.divider()
        menu = st.radio("Kontrol Paneli", ["💬 Hisse Sor / Analiz", "📡 Piyasa Radarı", "Çıkış"])
        
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    engine = TradingEngine()

    # --- 1. MODÜL: HİSSE SORMA ---
    if menu == "💬 Hisse Sor / Analiz":
        st.title("💬 Hisse Analiz Asistanı")
        st.markdown("Yapay zekaya analiz ettirmek istediğin hisseyi yaz.")
        
        c1, c2 = st.columns([3,1])
        with c1:
            symbol = st.text_input("Hisse Kodu (Örn: THYAO):", "").upper()
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("Analiz Et 🔍", type="primary")

        if btn and symbol:
            with st.spinner(f"{symbol} taranıyor..."):
                res = engine.analyze(symbol)
                if res:
                    # Özet Kartları
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Fiyat", f"{res['Fiyat']:.2f} TL")
                    m2.metric("AI Güveni", f"%{res['Skor']:.1f}")
                    m3.metric("RSI", f"{res['RSI']:.0f}")
                    
                    st.divider()
                    
                    # Sinyal Kutusu
                    if res['Renk'] == 'green':
                        st.success(f"### 📢 KARAR: {res['Sinyal']}")
                        c1, c2 = st.columns(2)
                        c1.info(f"🛡️ **Stop-Loss:** {res['Stop']:.2f} TL")
                        c2.success(f"🎯 **Hedef:** {res['Hedef']:.2f} TL")
                    elif res['Renk'] == 'red':
                        st.error(f"### 📢 KARAR: {res['Sinyal']}")
                        st.warning("Düşüş trendi hakim. Alım önerilmez.")
                    else:
                        st.warning(f"### 📢 KARAR: {res['Sinyal']}")
                        st.info("Yön belirsiz. Beklemede kalmak en iyisi.")
                        
                    # Grafik
                    st.subheader("📊 Grafik Analizi")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=res['Data'].index, 
                                               open=res['Data']['Open'], high=res['Data']['High'],
                                               low=res['Data']['Low'], close=res['Data']['Close'], name="Fiyat"))
                    fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data']['VWAP'], line=dict(color='orange'), name="VWAP"))
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Hisse bulunamadı veya verisi yetersiz.")

    # --- 2. MODÜL: OTOMATİK TARAMA (RADAR) ---
    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        st.info("Bu ekran, seçili hisseleri anlık tarayıp fırsat olanları listeler. (Ayrıca Telegram botu arka planda otomatik çalışmaya devam eder).")
        
        if st.button("TARAMAYI BAŞLAT 🚀"):
            hisseler = ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS", "HEKTS", "PETKM"]
            results = []
            bar = st.progress(0)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(engine.analyze, t): t for t in hisseler}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    r = future.result()
                    if r: results.append(r)
                    completed += 1
                    bar.progress(completed/len(hisseler))
            
            bar.empty()
            
            if results:
                # Sadece fırsat olanları veya nötr olanları gösterelim
                df = pd.DataFrame(results)
                # DataFrame'i güzelleştirme
                st.dataframe(
                    df[['Hisse', 'Fiyat', 'Sinyal', 'Skor', 'RSI']]
                    .style.background_gradient(subset=['Skor'], cmap='Greens'),
                    use_container_width=True
                )
            else:
                st.warning("Şu an kriterlere uyan fırsat bulunamadı.")

if __name__ == "__main__":
    main()
    
