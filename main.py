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

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="MERTT AI", layout="wide", page_icon="🛡️")

# --- PWA MODU ---
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
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            try: st.image("logo.png", use_column_width=True)
            except: pass
            st.markdown("<h3 style='text-align: center;'>Gelecek İçin Bilgi ve Teknoloji</h3>", unsafe_allow_html=True)
            sifre = st.text_input("Erişim Anahtarı:", type="password")
            if st.button("Sisteme Giriş Yap", type="primary", use_container_width=True):
                try:
                    # Şifre Streamlit Secrets'tan çekilecek
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

    def analyze(self, ticker):
        if not ticker.endswith('.IS'): ticker += '.IS'
        try:
            df = yf.download(ticker, period="5d", interval="15m", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            if len(df) < 30: return None
            
            live_price = self.get_live_price(ticker)
            if live_price: df.iloc[-1, df.columns.get_loc('Close')] = live_price
            
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.dropna()
            
            self.model.fit(df.iloc[:-1][['RSI', 'VWAP']], df.iloc[:-1]['Target'])
            prob = self.model.predict_proba(df.iloc[[-1]][['RSI', 'VWAP']])[0][1] * 100
            
            last = df.iloc[-1]
            if prob > 60 and last['Close'] > last['VWAP']:
                return {"Hisse": ticker.replace('.IS',''), "Fiyat": last['Close'], "Skor": prob, "RSI": last['RSI']}
        except: return None

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        try: st.image("logo.png")
        except: pass
        st.markdown("<h3 style='text-align: center;'>MERTT AI</h3>", unsafe_allow_html=True)
        menu = st.radio("Menü", ["Radar", "Hakkında", "Çıkış"])
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    if menu == "Radar":
        st.title("📡 MERTT Piyasa Radarı")
        hisseler = ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS"]
        if st.button("TARAMAYI BAŞLAT 🚀", type="primary"):
            results = []
            bar = st.progress(0)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(TradingEngine().analyze, t): t for t in hisseler}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    r = future.result()
                    if r: results.append(r)
                    completed += 1
                    bar.progress(completed/len(hisseler))
            
            bar.empty()
            if results:
                st.success(f"{len(results)} Fırsat Bulundu!")
                st.dataframe(pd.DataFrame(results).style.background_gradient(subset=['Skor'], cmap='Greens'))
            else: st.info("Fırsat yok.")

if __name__ == "__main__":
    main()
