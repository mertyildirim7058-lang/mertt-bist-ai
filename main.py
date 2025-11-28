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
        
        # Canlı Fiyatı Ekle
        live_price = self.get_live_price(ticker)
        if live_price: df.iloc[-1, df.columns.get_loc('Close')] = live_price
        
        # İndikatörler
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        clean_df = df.dropna()
        features = ['RSI', 'VWAP', 'ATR']
        
        self.model.fit(clean_df.iloc[:-1][features], clean_df.iloc[:-1]['Target'])
        prob = self.model.predict_proba(clean_df.iloc[[-1]][features])[0][1] * 100
        
        last = df.iloc[-1]
        
        # Sinyal ve Hedefler
        signal = "NÖTR / İZLE"
        color = "gray"
        stop_loss = last['Close'] - (last['ATR'] * 1.5)
        target_price = last['Close'] + (last['ATR'] * 3.0)

        if prob > 60 and last['Close'] > last['VWAP']:
            signal = "GÜÇLÜ AL 🚀"
            color = "green"
        elif prob < 40 and last['Close'] < last['VWAP']:
            signal = "SAT / DÜŞÜŞ BEKLENTİSİ 🔻"
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
            "Data": df # Grafiği çizmek için veriyi de döndürüyoruz
        }

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        try: st.image("logo.png")
        except: pass
        st.markdown("<h3 style='text-align: center;'>MERTT AI</h3>", unsafe_allow_html=True)
        # MENÜYÜ GÜNCELLEDİK
        menu = st.radio("Menü", ["💬 Hisse Sor / Analiz", "📡 Piyasa Radarı", "Çıkış"])
        
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    engine = TradingEngine()

    # --- YENİ BÖLÜM: HİSSE SORMA KISMI ---
    if menu == "💬 Hisse Sor / Analiz":
        st.title("🤖 Yapay Zeka Asistanı")
        st.markdown("Merak ettiğin hisseyi yaz, yapay zeka senin için röntgenini çeksin.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.text_input("Hisse Kodu (Örn: THYAO, SASA):", "").upper()
        with col2:
            st.markdown("<br>", unsafe_allow_html=True) # Hizalama boşluğu
            analyze_btn = st.button("Analiz Et 🔍", type="primary")

        if analyze_btn and symbol:
            with st.spinner(f"{symbol} analiz ediliyor..."):
                res = engine.analyze(symbol)
                
                if res:
                    # 1. ÖZET KARTLARI
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Canlı Fiyat", f"{res['Fiyat']:.2f} TL")
                    k2.metric("AI Güven Skoru", f"%{res['Skor']:.1f}")
                    k3.metric("RSI (Güç)", f"{res['RSI']:.0f}")
                    k4.metric("Risk Seviyesi", "DÜŞÜK" if res['RSI'] < 30 else "YÜKSEK" if res['RSI'] > 70 else "NORMAL")
                    
                    st.divider()
                    
                    # 2. KARAR VE HEDEFLER
                    if res['Renk'] == "green":
                        st.success(f"### 📢 KARAR: {res['Sinyal']}")
                        c1, c2 = st.columns(2)
                        c1.info(f"🛑 **Stop-Loss (Zarar Kes):** {res['Stop']:.2f} TL")
                        c2.success(f"🎯 **Hedef (Kar Al):** {res['Hedef']:.2f} TL")
                    elif res['Renk'] == "red":
                        st.error(f"### 📢 KARAR: {res['Sinyal']}")
                        st.warning("Trend aşağı yönlü. Alım için acele etme.")
                    else:
                        st.warning(f"### 📢 KARAR: {res['Sinyal']}")
                        st.info("Piyasa kararsız. Net bir fırsat görünmüyor.")

                    # 3. GRAFİK
                    st.subheader("📊 Teknik Görünüm")
                    df = res['Data']
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=2), name='VWAP'))
                    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Veri alınamadı veya hisse kodu hatalı.")

    # --- ESKİ BÖLÜM: RADAR ---
    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        st.info("BIST 30 Hisseleri taranıyor...")
        if st.button("TARAMAYI BAŞLAT 🚀"):
            hisseler = ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS"]
            results = []
            bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(engine.analyze, t): t for t in hisseler}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    r = future.result()
                    # Sadece verisi olanları listeye ekle
                    if r: results.append({"Hisse": r['Hisse'], "Fiyat": r['Fiyat'], "Skor": r['Skor'], "Sinyal": r['Sinyal']})
                    completed += 1
                    bar.progress(completed/len(hisseler))
            bar.empty()
            if results:
                st.dataframe(pd.DataFrame(results).style.background_gradient(subset=['Skor'], cmap='Greens'))
            else: st.info("Fırsat yok.")

if __name__ == "__main__":
    main()
            
