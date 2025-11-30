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
import random

# --- 1. AYARLAR ---
LOGO_INTERNET_LINKI = "https://raw.githubusercontent.com/kullaniciadi/proje/main/logo.png"

st.set_page_config(
    page_title="MERTT AI", 
    layout="wide", 
    page_icon="🛡️"  
)

def logo_goster():
    try: st.image("logo.png", use_container_width=True)
    except:
        try: st.image(LOGO_INTERNET_LINKI, use_container_width=True)
        except: st.header("🦅 MERTT AI")

def pwa_kodlari():
    pwa_html = f"""
    <meta name="theme-color" content="#0e1117">
    <link rel="apple-touch-icon" href="{LOGO_INTERNET_LINKI}">
    <link rel="icon" type="image/png" href="{LOGO_INTERNET_LINKI}">
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
            logo_goster()
            st.markdown("<h4 style='text-align: center;'>Gelecek İçin Bilgi ve Teknoloji</h4>", unsafe_allow_html=True)
            st.divider()
            sifre = st.text_input("Erişim Anahtarı:", type="password")
            if st.button("Sisteme Giriş Yap", type="primary", use_container_width=True):
                try:
                    if sifre == st.secrets["GIRIS_SIFRESI"]: 
                        st.session_state['giris_yapildi'] = True
                        st.rerun()
                    else: st.error("⛔ Yetkisiz Erişim!")
                except: st.error("Sistem Hatası: Şifre tanımlı değil.")
        return False
    return True

if not guvenlik_kontrolu(): st.stop()

# --- CANLI HİSSE LİSTESİ (YEDEKSİZ) ---
@st.cache_data(ttl=600) # 10 dakikada bir yenile
def tum_hisseleri_getir():
    """
    Sadece canlı veriyi çeker. Çekemezse BOŞ liste döner.
    Yedek liste İPTAL edildi.
    """
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        # Timeout süresini biraz artırdık ki hemen pes etmesin
        response = requests.get(url, timeout=10) 
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'tableHisseOnerileri'})
            
            if table:
                rows = table.find('tbody').find_all('tr')
                hisseler = []
                for row in rows:
                    cols = row.find_all('td')
                    if cols:
                        code = cols[0].find('a').text.strip()
                        hisseler.append(code)
                
                # Eğer liste doluysa döndür
                if len(hisseler) > 0:
                    return sorted(list(set(hisseler)))
    except:
        pass
    
    # Hata olursa veya site açılmazsa BOŞ liste dön
    return []

# --- ANALİZ MOTORU ---
class TradingEngine:
    def __init__(self):
        try: from sklearn.preprocessing import StandardScaler
        except: pass
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    
    def get_live_price(self, ticker):
        try:
            url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker.replace('.IS','')}-detay/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=3)
            soup = BeautifulSoup(resp.content, "html.parser")
            price_span = soup.find("span", {"class": "text-2"})
            if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
            if price_span: return float(price_span.text.strip().replace(',', '.'))
            return None
        except: return None

    def analyze(self, ticker):
        if not ticker.endswith('.IS'): ticker += '.IS'
        try:
            df = yf.download(ticker, period="1mo", interval="60m", progress=False)
            if df is None or df.empty or len(df) < 50: return None
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            df = df.ffill().bfill()
            
            live_price = self.get_live_price(ticker)
            if live_price and (abs(live_price - df.iloc[-1]['Close']) / df.iloc[-1]['Close'] < 0.15):
                df.iloc[-1, df.columns.get_loc('Close')] = live_price

            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            clean_df = df.dropna()
            if len(clean_df) < 20: return None

            features = ['RSI', 'VWAP', 'ATR']
            self.model.fit(clean_df.iloc[:-1][features], clean_df.iloc[:-1]['Target'])
            prob = self.model.predict_proba(clean_df.iloc[[-1]][features])[0][1] * 100
            
            last = df.iloc[-1]
            if pd.isna(last['RSI']) or pd.isna(last['Skor']): return None

            signal, color = "NÖTR / İZLE", "gray"
            stop = last['Close'] - (last['ATR'] * 1.5)
            target = last['Close'] + (last['ATR'] * 3.0)

            if prob > 60 and last['Close'] > last['VWAP']: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif prob < 40 and last['Close'] < last['VWAP']: signal, color = "SAT 🔻", "red"
            
            return {
                "Hisse": ticker.replace('.IS',''), "Fiyat": last['Close'], "Skor": prob, 
                "RSI": last['RSI'], "Sinyal": signal, "Renk": color, 
                "Stop": stop, "Hedef": target, "Data": df
            }
        except: return None

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.markdown("<h3 style='text-align: center;'>Yapay Zeka Üssü</h3>", unsafe_allow_html=True)
        st.divider()
        menu = st.radio("Panel", ["💬 Hisse Sor", "📡 Piyasa Radarı", "Çıkış"])
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    engine = TradingEngine()
    
    # Hisseleri çek (Boş gelirse boş kalır, yedek yok)
    tum_hisseler = tum_hisseleri_getir()

    if menu == "💬 Hisse Sor":
        st.title("🤖 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu (Örn: THYAO):", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("Analiz Et", type="primary")

        if btn and sembol:
            with st.spinner(f"{sembol} analiz ediliyor..."):
                res = engine.analyze(sembol)
                if res:
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f}")
                    k2.metric("AI Güveni", f"%{res['Skor']:.1f}")
                    k3.metric("RSI (14)", f"{res['RSI']:.0f}")
                    st.divider()
                    if res['Renk'] == 'green':
                        st.success(f"### {res['Sinyal']}")
                        st.info(f"🛡️ Stop: {res['Stop']:.2f} | 🎯 Hedef: {res['Hedef']:.2f}")
                    elif res['Renk'] == 'red': st.error(f"### {res['Sinyal']}")
                    else: st.warning(f"### {res['Sinyal']}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=res['Data'].index, open=res['Data']['Open'], high=res['Data']['High'], low=res['Data']['Low'], close=res['Data']['Close']))
                    fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data']['VWAP'], line=dict(color='orange'), name='VWAP'))
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Hisse verisi bulunamadı.")

    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        
        # LİSTE KONTROLÜ
        if not tum_hisseler:
            st.error("⚠️ HATA: Canlı Borsa verisine ulaşılamadı!")
            st.warning("İş Yatırım sitesi yanıt vermiyor olabilir. Lütfen daha sonra tekrar deneyin.")
            st.stop() # Programı burada durdur, aşağıyı çalıştırma
            
        st.info(f"{len(tum_hisseler)} Hisse Canlı Olarak Takipte.")
        tarama_limiti = st.slider("Tarama Hızı (Hisse Sayısı)", 10, 50, 20)
        
        if st.button("TARAMAYI BAŞLAT 🚀"):
            random.shuffle(tum_hisseler)
            secilenler = tum_hisseler[:tarama_limiti]
            results = []
            bar = st.progress(0)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(engine.analyze, t): t for t in secilenler}
                done = 0
                for future in concurrent.futures.as_completed(futures):
                    r = future.result()
                    if r: results.append(r)
                    done += 1
                    bar.progress(done/len(secilenler))
            bar.empty()
            
            if results:
                df = pd.DataFrame(results)
                try:
                    st.dataframe(
                        df[['Hisse', 'Fiyat', 'Sinyal', 'Skor', 'RSI']]
                        .style.format({"Fiyat": "{:.2f}", "Skor": "{:.1f}", "RSI": "{:.0f}"})
                        .background_gradient(subset=['Skor'], cmap='Greens'),
                        use_container_width=True
                    )
                except: st.dataframe(df, use_container_width=True)
            else: st.warning("Seçilen hisselerde sinyal bulunamadı.")

if __name__ == "__main__":
    main()
