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
import time
import numpy as np

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

# --- HİSSE LİSTESİ ALTYAPISI ---
@st.cache_data(ttl=600)
def tum_hisseleri_getir():
    """Canlı çeker, olmazsa YEDEK LİSTE"""
    canli_liste = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'tableHisseOnerileri'})
            if table:
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if cols: canli_liste.append(cols[0].find('a').text.strip())
    except: pass
    
    if len(canli_liste) > 50: 
        return sorted(list(set(canli_liste)))
    else:
        # Site çalışmazsa temel BIST 100 listesi
        return ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS", "HEKTS", "PETKM", "ISCTR", "SAHOL", "FROTO", "YKBNK", "EKGYO", "ODAS", "KOZAL", "KONTR", "ASTOR", "EUPWR", "GUBRF", "OYAKC", "TCELL", "TTKOM", "ENKAI", "VESTL", "ARCLK", "TOASO", "PGSUS", "TAVHL", "MGROS", "SOKM", "AEFES", "AGHOL", "AHGAZ", "AKFGY", "AKSA", "AKSEN", "ALARK", "ALBRK", "ALFAS", "ANSGR", "ARASE", "BERA", "BIOEN", "BOBET", "BRSAN", "BRYAT", "BUCIM", "CANTE", "CCOLA", "CEMTS", "CIMSA", "CWENE", "DOAS", "DOHOL", "ECILC", "ECZYT", "EGEEN", "ENJSA", "ENVER", "ERBOS", "EUREN", "FENE", "GENIL", "GESAN", "GLYHO", "GSDHO", "GWIND", "HALKB", "ISDMR", "ISGYO", "ISMEN", "IZMDC", "KARSN", "KAYSE", "KCAER", "KMPUR", "KORDS", "KOZAA", "KZBGY", "MAVI", "MIATK", "OTKAR", "OYYAT", "PENTA", "QUAGR", "REEDR", "SANTM", "SMRTG", "SKBNK", "SNGYO", "TATGD", "TKFEN", "TMSN", "TSKB", "TURSG", "ULKER", "VAKBN", "VESBE", "YEOTK", "YYLGD", "ZOREN"]

# --- ANALİZ MOTORU (DÜZELTİLMİŞ) ---
class TradingEngine:
    def __init__(self):
        try: from sklearn.preprocessing import StandardScaler
        except: pass
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    
    def get_live_price(self, ticker):
        """
        DÜZELTME: Sayfanın gerçekten o hisseye ait olup olmadığını kontrol eder.
        Yanlış yönlendirme (Redirect) varsa veriyi almaz.
        """
        try:
            clean_ticker = ticker.replace('.IS','')
            url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{clean_ticker}-detay/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=3)
            
            # 1. KONTROL: Sayfa başlığında hisse adı geçiyor mu?
            if clean_ticker not in resp.text:
                return None # Yanlış sayfa, çık.

            soup = BeautifulSoup(resp.content, "html.parser")
            price_span = soup.find("span", {"class": "text-2"})
            if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
            
            if price_span: 
                price = float(price_span.text.strip().replace(',', '.'))
                if price > 0: return price
            return None
        except: return None

    def analyze(self, ticker):
        if not ticker.endswith('.IS'): ticker += '.IS'
        
        # Throttling önlemek için rastgele mini bekleme
        time.sleep(random.uniform(0.1, 0.5))
        
        try:
            # 1. Yahoo Finance Verisi (Ana Kaynak)
            df = yf.download(ticker, period="3mo", interval="60m", progress=False)
            
            # --- FİLTRE 1: VERİ YOKSA ---
            if df is None or df.empty or len(df) < 50: return None
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            df = df.ffill().bfill()
            
            # --- FİLTRE 2: SON FİYAT SIFIRSA ---
            last_graph_price = df.iloc[-1]['Close']
            if last_graph_price <= 0: return None

            # 2. Canlı Fiyat Kontrolü (Hata Önleyici)
            live_price = self.get_live_price(ticker)
            
            if live_price:
                # Canlı fiyat ile grafik fiyatı arasında %20'den fazla fark varsa
                # Muhtemelen canlı veri yanlıştır (BIMAS verisi çekmiştir vs.)
                fark_orani = abs(live_price - last_graph_price) / last_graph_price
                if fark_orani < 0.20:
                    # Güvenilir, kullan
                    df.iloc[-1, df.columns.get_loc('Close')] = live_price
                else:
                    # Güvenilmez, grafikteki son fiyatı kullanmaya devam et
                    pass

            # 3. İndikatörler
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
            rsi_val = last['RSI']
            
            # --- FİLTRE 3: RSI SAÇMALAMIŞSA ---
            if pd.isna(rsi_val) or rsi_val <= 1 or rsi_val >= 99: return None

            signal, color = "NÖTR / İZLE", "gray"
            stop = last['Close'] - (last['ATR'] * 1.5)
            target = last['Close'] + (last['ATR'] * 3.0)

            if prob > 60 and last['Close'] > last['VWAP']: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif prob < 40 and last['Close'] < last['VWAP']: signal, color = "SAT 🔻", "red"
            
            return {
                "Hisse": ticker.replace('.IS',''), 
                "Fiyat": last['Close'], 
                "Skor": prob, 
                "RSI": rsi_val, 
                "Sinyal": signal, 
                "Renk": color, 
                "Stop": stop, 
                "Hedef": target,
                "Data": df
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
                else: st.error("Hisse bulunamadı veya verisi bozuk.")

    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        st.info(f"Takipteki Hisse: {len(tum_hisseler)}")
        
        if st.button("TÜM BORSAYI TARA 🚀", type="primary"):
            random.shuffle(tum_hisseler)
            secilenler = tum_hisseler 
            results = []
            bar_text = st.empty()
            bar = st.progress(0)
            
            # Worker sayısını düşürdüm (10) ki Yahoo Finance engellemesin
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(engine.analyze, t): t for t in secilenler}
                done = 0
                for future in concurrent.futures.as_completed(futures):
                    r = future.result()
                    # FİYAT SIFIRDAN BÜYÜKSE VE SİNYAL VARSA
                    if r and (r['Renk'] == 'green' or r['Renk'] == 'red') and r['Fiyat'] > 0:
                        results.append(r)
                    
                    done += 1
                    bar.progress(done/len(secilenler))
                    bar_text.text(f"Analiz ediliyor: {done}/{len(secilenler)}")
            
            bar.empty()
            bar_text.empty()
            
            if results:
                st.success(f"Tarama Tamamlandı! {len(results)} Fırsat Bulundu.")
                df = pd.DataFrame(results)
                
                try:
                    st.dataframe(
                        df[['Hisse', 'Fiyat', 'Sinyal', 'Skor', 'RSI']]
                        .style.format({"Fiyat": "{:.2f}", "Skor": "{:.1f}", "RSI": "{:.0f}"})
                        .background_gradient(subset=['Skor'], cmap='Greens'),
                        use_container_width=True
                    )
                except: st.dataframe(df, use_container_width=True)
            else:
                st.warning("Piyasada şu an net bir sinyal bulunamadı.")

if __name__ == "__main__":
    main()
    
