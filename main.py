import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests
import time

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

# --- CANLI LİSTE MOTORU (YEDEKSİZ) ---
@st.cache_data(ttl=600) # 10 dakikada bir yenile
def get_live_tickers():
    """
    Sadece İş Yatırım sitesinden canlı listeyi çeker.
    Yedek liste yoktur. Siteye ulaşamazsa BOŞ döner.
    """
    canli_liste = []
    try:
        # Robot korumasını aşmak için User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'tableHisseOnerileri'})
            
            if table:
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if cols:
                        # Hisse kodunu al (Örn: THYAO)
                        code = cols[0].find('a').text.strip()
                        canli_liste.append(code)
    except:
        pass
    
    return sorted(list(set(canli_liste)))

# --- TEK HİSSE ANALİZİ (Manuel Sorgu) ---
def analyze_single(ticker):
    try:
        t = f"{ticker}.IS"
        # 3 aylık veri çekiyoruz ki RSI otursun
        df = yf.download(t, period="3mo", interval="60m", progress=False)
        
        if df is None or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        
        # Boşlukları doldur
        df = df.ffill().bfill()
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        last = df.iloc[-1]
        if pd.isna(last['RSI']): return None
        
        signal = "NÖTR"
        color = "gray"
        if last['RSI'] < 45 and last['Close'] > last['VWAP']: 
            signal = "GÜÇLÜ AL"
            color = "green"
        elif last['RSI'] > 70:
            signal = "SAT"
            color = "red"
            
        return {
            "Fiyat": last['Close'], "RSI": last['RSI'], 
            "Sinyal": signal, "Renk": color, 
            "Stop": last['Close'] - last['ATR']*1.5,
            "Hedef": last['Close'] + last['ATR']*3,
            "Data": df
        }
    except: return None

# --- TOPLU ANALİZ (Batch - Turbo Mod) ---
def analyze_batch(tickers_list):
    results = []
    symbols = [f"{t}.IS" for t in tickers_list]
    
    try:
        # Toplu veri indirme (Hızlı)
        data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
        
        for ticker in tickers_list:
            try:
                # Veriyi al
                try:
                    df = data[f"{ticker}.IS"].copy()
                except:
                    continue # Veri yoksa atla

                # Veri Kontrolü
                if df.empty or df['Close'].isnull().all(): continue
                
                # HALKA ARZ FİLTRESİ:
                # Eğer hissenin verisi 50 mumdan azsa (Çok yeni arz), analiz etme.
                df = df.dropna()
                if len(df) < 50: continue 
                
                # İndikatörler
                rsi = ta.rsi(df['Close'], length=14)
                vwap = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
                
                last_close = df['Close'].iloc[-1]
                last_rsi = rsi.iloc[-1]
                last_vwap = vwap.iloc[-1]
                
                if last_close <= 0 or pd.isna(last_rsi): continue
                
                signal = "NÖTR"
                skor = 50
                
                # Strateji
                if last_rsi < 45 and last_close > last_vwap:
                    signal = "GÜÇLÜ AL"
                    skor = 85
                elif last_rsi > 75:
                    signal = "SAT"
                    skor = 20
                elif last_close < last_vwap and last_rsi < 50:
                    signal = "DÜŞÜŞ TRENDİ"
                    skor = 30
                
                if "AL" in signal or "SAT" in signal or "DÜŞÜŞ" in signal:
                    results.append({
                        "Hisse": ticker,
                        "Fiyat": last_close,
                        "Sinyal": signal,
                        "RSI": last_rsi,
                        "Skor": skor
                    })
            except: continue
    except: pass
    return results

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.markdown("<h3 style='text-align: center;'>Yapay Zeka Üssü</h3>", unsafe_allow_html=True)
        st.divider()
        menu = st.radio("Panel", ["💬 Hisse Sor", "📡 Piyasa Radarı (Batch)", "Çıkış"])
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    # --- CANLI LİSTE ÇEKİMİ ---
    tum_hisseler = get_live_tickers()

    if menu == "💬 Hisse Sor":
        st.title("🤖 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu (Örn: THYAO):", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("Analiz Et", type="primary")

        if btn and sembol:
            with st.spinner("Analiz ediliyor..."):
                res = analyze_single(sembol)
                if res:
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f}")
                    k2.metric("Sinyal", res['Sinyal'])
                    k3.metric("RSI", f"{res['RSI']:.0f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=res['Data'].index, open=res['Data']['Open'], high=res['Data']['High'], low=res['Data']['Low'], close=res['Data']['Close']))
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Veri bulunamadı.")

    elif menu == "📡 Piyasa Radarı (Batch)":
        st.title("📡 MERTT Piyasa Radarı")
        
        # Eğer liste boşsa (Siteye ulaşılamadıysa) DURDUR.
        if not tum_hisseler:
            st.error("⚠️ HATA: Canlı borsa listesine ulaşılamıyor!")
            st.warning("İş Yatırım sitesi yanıt vermiyor olabilir. Eski veri kullanmamak için işlem durduruldu.")
            st.stop()
            
        st.info(f"Canlı Takipteki Hisse Sayısı: {len(tum_hisseler)}")
        
        if st.button("TÜM BORSAYI TARA (Turbo Mod) 🚀", type="primary"):
            all_results = []
            chunk_size = 50 
            chunks = [tum_hisseler[i:i + chunk_size] for i in range(0, len(tum_hisseler), chunk_size)]
            
            bar = st.progress(0)
            status = st.empty()
            
            for i, chunk in enumerate(chunks):
                status.text(f"Analiz Paketi {i+1}/{len(chunks)} işleniyor...")
                batch_res = analyze_batch(chunk)
                all_results.extend(batch_res)
                bar.progress((i + 1) / len(chunks))
                time.sleep(1) 
            
            bar.empty()
            status.empty()
            
            if all_results:
                df = pd.DataFrame(all_results)
                try:
                    st.success(f"Tarama Bitti! {len(df)} Sinyal Bulundu.")
                    st.dataframe(
                        df.style.format({"Fiyat": "{:.2f}", "RSI": "{:.0f}", "Skor": "{:.0f}"})
                        .background_gradient(subset=['Skor'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                except: st.dataframe(df)
            else:
                st.warning("Hiçbir sinyal bulunamadı.")

if __name__ == "__main__":
    main()
    
