import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import time
from PIL import Image

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

# --- SADECE CANLI LİSTE (YEDEK YOK) ---
@st.cache_data(ttl=300) # 5 dakikada bir yenile
def get_live_tickers():
    """
    Sadece canlı veriyi çekmeye çalışır.
    Başarısız olursa BOŞ liste döner. Yedek yoktur.
    """
    canli_liste = []
    url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
    
    try:
        # Yöntem 1: Pandas ile Tablo Okuma (En Kapsamlısı)
        tables = pd.read_html(url)
        df = tables[0]
        # İlk sütun hisse kodlarıdır
        raw_list = df.iloc[:, 0].tolist()
        canli_liste = [str(x).strip() for x in raw_list if str(x).isalnum()]
        
    except Exception as e:
        # Yöntem 2: Pandas çalışmazsa Requests ile dene
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'tableHisseOnerileri'})
            if table:
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if cols: canli_liste.append(cols[0].find('a').text.strip())
        except: pass

    # Sonuç Dön
    return sorted(list(set(canli_liste)))

# --- TEK HİSSE ANALİZİ ---
def analyze_single(ticker):
    try:
        t = f"{ticker}.IS"
        df = yf.download(t, period="3mo", interval="60m", progress=False)
        
        if df is None or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        
        df = df.ffill().bfill()
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        last = df.iloc[-1]
        if pd.isna(last['RSI']): return None
        
        signal = "NÖTR"
        color = "gray"
        skor = 50
        
        if last['RSI'] < 45 and last['Close'] > last['VWAP']: 
            signal = "GÜÇLÜ AL"
            color = "green"
            skor = min(95, 50 + (50 - last['RSI']) * 1.5)
        elif last['RSI'] > 70:
            signal = "SAT"
            color = "red"
            skor = min(90, (last['RSI'] - 50) * 2)
            
        return {
            "Fiyat": last['Close'], "RSI": last['RSI'], 
            "Sinyal": signal, "Renk": color, "Skor": int(skor),
            "Stop": last['Close'] - last['ATR']*1.5,
            "Hedef": last['Close'] + last['ATR']*3,
            "Data": df
        }
    except: return None

# --- TOPLU ANALİZ (Batch) ---
def analyze_batch(tickers_list):
    results = []
    symbols = [f"{t}.IS" for t in tickers_list]
    
    try:
        data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
        
        for ticker in tickers_list:
            try:
                try: df = data[f"{ticker}.IS"].copy()
                except: continue

                if df.empty or df['Close'].isnull().all(): continue
                df = df.dropna()
                if len(df) < 50: continue 
                
                rsi = ta.rsi(df['Close'], length=14)
                vwap = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
                
                last_close = df['Close'].iloc[-1]
                last_rsi = rsi.iloc[-1]
                last_vwap = vwap.iloc[-1]
                
                if last_close <= 0 or pd.isna(last_rsi): continue
                
                signal = "NÖTR"
                skor = 50 
                
                if last_rsi < 45 and last_close > last_vwap:
                    signal = "GÜÇLÜ AL"
                    skor = 50 + ((50 - last_rsi) * 2)
                    if skor > 99: skor = 99
                elif last_rsi > 75:
                    signal = "SAT"
                    skor = (last_rsi - 50) * 2
                    if skor > 95: skor = 95
                elif last_close < last_vwap and last_rsi < 50:
                    signal = "DÜŞÜŞ TRENDİ"
                    skor = 30
                
                if "AL" in signal or "SAT" in signal or "DÜŞÜŞ" in signal:
                    results.append({
                        "Hisse": ticker,
                        "Fiyat": last_close,
                        "Sinyal": signal,
                        "RSI": last_rsi,
                        "Skor": int(skor)
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

    # CANLI LİSTE ÇEK
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
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL")
                    k2.metric("Sinyal", res['Sinyal'], delta=f"Güven: %{res['Skor']}")
                    k3.metric("RSI", f"{res['RSI']:.0f}")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=res['Data'].index, open=res['Data']['Open'], high=res['Data']['High'], low=res['Data']['Low'], close=res['Data']['Close']))
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Veri bulunamadı.")

    elif menu == "📡 Piyasa Radarı (Batch)":
        st.title("📡 MERTT Piyasa Radarı")
        
        # Eğer liste boşsa HATA VER ve DUR
        if not tum_hisseler:
            st.error("⚠️ KRİTİK HATA: Canlı Borsa Listesine Ulaşılamıyor!")
            st.warning("İş Yatırım sitesi cevap vermiyor. Eski verilerle işlem yapmamak için sistem durduruldu.")
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
                        df,
                        column_config={
                            "Hisse": st.column_config.TextColumn("Hisse Kodu"),
                            "Fiyat": st.column_config.NumberColumn("Fiyat", format="%.2f TL"),
                            "Sinyal": st.column_config.TextColumn("AI Kararı"),
                            "RSI": st.column_config.NumberColumn("RSI Gücü", format="%.0f"),
                            "Skor": st.column_config.ProgressColumn(
                                "Güven Skoru",
                                format="%d",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                except: st.dataframe(df)
            else:
                st.warning("Hiçbir sinyal bulunamadı.")

if __name__ == "__main__":
    main()
    
