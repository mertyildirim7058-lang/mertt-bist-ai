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
import random
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

# --- HAYALET MODÜLÜ (USER-AGENT) ---
def get_stealth_headers():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]
    return {'User-Agent': random.choice(user_agents), 'Referer': 'https://www.google.com/'}

# --- CANLI LİSTE MOTORU ---
@st.cache_data(ttl=600)
def get_live_tickers():
    canli_liste = []
    url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
    
    try:
        response = requests.get(url, headers=get_stealth_headers(), timeout=10)
        tables = pd.read_html(response.text)
        df = tables[0]
        raw_list = df.iloc[:, 0].tolist()
        canli_liste = [str(x).strip() for x in raw_list if str(x).isalnum()]
    except:
        # Manuel parsing yedeği
        try:
            response = requests.get(url, headers=get_stealth_headers(), timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'tableHisseOnerileri'})
            if table:
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if cols: canli_liste.append(cols[0].find('a').text.strip())
        except: pass

    time.sleep(random.uniform(0.5, 1.5))
    return sorted(list(set(canli_liste)))

# --- TEK HİSSE DETAYLI ANALİZ ---
def analyze_single(ticker):
    try:
        t = f"{ticker}.IS"
        # 6 Aylık veri çekiyoruz (MACD ve Bollinger için daha sağlıklı)
        df = yf.download(t, period="6mo", interval="60m", progress=False)
        
        if df is None or len(df) < 100: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        
        df = df.ffill().bfill()
        
        # --- İNDİKATÖRLER (AĞIR SİLAHLAR) ---
        # 1. RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # 2. MACD (Trend)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        
        # 3. Bollinger Bantları (Volatilite)
        bb = ta.bbands(df['Close'], length=20)
        df = pd.concat([df, bb], axis=1) # BBL_5_2.0 (Alt), BBU_5_2.0 (Üst)
        
        # 4. Stochastic RSI (Hassas)
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, stoch], axis=1) # STOCHk_14_3_3, STOCHd_14_3_3
        
        # 5. VWAP & ATR
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        if pd.isna(last['RSI']): return None
        
        # --- PUANLAMA MANTIĞI (0-100) ---
        puan = 50 # Nötr başlangıç
        sebepler = []
        
        # 1. RSI Analizi (Max 20 Puan)
        if last['RSI'] < 30: 
            puan += 20
            sebepler.append("RSI Aşırı Satımda (Ucuz)")
        elif last['RSI'] < 45: 
            puan += 10
        elif last['RSI'] > 70: 
            puan -= 20
            sebepler.append("RSI Aşırı Alımda (Pahalı)")
            
        # 2. MACD Analizi (Max 30 Puan) - Altın Vuruş
        # MACD Çizgisi > Sinyal Çizgisi (AL)
        macd_col = 'MACD_12_26_9'
        macds_col = 'MACDs_12_26_9'
        if last[macd_col] > last[macds_col]:
            puan += 20
            sebepler.append("MACD Al Sinyali (Trend Yukarı)")
            # Yeni kesişimse ekstra puan
            if prev[macd_col] < prev[macds_col]:
                puan += 10
                sebepler.append("MACD Yeni Kesişim! (Güçlü)")
        else:
            puan -= 20
            
        # 3. Bollinger Analizi (Max 20 Puan)
        lower_band = 'BBL_20_2.0'
        upper_band = 'BBU_20_2.0'
        if last['Close'] <= last[lower_band] * 1.02: # Alt banda yakınsa
            puan += 20
            sebepler.append("Bollinger Alt Bandında (Tepki Bekleniyor)")
        elif last['Close'] >= last[upper_band]:
            puan -= 20
            
        # 4. VWAP Analizi (Max 10 Puan)
        if last['Close'] > last['VWAP']:
            puan += 10
        
        # Sınırla
        puan = max(0, min(100, puan))
        
        # Sinyal Kararı
        signal = "NÖTR"
        color = "gray"
        if puan >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
        elif puan >= 60: signal, color = "AL 🌱", "blue"
        elif puan <= 30: signal, color = "SAT 🔻", "red"
            
        return {
            "Fiyat": last['Close'], "RSI": last['RSI'], 
            "Sinyal": signal, "Renk": color, "Skor": puan,
            "Sebepler": sebepler,
            "Stop": last['Close'] - last['ATR']*1.5,
            "Hedef": last['Close'] + last['ATR']*3,
            "Data": df
        }
    except Exception as e: 
        return None

# --- TOPLU ANALİZ (HIZLANDIRILMIŞ) ---
def analyze_batch(tickers_list):
    results = []
    symbols = [f"{t}.IS" for t in tickers_list]
    time.sleep(random.uniform(1.0, 2.0))
    
    try:
        # Sadece 3 aylık veri çek (Hız için, detayda 6 ay çekeriz)
        data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
        
        for ticker in tickers_list:
            try:
                try: df = data[f"{ticker}.IS"].copy()
                except: continue

                if df.empty or df['Close'].isnull().all(): continue
                df = df.dropna()
                if len(df) < 50: continue 
                
                # Hızlı İndikatörler
                rsi = ta.rsi(df['Close'], length=14)
                # MACD
                macd = ta.macd(df['Close'])
                df = pd.concat([df, macd], axis=1)
                
                last = df.iloc[-1]
                prev = df.iloc[-2]
                
                if last['Close'] <= 0 or pd.isna(last['RSI']): continue
                
                # --- HIZLI SKORLAMA ---
                puan = 50
                # RSI
                if last['RSI'] < 30: puan += 25
                elif last['RSI'] < 45: puan += 10
                elif last['RSI'] > 70: puan -= 25
                
                # MACD (Kolon adları kütüphaneye göre standarttır)
                if last['MACD_12_26_9'] > last['MACDs_12_26_9']: puan += 25
                else: puan -= 15
                
                # VWAP (Manuel hesapla)
                vwap = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
                if last['Close'] > vwap.iloc[-1]: puan += 10
                
                puan = max(0, min(100, puan))
                
                signal = "NÖTR"
                if puan >= 75: signal = "GÜÇLÜ AL"
                elif puan >= 60: signal = "AL"
                elif puan <= 30: signal = "SAT"
                
                # Sadece AL/SAT olanları kaydet
                if signal != "NÖTR":
                    results.append({
                        "Hisse": ticker,
                        "Fiyat": last['Close'],
                        "Sinyal": signal,
                        "RSI": last['RSI'],
                        "Skor": int(puan)
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
        menu = st.radio("Panel", ["💬 Detaylı Hisse Analizi", "📡 Piyasa Radarı (Batch)", "Çıkış"])
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    tum_hisseler = get_live_tickers()

    if menu == "💬 Detaylı Hisse Analizi":
        st.title("🤖 Hisse Röntgeni")
        st.info("RSI, MACD, Bollinger ve VWAP analizi yapar.")
        
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu (Örn: THYAO):", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("Analiz Et", type="primary")

        if btn and sembol:
            with st.spinner("Büyük Veri Analiz Ediliyor..."):
                res = analyze_single(sembol)
                if res:
                    # 1. Ana Metrikler
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL")
                    k2.metric("AI Kararı", res['Sinyal'], delta=f"Puan: {res['Skor']}/100")
                    k3.metric("RSI", f"{res['RSI']:.0f}")
                    
                    st.divider()
                    
                    # 2. Yapay Zeka Yorumları (Neden bu puanı verdi?)
                    if res['Sebepler']:
                        st.subheader("💡 Yapay Zeka Tespitleri")
                        for sebep in res['Sebepler']:
                            st.success(f"✅ {sebep}")
                    
                    # 3. Hedefler
                    if res['Renk'] in ['green', 'blue']:
                        c1, c2 = st.columns(2)
                        c1.info(f"🛡️ **Zarar Kes (Stop):** {res['Stop']:.2f} TL")
                        c2.success(f"🎯 **Hedef (Take Profit):** {res['Hedef']:.2f} TL")
                    
                    # 4. Grafik
                    st.subheader("📊 Teknik Grafik")
                    fig = go.Figure()
                    # Mumlar
                    fig.add_trace(go.Candlestick(x=res['Data'].index, open=res['Data']['Open'], high=res['Data']['High'], low=res['Data']['Low'], close=res['Data']['Close'], name="Fiyat"))
                    # Bollinger
                    fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data']['BBU_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name='Bollinger Üst'))
                    fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data']['BBL_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name='Bollinger Alt'))
                    # VWAP
                    fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data']['VWAP'], line=dict(color='orange', width=2), name='VWAP'))
                    
                    fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Veri bulunamadı.")

    elif menu == "📡 Piyasa Radarı (Batch)":
        st.title("📡 MERTT Piyasa Radarı")
        
        if not tum_hisseler:
            st.error("⚠️ Liste çekilemedi. Lütfen sayfayı yenileyin.")
            st.stop()
            
        st.info(f"Takipteki Hisse Sayısı: {len(tum_hisseler)}")
        
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
            else:
                st.warning("Hiçbir sinyal bulunamadı.")

if __name__ == "__main__":
    main()
    
