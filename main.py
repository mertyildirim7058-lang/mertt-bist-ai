import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import numpy as np
import plotly.graph_objects as go
import requests
import feedparser
from bs4 import BeautifulSoup
import time
import random
from PIL import Image
from datetime import datetime

# --- 1. AYARLAR & KONFİGÜRASYON ---
LOGO_INTERNET_LINKI = "https://raw.githubusercontent.com/kullaniciadi/proje/main/logo.png"

st.set_page_config(
    page_title="MERTT AI Quantum", 
    layout="wide", 
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# --- GÖRSELLEŞTİRME YARDIMCILARI ---
def logo_goster():
    try: st.image("logo.png", use_container_width=True)
    except:
        try: st.image(LOGO_INTERNET_LINKI, use_container_width=True)
        except: st.header("🦅 MERTT AI")

def pwa_kodlari():
    pwa_html = f"""
    <meta name="theme-color" content="#000000">
    <link rel="apple-touch-icon" href="{LOGO_INTERNET_LINKI}">
    <link rel="icon" type="image/png" href="{LOGO_INTERNET_LINKI}">
    <style>
        .stApp {{ background-color: #0e1117; }}
        .stButton>button {{ width: 100%; border-radius: 10px; }}
    </style>
    """
    components.html(f"<html><head>{pwa_html}</head></html>", height=0, width=0)
pwa_kodlari()

# --- GÜVENLİK DUVARI ---
def guvenlik_kontrolu():
    if 'giris_yapildi' not in st.session_state: st.session_state['giris_yapildi'] = False
    if not st.session_state['giris_yapildi']:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            logo_goster()
            st.markdown("<h4 style='text-align: center; color: #4CAF50;'>Gelecek İçin Bilgi ve Teknoloji</h4>", unsafe_allow_html=True)
            st.divider()
            sifre = st.text_input("Kuantum Erişim Anahtarı:", type="password")
            if st.button("Sisteme Bağlan", type="primary", use_container_width=True):
                try:
                    if sifre == st.secrets["GIRIS_SIFRESI"]: 
                        st.session_state['giris_yapildi'] = True
                        st.rerun()
                    else: st.error("⛔ Yetkisiz Erişim Denemesi!")
                except: st.error("Sistem Hatası: Şifre tanımlı değil.")
        return False
    return True

if not guvenlik_kontrolu(): st.stop()

# --- VERİ MOTORLARI ---

# 1. CANLI LİSTE (Yedeksiz, Saf Canlı)
@st.cache_data(ttl=600)
def get_live_tickers():
    canli_liste = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'id': 'tableHisseOnerileri'})
            if table:
                rows = table.find('tbody').find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if cols: canli_liste.append(cols[0].find('a').text.strip())
    except: pass
    return sorted(list(set(canli_liste)))

# 2. CANLI FİYAT (Sniper Mode)
def get_realtime_price(ticker):
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker.replace('.IS','')}-detay/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(resp.content, "html.parser")
        price_span = soup.find("span", {"class": "text-2"})
        if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
        if price_span: return float(price_span.text.strip().replace(',', '.'))
    except: return None

# 3. KÜRESEL İSTİHBARAT (Global Intel)
class GlobalIntel:
    def __init__(self):
        self.risk_keywords = ['savaş', 'kriz', 'çöküş', 'enflasyon', 'faiz', 'gerilim', 'yaptırım']
        self.tech_keywords = ['yapay zeka', 'rekor', 'büyüme', 'anlaşma', 'onay', 'ihracat', 'yatırım', 'temettü']

    def get_global_indices(self):
        indices = {"S&P 500": "^GSPC", "Altın": "GC=F", "Bitcoin": "BTC-USD", "Dolar": "DX-Y.NYB"}
        data = {}
        try:
            tickers = " ".join(indices.values())
            df = yf.download(tickers, period="2d", progress=False)['Close']
            for name, symbol in indices.items():
                try:
                    price = df[symbol].iloc[-1]
                    prev = df[symbol].iloc[-2]
                    change = ((price - prev) / prev) * 100
                    data[name] = {"Fiyat": price, "Degisim": change}
                except: data[name] = {"Fiyat": 0, "Degisim": 0}
        except: pass
        return data

    def analyze_news(self, ticker=""):
        sentiment = 0
        news_list = []
        if ticker: query = f"{ticker} hisse kap"
        else: query = "Borsa İstanbul Ekonomi"
        
        url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = entry.title.lower()
                news_list.append({"Title": entry.title, "Link": entry.link, "Date": entry.published})
                for w in self.tech_keywords: 
                    if w in title: sentiment += 2
                for w in self.risk_keywords: 
                    if w in title: sentiment -= 3
        except: pass
        return sentiment, news_list

# --- 4. KUANTUM ANALİZ MOTORU ---
class TradingEngine:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        self.intel = GlobalIntel()

    def get_fundamentals(self, ticker):
        try:
            stock = yf.Ticker(f"{ticker}.IS")
            info = stock.info
            # Güvenli veri çekme
            fk = info.get('trailingPE', None)
            pddd = info.get('priceToBook', None)
            
            yorum = "NÖTR"
            if fk and fk < 8 and pddd and pddd < 2: yorum = "UCUZ (KELEPİR)"
            elif fk and fk > 35: yorum = "PAHALI"
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-", "Yorum": yorum}
        except: return None

    def analyze(self, ticker, mode="BATCH"):
        """
        mode='BATCH': Hızlı tarama (Yfinance verisi)
        mode='PRO': Detaylı analiz (Canlı veri yaması + Haberler + Temel)
        """
        try:
            t = f"{ticker}.IS"
            # Veri Çekme
            df = yf.download(t, period="6mo", interval="60m", progress=False)
            if df is None or len(df) < 100: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            
            # TR Saati Ayarı
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Istanbul')
            df = df.ffill().bfill()

            is_live = False
            # PRO MOD: Canlı Fiyat Yaması
            if mode == "PRO":
                live_price = get_realtime_price(ticker)
                if live_price:
                    if abs(live_price - df.iloc[-1]['Close']) / df.iloc[-1]['Close'] < 0.20:
                        df.iloc[-1, df.columns.get_loc('Close')] = live_price
                        # Mumun diğer kısımlarını da düzelt
                        if live_price > df.iloc[-1]['High']: df.iloc[-1, df.columns.get_loc('High')] = live_price
                        if live_price < df.iloc[-1]['Low']: df.iloc[-1, df.columns.get_loc('Low')] = live_price
                        is_live = True

            # --- İNDİKATÖRLER ---
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            bb = ta.bbands(df['Close'], length=20)
            df = pd.concat([df, bb], axis=1)
            
            # Ichimoku & PSAR (Sadece PRO Modda görselleştirilir ama hesaplanır)
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
            df = pd.concat([df, ichimoku], axis=1)
            psar = ta.psar(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, psar], axis=1)
            psar_col = [c for c in df.columns if c.startswith('PSAR')][0]

            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            last = df.iloc[-1]
            if pd.isna(last['RSI']): return None

            # --- PUANLAMA ---
            score = 50
            reasons = []

            # Teknik Puanlar
            if last['Close'] > last['VWAP']: score += 10; reasons.append("Fiyat VWAP Üzerinde")
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 15; reasons.append("MACD Al Sinyali")
            if df[psar_col].iloc[-1] < last['Close']: score += 10; reasons.append("PSAR Yükseliş")
            
            if last['RSI'] < 30: score += 20; reasons.append("RSI Aşırı Satım (Fırsat)")
            elif last['RSI'] > 70: score -= 15; reasons.append("RSI Aşırı Alım")
            
            span_a = df['ISA_9'].iloc[-1]
            span_b = df['ISB_26'].iloc[-1]
            if last['Close'] > span_a and last['Close'] > span_b: score += 15; reasons.append("Ichimoku Bulutu Üstünde")

            # PRO MOD: Haber Analizi
            news_data = None
            if mode == "PRO":
                news_score, news_list = self.intel.analyze_news(ticker)
                score += news_score
                news_data = news_list
                if news_score > 0: reasons.append("Haber Akışı Pozitif")
            
            score = max(0, min(100, score))
            
            # Karar
            signal, color = "NÖTR / İZLE", "gray"
            if score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif score >= 60: signal, color = "AL 🌱", "blue"
            elif score <= 30: signal, color = "SAT 🔻", "red"

            stop = last['Close'] - (last['ATR'] * 1.5)
            hedef = last['Close'] + (last['ATR'] * 3.0)
            
            # PRO Modda Temel Analiz de döner
            temel = self.get_fundamentals(ticker) if mode == "PRO" else None

            return {
                "Hisse": ticker, "Fiyat": last['Close'], "Skor": int(score),
                "Sinyal": signal, "Renk": color, "RSI": last['RSI'],
                "Stop": stop, "Hedef": hedef, "Yorumlar": reasons, 
                "Data": df, "Tarih": df.index[-1].strftime('%d %B %H:%M'),
                "Is_Live": is_live, "Temel": temel, "Haberler": news_data
            }
        except: return None

    # Batch Tarama (Hızlandırılmış)
    def analyze_batch_fast(self, tickers_list):
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
                    last = df.iloc[-1]
                    if last['Close'] <= 0 or pd.isna(last.name): continue # RSI NaN check
                    
                    # Hızlı Skor
                    score = 50
                    if rsi.iloc[-1] < 45 and last['Close'] > vwap.iloc[-1]: score = 85
                    elif rsi.iloc[-1] > 70: score = 20
                    
                    signal = "NÖTR"
                    if score >= 80: signal = "GÜÇLÜ AL"
                    elif score <= 30: signal = "SAT"
                    
                    if signal != "NÖTR":
                        results.append({"Hisse": ticker, "Fiyat": last['Close'], "Sinyal": signal, "RSI": rsi.iloc[-1], "Skor": score})
                except: continue
        except: pass
        return results

# --- ARAYÜZ (FRONTEND) ---
def main():
    with st.sidebar:
        logo_goster()
        st.markdown("<h3 style='text-align: center;'>MERTT AI</h3>", unsafe_allow_html=True)
        st.caption("v30.0 - Quantum Edition")
        st.divider()
        menu = st.radio("Panel", ["🖥️ PRO Analiz Masası", "📡 Piyasa Radarı", "Çıkış"])
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    engine = TradingEngine()
    intel = GlobalIntel()
    
    # Canlı Listeyi Çek
    tum_hisseler = get_live_tickers()

    if menu == "🖥️ PRO Analiz Masası":
        st.title("🖥️ PRO Terminal (Haber + Teknik + Temel)")
        
        # Global Pano
        st.markdown("### 🌍 Küresel Piyasalar")
        indices = intel.get_global_indices()
        if indices:
            cols = st.columns(len(indices))
            for i, (name, data) in enumerate(indices.items()):
                cols[i].metric(label=name, value=f"{data['Fiyat']:.2f}", delta=f"%{data['Degisim']:.2f}")
        st.divider()

        if not tum_hisseler:
            st.error("⚠️ Liste çekilemedi. Bağlantıyı kontrol edin.")
            st.stop()

        c1, c2 = st.columns([3,1])
        with c1: secilen_hisse = st.selectbox("Hisse Seçin:", ["THYAO"] + tum_hisseler)
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("ANALİZİ BAŞLAT ⚡", type="primary")

        if btn:
            with st.spinner("Büyük Veri Analiz Ediliyor..."):
                res = engine.analyze(secilen_hisse, mode="PRO")
                
                if res:
                    # Üst Bilgiler
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL", delta="Canlı" if res['Is_Live'] else "Gecikmeli")
                    k2.metric("Skor", f"{res['Skor']}/100")
                    k3.metric("Karar", res['Sinyal'])
                    temel = res['Temel']
                    k4.metric("Temel Görünüm", temel['Yorum'] if temel else "-")
                    
                    st.divider()
                    
                    # Grafik ve Detaylar
                    col_g, col_d = st.columns([2, 1])
                    with col_g:
                        st.subheader(f"📊 {secilen_hisse} Teknik Grafik")
                        df = res['Data']
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
                        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name='Bollinger Üst', visible='legendonly'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], line=dict(color='gray', width=1, dash='dot'), name='Bollinger Alt', visible='legendonly'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=2), name='VWAP'))
                        
                        # Ichimoku
                        fig.add_trace(go.Scatter(x=df.index, y=df['ISA_9'], line=dict(color='green', width=1), name='Senkou A', visible='legendonly'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['ISB_26'], line=dict(color='red', width=1), name='Senkou B', visible='legendonly'))
                        
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, title=f"Veri: {res['Tarih']}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_d:
                        st.subheader("🧠 Yapay Zeka Raporu")
                        if res['Renk'] == 'green': st.success(f"**{res['Sinyal']}**")
                        else: st.warning(f"**{res['Sinyal']}**")
                        
                        st.write("#### 🎯 Hedefler")
                        st.info(f"Giriş: {res['Fiyat']:.2f}")
                        st.error(f"Stop: {res['Stop']:.2f}")
                        st.success(f"Hedef: {res['Hedef']:.2f}")
                        
                        if temel:
                            st.write("#### 🏢 Temel Veriler")
                            st.write(f"**F/K:** {temel['FK']}")
                            st.write(f"**PD/DD:** {temel['PD_DD']}")
                        
                        st.write("#### 📝 Nedenleri")
                        for y in res['Yorumlar']: st.markdown(f"✅ {y}")

                    # Haberler
                    st.markdown("---")
                    st.subheader("📰 İlgili Haberler")
                    if res['Haberler']:
                        for n in res['Haberler']:
                            st.markdown(f"🔹 **[{n['Title']}]({n['Link']})** - *{n['Date']}*")
                    else: st.info("Önemli haber akışı yok.")

                else: st.error("Veri alınamadı.")

    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        st.info(f"Takipteki Hisse Sayısı: {len(tum_hisseler)}")
        
        if st.button("TÜM BORSAYI TARA (Turbo) 🚀", type="primary"):
            st.warning("Tarama başladı (Bu işlem sadece teknik tarama yapar)...")
            
            # Batch analiz
            all_results = []
            chunk_size = 50 
            chunks = [tum_hisseler[i:i + chunk_size] for i in range(0, len(tum_hisseler), chunk_size)]
            
            bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                batch_res = engine.analyze_batch_fast(chunk)
                all_results.extend(batch_res)
                bar.progress((i + 1) / len(chunks))
                time.sleep(1)
            
            bar.empty()
            
            if all_results:
                df = pd.DataFrame(all_results)
                st.success(f"Tarama Bitti! {len(df)} Fırsat Bulundu.")
                st.dataframe(
                    df,
                    column_config={
                        "Hisse": st.column_config.TextColumn("Hisse"),
                        "Fiyat": st.column_config.NumberColumn("Fiyat", format="%.2f TL"),
                        "Sinyal": st.column_config.TextColumn("Karar"),
                        "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                        "Skor": st.column_config.ProgressColumn("Güven", format="%d", min_value=0, max_value=100),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("Sinyal yok.")

if __name__ == "__main__":
    main()
