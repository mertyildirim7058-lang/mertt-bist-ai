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

# --- 1. AYARLAR ---
LOGO_INTERNET_LINKI = "https://raw.githubusercontent.com/kullaniciadi/proje/main/logo.png"

st.set_page_config(
    page_title="MERTT AI Terminal", 
    layout="wide", 
    page_icon="🦅"  
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

# --- LİSTE VE FİYAT MOTORLARI ---
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

def get_realtime_price(ticker):
    time.sleep(random.uniform(0.5, 1.0))
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker.replace('.IS','')}-detay/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(resp.content, "html.parser")
        price_span = soup.find("span", {"class": "text-2"})
        if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
        if price_span: return float(price_span.text.strip().replace(',', '.'))
    except: return None

# --- GELİŞMİŞ HABER İSTİHBARAT MOTORU ---
class GlobalIntel:
    def __init__(self):
        self.risk_keywords = ['savaş', 'kriz', 'çöküş', 'enflasyon', 'faiz', 'gerilim', 'yaptırım', 'ceza', 'satış', 'zarar', 'düşüş']
        self.tech_keywords = ['yapay zeka', 'rekor', 'büyüme', 'anlaşma', 'onay', 'ihracat', 'yatırım', 'temettü', 'kar', 'bedelsiz', 'geri alım']
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    def get_global_indices(self):
        indices = {"S&P 500": "^GSPC", "Altın": "GC=F", "Bitcoin": "BTC-USD", "Dolar": "TRY=X", "Petrol": "BZ=F"}
        data = {}
        try:
            tickers = " ".join(indices.values())
            df = yf.download(tickers, period="5d", interval="15m", progress=False)['Close'].ffill().bfill()
            for name, symbol in indices.items():
                try:
                    price = df[symbol].dropna().iloc[-1]
                    prev = df[symbol].dropna().iloc[-2]
                    change = ((price - prev) / prev) * 100
                    fmt = "%.0f" if "Bitcoin" in name else "%.2f"
                    data[name] = {"Fiyat": price, "Degisim": change, "Fmt": fmt}
                except: data[name] = {"Fiyat": 0.0, "Degisim": 0.0, "Fmt": "%.2f"}
        except: pass
        return data

    def analyze_news(self, query_type="GENEL", ticker=""):
        """
        Haberleri çeker, analiz eder ve listeler.
        """
        news_list = []
        if query_type == "HISSE":
            # Hisse özelinde daha geniş arama
            feeds = [
                f"https://news.google.com/rss/search?q={ticker}+hisse+kap+haberleri&hl=tr&gl=TR&ceid=TR:tr",
                f"https://news.google.com/rss/search?q={ticker}+borsa+yorum&hl=tr&gl=TR&ceid=TR:tr"
            ]
        else:
            feeds = ["https://news.google.com/rss/search?q=Borsa+İstanbul+Ekonomi&hl=tr&gl=TR&ceid=TR:tr"]
            
        sentiment_score = 0
        
        for url in feeds:
            try:
                # Requests ile çekip parse ediyoruz (Engel aşmak için)
                r = requests.get(url, headers=self.headers, timeout=5)
                if r.status_code == 200:
                    feed = feedparser.parse(r.content)
                    
                    for entry in feed.entries[:5]: # Her kaynaktan 5 haber
                        title = entry.title.replace(" - Haberler", "")
                        
                        # Tarih formatı
                        try:
                            dt = datetime(*entry.published_parsed[:6])
                            date_str = dt.strftime("%d.%m %H:%M")
                        except: date_str = "Yeni"

                        # Duygu Analizi
                        t_lower = title.lower()
                        impact = "Nötr"
                        color = "gray"
                        
                        score_delta = 0
                        for w in self.tech_keywords: 
                            if w in t_lower: 
                                score_delta += 2
                                impact = "Pozitif"
                                color = "green"
                        for w in self.risk_keywords: 
                            if w in t_lower: 
                                score_delta -= 3
                                impact = "Negatif"
                                color = "red"
                        
                        sentiment_score += score_delta
                        
                        news_list.append({
                            "Title": title, 
                            "Link": entry.link, 
                            "Date": date_str,
                            "Impact": impact,
                            "Color": color
                        })
            except: pass
            
        # Tekrarlanan haberleri temizle
        seen = set()
        unique_news = []
        for n in news_list:
            if n['Title'] not in seen:
                unique_news.append(n)
                seen.add(n['Title'])
                
        return max(-20, min(20, sentiment_score)), unique_news[:10]

# --- ANALİZ MOTORU ---
class TradingEngine:
    def __init__(self):
        try: from sklearn.preprocessing import StandardScaler
        except: pass
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        self.intel = GlobalIntel()

    def analyze(self, ticker, mode="PRO"):
        try:
            t = f"{ticker}.IS"
            df = yf.download(t, period="6mo", interval="60m", progress=False)
            if df is None or len(df) < 100: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Istanbul')
            df = df.ffill().bfill()

            is_live = False
            # Canlı Yama (Sadece Tekli Sorguda)
            if mode == "PRO":
                live_price = get_realtime_price(ticker)
                if live_price and live_price > 0:
                    if abs(live_price - df.iloc[-1]['Close']) / df.iloc[-1]['Close'] < 0.20:
                        df.iloc[-1, df.columns.get_loc('Close')] = live_price
                        df.iloc[-1, df.columns.get_loc('High')] = max(live_price, df.iloc[-1]['High'])
                        df.iloc[-1, df.columns.get_loc('Low')] = min(live_price, df.iloc[-1]['Low'])
                        is_live = True

            # İndikatörler
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            
            bb = ta.bbands(df['Close'], length=20)
            if bb is not None: df = pd.concat([df, bb], axis=1)
            
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
            df = pd.concat([df, ichimoku], axis=1)
            
            psar = ta.psar(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, psar], axis=1)
            psar_col = next((c for c in df.columns if c.startswith('PSAR')), None)

            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            last = df.iloc[-1]
            if pd.isna(last['RSI']): return None

            # --- PUANLAMA ---
            score = 50
            reasons = []

            if last['Close'] > last['VWAP']: score += 10; reasons.append("Fiyat VWAP Üzerinde")
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 15; reasons.append("MACD Al Sinyali")
            if psar_col and df[psar_col].iloc[-1] < last['Close']: score += 10; reasons.append("PSAR Yükseliş")
            if last['RSI'] < 30: score += 20; reasons.append("RSI Aşırı Satım (Fırsat)")
            elif last['RSI'] > 70: score -= 15; reasons.append("RSI Aşırı Alım")
            
            span_a = df['ISA_9'].iloc[-1]
            span_b = df['ISB_26'].iloc[-1]
            if last['Close'] > span_a and last['Close'] > span_b: score += 15; reasons.append("Ichimoku Bulutu Üstünde")

            # Haber Analizi (Sadece Tekli Modda)
            news_data = []
            if mode == "PRO":
                news_score, news_list = self.intel.analyze_news("HISSE", ticker)
                score += news_score
                news_data = news_list
                if news_score > 0: reasons.append(f"Haber Akışı Pozitif ({news_score} Puan)")
                elif news_score < 0: reasons.append(f"Haber Akışı Negatif ({news_score} Puan)")
            
            score = max(0, min(100, score))
            
            signal, color = "NÖTR / İZLE", "gray"
            if score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif score >= 60: signal, color = "AL 🌱", "blue"
            elif score <= 30: signal, color = "SAT 🔻", "red"

            stop = last['Close'] - (last['ATR'] * 1.5)
            hedef = last['Close'] + (last['ATR'] * 3.0)
            
            # Temel Analiz (Basit)
            try:
                stock = yf.Ticker(f"{ticker}.IS")
                info = stock.info
                fk = info.get('trailingPE', '-')
                pddd = info.get('priceToBook', '-')
            except: fk, pddd = "-", "-"

            return {
                "Hisse": ticker, "Fiyat": last['Close'], "Skor": int(score),
                "Sinyal": signal, "Renk": color, "RSI": last['RSI'],
                "Stop": stop, "Hedef": hedef, "Yorumlar": reasons, 
                "Data": df, "Tarih": df.index[-1].strftime('%d %B %H:%M'),
                "Is_Live": is_live, "Temel": {"FK": fk, "PD_DD": pddd}, "Haberler": news_data
            }
        except: return None

    def analyze_batch(self, tickers_list):
        # Batch analizde sadece teknik kullanılır (Hız için)
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
                    
                    score = 50
                    if rsi.iloc[-1] < 45 and last_close > vwap.iloc[-1]: score = 85
                    elif rsi.iloc[-1] > 70: score = 20
                    
                    signal = "NÖTR"
                    if score >= 80: signal = "GÜÇLÜ AL"
                    elif score <= 30: signal = "SAT"
                    
                    if signal != "NÖTR":
                        results.append({"Hisse": ticker, "Fiyat": last_close, "Sinyal": signal, "RSI": rsi.iloc[-1], "Skor": score})
                except: continue
        except: pass
        return results

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.markdown("<h3 style='text-align: center;'>Yapay Zeka Üssü</h3>", unsafe_allow_html=True)
        st.divider()
        menu = st.radio("Panel", ["💬 Hisse Sor / Analiz", "📡 Piyasa Radarı", "🌍 Global & Haber Odası", "Çıkış"])
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    engine = TradingEngine()
    intel = GlobalIntel()
    tum_hisseler = get_live_tickers()

    # --- 1. KISIM: MANUEL SORGU (YENİ HABER ÖZELLİKLİ) ---
    if menu == "💬 Hisse Sor / Analiz":
        st.title("💬 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu (Örn: THYAO):", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("ANALİZ ET 🔍", type="primary")

        if btn and sembol:
            with st.spinner(f"{sembol} için haberler taranıyor ve teknik analiz yapılıyor..."):
                res = engine.analyze(sembol, mode="PRO")
                
                if res:
                    # Üst Panel
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL", delta="Canlı" if res['Is_Live'] else "Gecikmeli")
                    k2.metric("Skor", f"{res['Skor']}/100")
                    k3.metric("Karar", res['Sinyal'])
                    temel = res['Temel']
                    k4.metric("F/K", f"{temel['FK']}")
                    
                    st.divider()
                    
                    col_g, col_d = st.columns([2, 1])
                    with col_g:
                        st.subheader(f"📊 {sembol} Teknik Grafik")
                        df = res['Data']
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
                        bbu = next((c for c in df.columns if c.startswith('BBU')), None)
                        bbl = next((c for c in df.columns if c.startswith('BBL')), None)
                        if bbu and bbl:
                            fig.add_trace(go.Scatter(x=df.index, y=df[bbu], line=dict(color='gray', width=1, dash='dot'), name='Bollinger', visible='legendonly'))
                            fig.add_trace(go.Scatter(x=df.index, y=df[bbl], line=dict(color='gray', width=1, dash='dot'), name='Bollinger', visible='legendonly'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=2), name='VWAP'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['ISA_9'], line=dict(color='green', width=1), name='Ichimoku A', visible='legendonly'))
                        psar_col = next((c for c in df.columns if c.startswith('PSAR')), None)
                        if psar_col:
                            fig.add_trace(go.Scatter(x=df.index, y=df[psar_col], mode='markers', marker=dict(color='yellow', size=4), name='PSAR'))
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, title=f"Veri Zamanı: {res['Tarih']}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_d:
                        st.subheader("🧠 Yapay Zeka Raporu")
                        if res['Renk'] == 'green': st.success(f"**{res['Sinyal']}**")
                        elif res['Renk'] == 'red': st.error(f"**{res['Sinyal']}**")
                        else: st.warning(f"**{res['Sinyal']}**")
                        st.info(f"Hedef: {res['Hedef']:.2f}")
                        st.error(f"Stop: {res['Stop']:.2f}")
                        st.write("#### 📝 Nedenleri")
                        for y in res['Yorumlar']: st.markdown(f"✅ {y}")

                    # --- YENİ HABER BÖLÜMÜ ---
                    st.markdown("---")
                    st.subheader(f"📰 {sembol} İçin Son Haberler & Duygu Analizi")
                    
                    if res['Haberler']:
                        for news in res['Haberler']:
                            # Haberin etkisine göre renk
                            if news['Impact'] == "Pozitif":
                                st.success(f"🟢 **{news['Title']}** ({news['Date']})")
                            elif news['Impact'] == "Negatif":
                                st.error(f"🔴 **{news['Title']}** ({news['Date']})")
                            else:
                                st.info(f"⚪ **{news['Title']}** ({news['Date']})")
                            st.markdown(f"[Haberi Oku]({news['Link']})")
                    else:
                        st.warning("Bu hisse ile ilgili son 24 saatte kritik bir haber bulunamadı.")

                else: st.error("Hisse bulunamadı veya veri yok.")

    # --- 2. KISIM: RADAR ---
    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        if not tum_hisseler: st.error("Liste çekilemedi."); st.stop()
        st.info(f"Takipteki Hisse Sayısı: {len(tum_hisseler)}")
        if st.button("TÜM BORSAYI TARA (Turbo) 🚀", type="primary"):
            all_results = []
            chunk_size = 50 
            chunks = [tum_hisseler[i:i + chunk_size] for i in range(0, len(tum_hisseler), chunk_size)]
            bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                batch_res = engine.analyze_batch(chunk)
                all_results.extend(batch_res)
                bar.progress((i + 1) / len(chunks))
                time.sleep(1)
            bar.empty()
            if all_results:
                df = pd.DataFrame(all_results)
                st.success(f"{len(df)} Fırsat Bulundu!")
                st.dataframe(df.style.format({"Fiyat": "{:.2f}", "RSI": "{:.0f}"}).background_gradient(subset=['Skor'], cmap='RdYlGn'), use_container_width=True)
            else: st.warning("Sinyal yok.")

    # --- 3. KISIM: GLOBAL ---
    elif menu == "🌍 Global & Haber Odası":
        st.title("🌍 Dünya Piyasaları & Gündem")
        indices = intel.get_global_indices()
        if indices:
            cols = st.columns(len(indices))
            for i, (name, data) in enumerate(indices.items()):
                cols[i].metric(label=name, value=f"{data['Fiyat']:.2f}", delta=f"%{data['Degisim']:.2f}")
        st.divider()
        st.markdown("### 🇹🇷 Türkiye & Ekonomi Gündemi")
        _, news_list = intel.analyze_news("GENEL") 
        if news_list:
            for n in news_list:
                st.markdown(f"#### 📰 [{n['Title']}]({n['Link']})")
                st.caption(f"🗓️ {n['Date']}")
                st.write("---")
        else: st.info("Haber akışı alınamadı.")

if __name__ == "__main__":
    main()
