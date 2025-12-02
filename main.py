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
from datetime import datetime, timedelta

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
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
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

# --- YEDEK LİSTE (KURTARICI) ---
def get_backup_list():
    return [
        "THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS", 
        "HEKTS", "PETKM", "ISCTR", "SAHOL", "FROTO", "YKBNK", "EKGYO", "ODAS", "KOZAL", "KONTR", 
        "ASTOR", "EUPWR", "GUBRF", "OYAKC", "TCELL", "TTKOM", "ENKAI", "VESTL", "ARCLK", "TOASO",
        "PGSUS", "TAVHL", "MGROS", "SOKM", "AEFES", "AGHOL", "AHGAZ", "AKFGY", "AKSA", "AKSEN",
        "ALARK", "ALBRK", "ALFAS", "ANSGR", "ARASE", "BERA", "BIOEN", "BOBET", "BRSAN", "BRYAT"
    ]

# --- 3 KATMANLI LİSTE ÇEKİCİ ---
@st.cache_data(ttl=600)
def get_live_tickers():
    canli_liste = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
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
    
    if len(canli_liste) > 50: return sorted(list(set(canli_liste)))
    else: return sorted(list(set(get_backup_list())))

# --- 2. CANLI FİYAT (SNIPER) ---
def get_realtime_price(ticker):
    # Ban yememek için rastgele bekleme
    time.sleep(random.uniform(0.2, 0.8)) 
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker.replace('.IS','')}-detay/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(resp.content, "html.parser")
        price_span = soup.find("span", {"class": "text-2"})
        if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
        if price_span: return float(price_span.text.strip().replace(',', '.'))
    except: return None

# --- 3. HABER MOTORU (CACHE YOK - SÜREKLİ GÜNCEL) ---
class GlobalIntel:
    def __init__(self):
        self.risk_keywords = ['savaş', 'kriz', 'çöküş', 'enflasyon', 'faiz', 'gerilim', 'yaptırım', 'ceza', 'zarar', 'satış']
        self.tech_keywords = ['yapay zeka', 'rekor', 'büyüme', 'anlaşma', 'onay', 'ihracat', 'yatırım', 'temettü', 'kar', 'bedelsiz']
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def get_global_indices(self):
        indices = {"S&P 500": "^GSPC", "Altın": "GC=F", "Bitcoin": "BTC-USD", "Dolar": "TRY=X", "Petrol": "BZ=F"}
        data = {}
        try:
            df = yf.download(" ".join(indices.values()), period="5d", interval="15m", progress=False)['Close'].ffill().bfill()
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

    # CACHE YOK - HER SEFERİNDE TAZE HABER ÇEKER
    def analyze_news(self, query_type="GENEL", ticker=""):
        sentiment = 0
        news_display = []
        
        if query_type == "HISSE":
            # Hisse özelinde daha çok kaynak
            feeds = [
                f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr",
                f"https://news.google.com/rss/search?q={ticker}+borsa&hl=tr&gl=TR&ceid=TR:tr"
            ]
        else:
            feeds = [
                "https://news.google.com/rss/search?q=Borsa+İstanbul+Son+Dakika&hl=tr&gl=TR&ceid=TR:tr",
                "https://www.trthaber.com/xml/ekonomi.xml"
            ]
            
        for url in feeds:
            try:
                r = requests.get(url, headers=self.headers, timeout=5)
                if r.status_code == 200:
                    feed = feedparser.parse(r.content)
                    for entry in feed.entries[:8]: 
                        title = entry.title.replace(" - Haberler", "")
                        link = entry.link
                        try:
                            # Sadece Son 24 Saati Al (Daha Güncel)
                            if hasattr(entry, 'published_parsed'):
                                news_date = datetime(*entry.published_parsed[:6])
                                today = datetime.now()
                                if (today - news_date).days <= 2: # Son 2 gün
                                    date_str = news_date.strftime("%d.%m %H:%M")
                                    
                                    # Puanlama
                                    t_lower = title.lower()
                                    impact = "Nötr"; color = "gray"; score_delta = 0
                                    for w in self.tech_keywords: 
                                        if w in t_lower: score_delta += 2; impact = "Pozitif"; color = "green"
                                    for w in self.risk_keywords: 
                                        if w in t_lower: score_delta -= 3; impact = "Negatif"; color = "red"
                                    
                                    sentiment += score_delta
                                    news_display.append({"Title": title, "Link": link, "Date": date_str, "Impact": impact, "Color": color})
                        except: pass
            except: pass
            
        unique = []
        seen = set()
        for n in news_display:
            if n['Title'] not in seen: unique.append(n); seen.add(n['Title'])
        return max(-20, min(20, sentiment)), unique[:10]

# --- 4. ANALİZ MOTORU ---
class TradingEngine:
    def __init__(self):
        try: from sklearn.preprocessing import StandardScaler
        except: pass
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        self.intel = GlobalIntel()

    def get_fundamentals(self, ticker):
        try:
            stock = yf.Ticker(f"{ticker}.IS")
            info = stock.info
            fk = info.get('trailingPE', None)
            pddd = info.get('priceToBook', None)
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-"}
        except: return None

    # --- FULL ANALİZ (Hem Manuel Hem Radar İçin) ---
    def analyze_pro(self, ticker):
        try:
            t = f"{ticker}.IS"
            df = yf.download(t, period="6mo", interval="60m", progress=False)
            if df is None or len(df) < 100: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Istanbul')
            df = df.ffill().bfill()

            is_live = False
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
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            if kc is not None: df = pd.concat([df, kc], axis=1)
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
            df = pd.concat([df, ichimoku], axis=1)
            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            last = df.iloc[-1]
            if pd.isna(last['RSI']): return None

            score = 50
            reasons = []
            
            # Teknik Puanlama
            if last['Close'] > last['VWAP']: score += 10; reasons.append("Fiyat VWAP Üzerinde")
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 15; reasons.append("MACD Al Sinyali")
            if last['RSI'] < 30: score += 20; reasons.append("RSI Aşırı Satım")
            elif last['RSI'] > 70: score -= 15; reasons.append("RSI Aşırı Alım")
            if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']: score += 10; reasons.append("Ichimoku Bulutu Üstünde")

            # Bollinger (Dinamik İsim Kontrolü)
            bbl = next((c for c in df.columns if c.startswith('BBL')), None)
            if bbl and last['Close'] <= last[bbl] * 1.01: score += 15; reasons.append("Bollinger Alt Bandı Teması")

            # --- HABER ANALİZİ (HER ANALİZDE TAZE ÇEKER) ---
            news_score, news_list = self.intel.analyze_news("HISSE", ticker)
            score += news_score
            if news_score > 0: reasons.append("Haber Akışı Pozitif")
            elif news_score < 0: reasons.append("Haber Akışı Negatif")
            
            score = max(0, min(100, score))
            signal, color = "NÖTR / İZLE", "gray"
            if score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif score >= 60: signal, color = "AL 🌱", "blue"
            elif score <= 30: signal, color = "SAT 🔻", "red"

            stop = last['Close'] - (last['ATR'] * 1.5)
            hedef = last['Close'] + (last['ATR'] * 3.0)
            temel = self.get_fundamentals(ticker)

            return {
                "Hisse": ticker, "Fiyat": last['Close'], "Skor": int(score),
                "Sinyal": signal, "Renk": color, "RSI": last['RSI'],
                "Stop": stop, "Hedef": hedef, "Yorumlar": reasons, 
                "Data": df, "Tarih": df.index[-1].strftime('%d %B %H:%M'),
                "Is_Live": is_live, "Temel": temel, "Haberler": news_list
            }
        except: return None

    # --- HIZLI ÖN FİLTRELEME (Radar İçin) ---
    def filter_candidates(self, tickers_list):
        candidates = []
        symbols = [f"{t}.IS" for t in tickers_list]
        try:
            data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
            for ticker in tickers_list:
                try:
                    try: df = data[f"{ticker}.IS"].copy()
                    except: continue
                    if df.empty: continue
                    df = df.dropna()
                    if len(df) < 30: continue
                    
                    rsi = ta.rsi(df['Close'], length=14).iloc[-1]
                    
                    # Sadece potansiyel olanları seç (RSI < 45 veya > 70)
                    if rsi < 45 or rsi > 70:
                        candidates.append(ticker)
                except: continue
        except: pass
        return candidates

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.divider()
        menu = st.radio("Panel", ["💬 Hisse Sor / Analiz", "📡 Piyasa Radarı", "🌍 Global & Haber Odası", "Çıkış"])
        if menu == "Çıkış": st.session_state['giris_yapildi'] = False; st.rerun()

    engine = TradingEngine()
    intel = GlobalIntel()
    tum_hisseler = get_live_tickers()

    if menu == "💬 Hisse Sor / Analiz":
        st.title("💬 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu:", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("ANALİZ ET 🔍", type="primary")

        if btn and sembol:
            with st.spinner(f"{sembol} için tüm veriler taranıyor (Canlı Fiyat + Haber + Teknik + Temel)..."):
                res = engine.analyze(sembol, mode="PRO")
                if res:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL", delta="Canlı" if res['Is_Live'] else "Gecikmeli")
                    k2.metric("Skor", f"{res['Skor']}/100")
                    k3.metric("Karar", res['Sinyal'])
                    temel = res['Temel']
                    k4.metric("Temel", f"F/K: {temel['FK']}" if temel else "-")
                    st.divider()
                    col_g, col_d = st.columns([2, 1])
                    with col_g:
                        st.subheader(f"📊 {sembol} Grafik")
                        df = res['Data']
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
                        bbu = next((c for c in df.columns if c.startswith('BBU')), None)
                        bbl = next((c for c in df.columns if c.startswith('BBL')), None)
                        if bbu: fig.add_trace(go.Scatter(x=df.index, y=df[bbu], line=dict(color='gray', width=1, dash='dot'), name='Bollinger', visible='legendonly'))
                        kcu = next((c for c in df.columns if c.startswith('KCU')), None)
                        if kcu: fig.add_trace(go.Scatter(x=df.index, y=df[kcu], line=dict(color='purple', width=1), name='Keltner', visible='legendonly'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange', width=2), name='VWAP'))
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, title=f"Veri Zamanı: {res['Tarih']}")
                        st.plotly_chart(fig, use_container_width=True)
                    with col_d:
                        st.subheader("🧠 Analiz Raporu")
                        if res['Renk'] == 'green': st.success(f"**{res['Sinyal']}**")
                        elif res['Renk'] == 'red': st.error(f"**{res['Sinyal']}**")
                        else: st.warning(f"**{res['Sinyal']}**")
                        st.info(f"Hedef: {res['Hedef']:.2f}")
                        st.error(f"Stop: {res['Stop']:.2f}")
                        st.write("#### 📝 Nedenler")
                        for y in res['Yorumlar']: st.markdown(f"✅ {y}")
                    st.markdown("---")
                    st.subheader(f"📰 {sembol} Haberleri")
                    if res['Haberler']:
                        for n in res['Haberler']:
                            color = "🟢" if n['Impact'] == "Pozitif" else "🔴" if n['Impact'] == "Negatif" else "⚪"
                            st.markdown(f"{color} **[{n['Title']}]({n['Link']})** - *{n['Date']}*")
                    else: st.info("Önemli haber yok.")
                else: st.error("Veri bulunamadı.")

    # --- RADAR (YENİ: DETAYLI KART GÖRÜNÜMÜ) ---
    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        if not tum_hisseler: st.error("Liste çekilemedi."); st.stop()
        st.info(f"Takipteki Hisse: {len(tum_hisseler)}")
        
        if st.button("TÜM BORSAYI TARA 🚀", type="primary"):
            status = st.empty()
            status.info("1. Aşama: Hızlı Teknik Tarama yapılıyor (Filtreleme)...")
            
            # 1. Hızlıca Adayları Bul (Batch Filter)
            adaylar = engine.filter_candidates(tum_hisseler)
            
            status.info(f"2. Aşama: {len(adaylar)} potansiyel hisse bulundu. Derin analiz (Haber+Canlı Fiyat) başlatılıyor...")
            bar = st.progress(0)
            
            results_found = 0
            
            # 2. Adayları Detaylı Analiz Et ve Kart Olarak Bas
            for i, ticker in enumerate(adaylar):
                res = engine.analyze(ticker, mode="PRO") # FULL Analiz Çağır
                
                if res and (res['Sinyal'] == "GÜÇLÜ AL 🚀" or res['Sinyal'] == "SAT 🔻"):
                    results_found += 1
                    # --- SONUÇ KARTI ---
                    with st.expander(f"{res['Sinyal']} | {res['Hisse']} - {res['Fiyat']:.2f} TL (Skor: {res['Skor']})", expanded=False):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sinyal", res['Sinyal'])
                        c2.metric("RSI", f"{res['RSI']:.0f}")
                        c3.metric("F/K", f"{res['Temel']['FK']}" if res['Temel'] else "-")
                        
                        st.write(f"**Neden:** {', '.join(res['Yorumlar'])}")
                        
                        if res['Haberler']:
                            st.caption("Son Haber:")
                            st.markdown(f"🔹 [{res['Haberler'][0]['Title']}]({res['Haberler'][0]['Link']})")
                            
                bar.progress((i + 1) / len(adaylar))
            
            bar.empty()
            status.success(f"Tarama Tamamlandı. {results_found} kritik fırsat listelendi.")

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
