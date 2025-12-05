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
import os

# --- 1. AYARLAR ---
LOGO_INTERNET_LINKI = "https://raw.githubusercontent.com/kullaniciadi/proje/main/logo.png"
MEMORY_FILE = "ai_memory.csv"

st.set_page_config(
    page_title="MERTT AI - Galactic Terminal", 
    layout="wide", 
    page_icon="🦅"  
)

# Hafıza Dosyası
if not os.path.exists(MEMORY_FILE):
    df_mem = pd.DataFrame(columns=["RSI", "MACD_Diff", "VWAP_Diff", "News_Score", "Outcome"])
    df_mem.to_csv(MEMORY_FILE, index=False)

# --- UZAY ÜSSÜ ANİMASYONU (CSS) ---
def add_space_theme():
    st.markdown(
        """
        <style>
        /* Arka Plan - Uzay Efekti */
        .stApp {
            background-color: #000000;
            background-image: 
                radial-gradient(white, rgba(255,255,255,.2) 2px, transparent 3px),
                radial-gradient(white, rgba(255,255,255,.15) 1px, transparent 2px),
                radial-gradient(white, rgba(255,255,255,.1) 2px, transparent 3px);
            background-size: 550px 550px, 350px 350px, 250px 250px;
            background-position: 0 0, 40px 60px, 130px 270px;
            animation: star-move 200s linear infinite;
        }
        
        @keyframes star-move {
            from { background-position: 0 0, 40px 60px, 130px 270px; }
            to { background-position: -10000px 5000px, 40px 60px, 130px 270px; }
        }

        /* Giriş Kutusu - Neon */
        .login-box {
            border: 2px solid #00ffcc;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 0 20px #00ffcc;
            background: rgba(0, 0, 0, 0.8);
            text-align: center;
        }
        
        /* Başlık Efekti */
        .neon-text {
            color: #fff;
            text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 20px #00ffcc, 0 0 30px #00ffcc, 0 0 40px #00ffcc;
            font-family: 'Courier New', Courier, monospace;
            font-size: 30px;
            font-weight: bold;
        }

        /* Buton Tasarımı */
        .stButton>button {
            background: linear-gradient(45deg, #00ffcc, #0033cc);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px #00ffcc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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
    """
    components.html(f"<html><head>{pwa_html}</head></html>", height=0, width=0)
pwa_kodlari()

# --- GÜVENLİK DUVARI (GÖRSEL EFEKTLİ) ---
def guvenlik_kontrolu():
    if 'giris_yapildi' not in st.session_state: st.session_state['giris_yapildi'] = False
    if not st.session_state['giris_yapildi']:
        add_space_theme() # Uzay Temasını Yükle
        
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.markdown('<div class="login-box">', unsafe_allow_html=True)
            logo_goster()
            st.markdown('<p class="neon-text">SYSTEM ONLINE</p>', unsafe_allow_html=True)
            st.markdown("---")
            sifre = st.text_input("Kuantum Erişim Kodu:", type="password")
            if st.button("BAĞLANTIYI BAŞLAT 🚀", use_container_width=True):
                try:
                    if sifre == st.secrets["GIRIS_SIFRESI"]: 
                        with st.spinner("Biyometrik Doğrulama..."):
                            time.sleep(1)
                        with st.spinner("Uydu Bağlantısı Kuruluyor..."):
                            time.sleep(1)
                        st.session_state['giris_yapildi'] = True
                        st.rerun()
                    else: st.error("⛔ ERİŞİM REDDEDİLDİ!")
                except: st.error("Sistem Hatası: Şifre Tanımsız.")
            st.markdown('</div>', unsafe_allow_html=True)
        return False
    return True

if not guvenlik_kontrolu(): st.stop()

# --- YEDEK TAM LİSTE ---
def get_backup_list():
    return ["THYAO","ASELS","KCHOL","GARAN","AKBNK","SASA","SISE","EREGL","TUPRS","BIMAS","HEKTS","PETKM","ISCTR","SAHOL","FROTO","YKBNK","EKGYO","ODAS","KOZAL","KONTR","ASTOR","EUPWR","GUBRF","OYAKC","TCELL","TTKOM","ENKAI","VESTL","ARCLK","TOASO","PGSUS","TAVHL","MGROS","SOKM","AEFES","AGHOL","AHGAZ","AKFGY","AKSA","ALARK","ALFAS"]

# --- CANLI LİSTE MOTORU ---
@st.cache_data(ttl=600)
def get_live_tickers():
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
    if len(canli_liste) < 50: return sorted(list(set(get_backup_list())))
    return sorted(list(set(canli_liste)))

# --- CANLI FİYAT ---
def get_realtime_price(ticker):
    time.sleep(random.uniform(0.2, 0.5))
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker.replace('.IS','')}-detay/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(resp.content, "html.parser")
        p = soup.select_one('.price-arrow-down, .price-arrow-up, .text-2')
        if p: return float(p.text.strip().replace(',', '.'))
    except: return None

# --- GLOBAL & HABER ---
class GlobalIntel:
    def __init__(self):
        self.risk = ['savaş', 'kriz', 'çöküş', 'enflasyon', 'faiz', 'gerilim', 'yaptırım']
        self.tech = ['rekor', 'büyüme', 'onay', 'temettü', 'kar', 'anlaşma']

    def get_global_indices(self):
        indices = {"S&P 500": "^GSPC", "Altın": "GC=F", "Bitcoin": "BTC-USD", "Dolar": "TRY=X", "Petrol": "BZ=F"}
        data = {}
        try:
            df = yf.download(" ".join(indices.values()), period="5d", interval="15m", progress=False)['Close'].ffill().bfill()
            for name, symbol in indices.items():
                try:
                    price = df[symbol].iloc[-1]
                    prev = df[symbol].iloc[-2]
                    change = ((price - prev) / prev) * 100
                    fmt = "%.0f" if "Bitcoin" in name else "%.2f"
                    data[name] = {"Fiyat": price, "Degisim": change, "Fmt": fmt}
                except: data[name] = {"Fiyat": 0.0, "Degisim": 0.0, "Fmt": "%.2f"}
        except: pass
        return data

    def analyze_news(self, query_type="GENEL", ticker=""):
        sentiment = 0; news_display = []
        urls = [f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr"] if query_type == "HISSE" else ["https://news.google.com/rss/search?q=Borsa+Gündem&hl=tr&gl=TR&ceid=TR:tr"]
        for url in urls:
            try:
                feed = feedparser.parse(requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5).content)
                for entry in feed.entries[:10]: 
                    title = entry.title.replace(" - Haberler", ""); link = entry.link
                    try:
                        dt = datetime(*entry.published_parsed[:6])
                        if (datetime.now() - dt).days <= 7: 
                            d_str = dt.strftime("%H:%M"); t_lower = title.lower(); delta = 0
                            for w in self.tech: 
                                if w in t_lower: delta += 2
                            for w in self.risk: 
                                if w in t_lower: delta -= 3
                            sentiment += delta
                            if (datetime.now() - dt).days < 1:
                                col = "green" if delta > 0 else "red" if delta < 0 else "gray"
                                news_display.append({"Title": title, "Link": link, "Date": d_str, "Color": col})
                    except: pass
            except: pass
        unique = []; seen = set()
        for n in news_display:
            if n['Title'] not in seen: unique.append(n); seen.add(n['Title'])
        return max(-20, min(20, sentiment)), unique[:10]

# --- ANALİZ MOTORU (TAM TEÇHİZAT) ---
class TradingEngine:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        self.intel = GlobalIntel()
        try:
            self.memory = pd.read_csv(MEMORY_FILE)
            if len(self.memory) > 10:
                self.model.fit(self.memory.drop("Outcome", axis=1), self.memory["Outcome"])
                self.is_trained = True
            else: self.is_trained = False
        except: self.is_trained = False

    def save_feedback(self, features, is_correct):
        new_row = features.copy()
        new_row['Outcome'] = 1 if is_correct else 0
        pd.DataFrame([new_row]).to_csv(MEMORY_FILE, mode='a', header=False, index=False)

    def get_fundamentals(self, ticker):
        try:
            info = yf.Ticker(f"{ticker}.IS").info
            fk = info.get('trailingPE'); pddd = info.get('priceToBook')
            score = 0
            if fk and fk < 10: score += 10
            if pddd and pddd < 2: score += 5
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-", "Score": score}
        except: return {"FK": "-", "PD_DD": "-", "Score": 0}

    def calculate_fibonacci(self, df):
        try:
            high = df['High'].tail(100).max()
            low = df['Low'].tail(100).min()
            diff = high - low
            return {"0.618": high - 0.618*diff, "Score": 15}
        except: return {"Score": 0}

    def analyze(self, ticker):
        try:
            t = f"{ticker}.IS"
            # 1 YILLIK VERİ (EMA 200 İÇİN)
            df = yf.download(t, period="1y", interval="60m", progress=False)
            if df is None or len(df) < 200: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            df = df.ffill().bfill()
            
            live = get_realtime_price(ticker)
            if live and abs(live - df.iloc[-1]['Close'])/df.iloc[-1]['Close'] < 0.2:
                df.iloc[-1, df.columns.get_loc('Close')] = live

            # --- TÜM İNDİKATÖRLER ---
            df['RSI'] = ta.rsi(df['Close'], 14)
            df['EMA_9'] = ta.ema(df['Close'], 9)
            df['EMA_200'] = ta.ema(df['Close'], 200)
            df = pd.concat([df, ta.macd(df['Close'])], axis=1)
            
            bb = ta.bbands(df['Close'], 20)
            if bb is not None: df = pd.concat([df, bb], axis=1)
            
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            if kc is not None: df = pd.concat([df, kc], axis=1)
            
            df = pd.concat([df, ta.ichimoku(df['High'], df['Low'], df['Close'])[0]], axis=1)
            df = pd.concat([df, ta.psar(df['High'], df['Low'], df['Close'])], axis=1)
            
            df['VWAP'] = (df['Volume']*(df['High']+df['Low']+df['Close'])/3).cumsum()/df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            df['OBV'] = ta.obv(df['Close'], df['Volume'])

            last = df.iloc[-1]
            if pd.isna(last['RSI']): return None

            # --- PUANLAMA (HER ŞEY DAHİL) ---
            score = 50
            reasons = []
            
            # 1. Trend (EMA)
            if last['Close'] > last['EMA_200']: score += 10; reasons.append("Fiyat EMA 200 Üstünde")
            else: score -= 20
            
            # 2. Ichimoku
            if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']: score += 10; reasons.append("Ichimoku Bulut Üstü")

            # 3. Sıkışma (Bollinger < Keltner)
            try:
                bbu = df[[c for c in df.columns if c.startswith('BBU')][0]].iloc[-1]
                kcu = df[[c for c in df.columns if c.startswith('KCU')][0]].iloc[-1]
                if bbu < kcu: score += 10; reasons.append("Sıkışma (Patlama Yakın)")
            except: pass

            # 4. Fibonacci
            fib = self.calculate_fibonacci(df)
            if fib and abs(last['Close'] - fib['0.618'])/fib['0.618'] < 0.01:
                 score += 15; reasons.append("Fibonacci 0.618 Desteği")

            # 5. Klasikler
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 15; reasons.append("MACD Al")
            if last['Close'] > last['VWAP']: score += 10
            if last['RSI'] < 30: score += 20
            
            psar_col = next((c for c in df.columns if c.startswith('PSAR')), None)
            if psar_col and df[psar_col].iloc[-1] < last['Close']: score += 10

            # 6. Temel ve Haber
            fund = self.get_fundamentals(ticker)
            score += fund['Score']
            n_sc, n_lst = self.intel.analyze_news("HISSE", ticker)
            score += n_sc
            if n_sc > 0: reasons.append("Haber Akışı Pozitif")

            # AI
            features = {"RSI": last['RSI'], "MACD_Diff": last['MACD_12_26_9']-last['MACDs_12_26_9'], "VWAP_Diff": (last['Close']-last['VWAP'])/last['VWAP'], "News_Score": n_sc}
            ai_conf = 0
            if self.is_trained:
                ai_prob = self.model.predict_proba(pd.DataFrame([features]))[0][1] * 100
                ai_conf = ai_prob
                if ai_prob > 70: score += 10
            
            score = max(0, min(100, score))
            signal, color = "NÖTR", "gray"
            if score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif score >= 60: signal, color = "AL 🌱", "blue"
            elif score <= 30: signal, color = "SAT 🔻", "red"

            stop = last['Close']-(last['ATR']*1.5)
            hedef = last['Close']+(last['ATR']*3) if "AL" in signal else last['Close']-(last['ATR']*3)

            return {
                "Hisse": ticker, "Fiyat": last['Close'], "Skor": int(score),
                "Sinyal": signal, "Renk": color, "RSI": last['RSI'],
                "Stop": stop, "Hedef": hedef, "Yorumlar": reasons, "Haberler": n_lst, 
                "Data": df, "Features": features, "AI_Conf": ai_conf,
                "Temel": fund, "Tarih": df.index[-1].strftime('%d.%m %H:%M')
            }
        except: return None

    def analyze_batch(self, tickers):
        res = []
        try:
            # Batch için 6 ay yeterli (EMA 200'ü hesaplamak için sınırda ama çalışır)
            d = yf.download([f"{t}.IS" for t in tickers], period="6mo", interval="60m", group_by='ticker', progress=False, threads=True)
            for t in tickers:
                try:
                    df = d[f"{t}.IS"].dropna()
                    if len(df)<100: continue
                    rsi = ta.rsi(df['Close'], 14).iloc[-1]
                    last = df['Close'].iloc[-1]
                    vwap = (df['Volume']*(df['High']+df['Low']+df['Close'])/3).cumsum()/df['Volume'].cumsum()
                    ema200 = ta.ema(df['Close'], 200).iloc[-1]
                    
                    sc = 50
                    if last > ema200: sc += 15 # Trend Filtresi
                    if rsi < 40 and last > vwap.iloc[-1]: sc += 20
                    
                    if sc>=80: res.append({"Hisse":t, "Fiyat":last, "Sinyal":"GÜÇLÜ AL 🚀", "RSI":rsi, "Skor":sc})
                except: continue
        except: pass
        return res

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.divider()
        menu = st.radio("Panel", ["💬 Öğrenen Analiz", "📡 Piyasa Radarı", "🌍 Global", "Çıkış"])
        if menu == "Çıkış": st.session_state['giris_yapildi'] = False; st.rerun()

    engine = TradingEngine()
    intel = GlobalIntel()
    tum_hisseler = get_live_tickers()

    if menu == "💬 Öğrenen Analiz":
        st.title("🧠 Kendi Kendine Öğrenen Analist")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu:", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("ANALİZ ET", type="primary")

        if btn and sembol:
            with st.spinner("Analiz ediliyor..."):
                res = engine.analyze(sembol)
                if res: st.session_state['last_res'] = res

        if 'last_res' in st.session_state:
            res = st.session_state['last_res']
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL")
            k2.metric("Skor", f"{res['Skor']}/100")
            k3.metric("AI Deneyimi", f"%{res['AI_Conf']:.0f}" if res['AI_Conf']>0 else "-")
            fk = res['Temel']['FK'] if res['Temel'] else "-"
            k4.metric("F/K", fk)
            
            st.divider()
            g1, g2 = st.columns([2, 1])
            with g1:
                df = res['Data']
                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat")])
                
                # EMA 200 ÇİZ
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='blue', width=2), name='EMA 200'))
                
                try:
                    bbu = next((c for c in df.columns if c.startswith('BBU')), None)
                    if bbu: fig.add_trace(go.Scatter(x=df.index, y=df[bbu], line=dict(color='gray', dash='dot'), name='Bollinger'))
                    kcu = next((c for c in df.columns if c.startswith('KCU')), None)
                    if kcu: fig.add_trace(go.Scatter(x=df.index, y=df[kcu], line=dict(color='purple'), name='Keltner'))
                    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange'), name='VWAP'))
                except: pass

                fig.update_layout(template="plotly_dark", height=500, title=f"Son Veri: {res['Tarih']}")
                st.plotly_chart(fig, use_container_width=True)
            
            with g2:
                if res['Renk']=='green': st.success(f"**{res['Sinyal']}**")
                else: st.warning(f"**{res['Sinyal']}**")
                st.info(f"Hedef: {res['Hedef']:.2f}")
                st.error(f"Stop: {res['Stop']:.2f}")
                st.write("#### 📝 Nedenler")
                for y in res['Yorumlar']: st.markdown(f"✅ {y}")
                
                st.caption("Bu analizi öğret:")
                cb1, cb2 = st.columns(2)
                if cb1.button("👍 Doğru"):
                    engine.save_feedback(res['Features'], True); st.toast("Öğrendim!")
                if cb2.button("👎 Yanlış"):
                    engine.save_feedback(res['Features'], False); st.toast("Düzelttim!")

            if res['Haberler']:
                st.markdown("### 📰 Günün Haberleri")
                for n in res['Haberler']:
                    col = "🟢" if n['Color']=="green" else "🔴" if n['Color']=="red" else "⚪"
                    st.markdown(f"{col} **[{n['Title']}]({n['Link']})** ({n['Date']})")

    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        if st.button("TÜM BORSAYI TARA 🚀", type="primary"):
            chunks = [tum_hisseler[i:i+50] for i in range(0, len(tum_hisseler), 50)]
            res = []
            bar = st.progress(0)
            for i, c in enumerate(chunks):
                res.extend(engine.analyze_batch(c))
                bar.progress((i+1)/len(chunks))
                time.sleep(1)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res).style.background_gradient(subset=['Skor'], cmap='RdYlGn'))
            else: st.warning("Sinyal yok")

    elif menu == "🌍 Global":
        st.title("🌍 Piyasalar")
        idx = intel.get_global_indices()
        c = st.columns(len(idx))
        for i, (n, d) in enumerate(idx.items()): c[i].metric(n, f"{d['Fiyat']:.2f}", f"%{d['Degisim']:.2f}")
        st.divider()
        _, nws = intel.analyze_news("GENEL")
        for n in nws: st.markdown(f"#### 📰 [{n['Title']}]({n['Link']})")

if __name__ == "__main__":
    main()
