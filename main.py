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
    page_title="MERTT AI Terminal", 
    layout="wide", 
    page_icon="🦅"  
)

# Hafıza Dosyasını Başlat
if not os.path.exists(MEMORY_FILE):
    df_mem = pd.DataFrame(columns=["RSI", "MACD_Diff", "VWAP_Diff", "News_Score", "Outcome"])
    df_mem.to_csv(MEMORY_FILE, index=False)

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
            st.markdown("<h4 style='text-align: center;'>Gelecek İçin Bilgi ve Teknoloji</h4>", unsafe_allow_html=True)
            st.divider()
            sifre = st.text_input("Erişim Anahtarı:", type="password")
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

# --- YEDEK LİSTE ---
def get_backup_list():
    return ["THYAO","ASELS","KCHOL","GARAN","AKBNK","SASA","SISE","EREGL","TUPRS","BIMAS","HEKTS","PETKM","ISCTR","SAHOL","FROTO","YKBNK","EKGYO","ODAS","KOZAL","KONTR","ASTOR","EUPWR","GUBRF","OYAKC","TCELL","TTKOM","ENKAI","VESTL","ARCLK","TOASO","PGSUS","TAVHL","MGROS","SOKM","AEFES","AGHOL","AHGAZ","AKFGY","AKSA","ALARK","ALFAS"]

# --- CANLI LİSTE MOTORU ---
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

# --- GLOBAL & HABER MOTORU ---
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
                except: data[name] = {"Fiyat": 0, "Degisim": 0, "Fmt": "%.2f"}
        except: pass
        return data

    def analyze_news(self, query_type="GENEL", ticker=""):
        sentiment = 0
        news_display = []
        
        if query_type == "HISSE":
            feeds = [f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr"]
        else:
            feeds = ["https://news.google.com/rss/search?q=Borsa+İstanbul+Gündem&hl=tr&gl=TR&ceid=TR:tr"]
            
        for url in feeds:
            try:
                feed = feedparser.parse(requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5).content)
                for entry in feed.entries[:10]: 
                    title = entry.title.replace(" - Haberler", "")
                    link = entry.link
                    try:
                        if hasattr(entry, 'published_parsed'):
                            dt = datetime(*entry.published_parsed[:6])
                            if (datetime.now() - dt).days <= 7: # 7 Günlük Hafıza
                                d_str = dt.strftime("%d.%m %H:%M")
                                t_lower = title.lower()
                                imp = "Nötr"; col = "gray"; sd = 0
                                for w in self.tech: 
                                    if w in t_lower: sd += 2; imp="Pozitif"; col="green"
                                for w in self.risk: 
                                    if w in t_lower: sd -= 3; imp="Negatif"; col="red"
                                
                                sentiment += sd
                                # Sadece Bugünü Göster
                                if (datetime.now() - dt).days < 1:
                                    news_display.append({"Title": title, "Link": link, "Date": d_str, "Color": col})
                    except: pass
            except: pass
            
        unique = []; seen = set()
        for n in news_display:
            if n['Title'] not in seen: unique.append(n); seen.add(n['Title'])
        return max(-20, min(20, sentiment)), unique[:10]

# --- ANALİZ MOTORU (ÖĞRENEN & FULL) ---
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
            stock = yf.Ticker(f"{ticker}.IS")
            info = stock.info
            fk = info.get('trailingPE', None)
            pddd = info.get('priceToBook', None)
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-"}
        except: return None

    def analyze(self, ticker, mode="PRO"):
        try:
            t = f"{ticker}.IS"
            df = yf.download(t, period="6mo", interval="60m", progress=False)
            if df is None or len(df) < 100: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Istanbul')
            df = df.ffill().bfill()

            # Canlı Yama
            if mode == "PRO":
                live = get_realtime_price(ticker)
                if live and abs(live - df.iloc[-1]['Close'])/df.iloc[-1]['Close'] < 0.2:
                    df.iloc[-1, df.columns.get_loc('Close')] = live

            # İndikatörler
            df['RSI'] = ta.rsi(df['Close'], 14)
            df = pd.concat([df, ta.macd(df['Close'])], axis=1)
            bb = ta.bbands(df['Close'], 20)
            if bb is not None: df = pd.concat([df, bb], axis=1)
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            if kc is not None: df = pd.concat([df, kc], axis=1)
            df = pd.concat([df, ta.ichimoku(df['High'], df['Low'], df['Close'])[0]], axis=1)
            df['VWAP'] = (df['Volume']*(df['High']+df['Low']+df['Close'])/3).cumsum()/df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)
            df['OBV'] = ta.obv(df['Close'], df['Volume'])

            last = df.iloc[-1]
            if pd.isna(last['RSI']): return None

            # --- PUANLAMA ---
            score = 50
            reasons = []
            
            if last['Close'] > last['VWAP']: score += 10
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 15; reasons.append("MACD Al")
            if last['RSI'] < 30: score += 20; reasons.append("RSI Dip")
            elif last['RSI'] > 70: score -= 15
            if last['Close'] > last['ISA_9']: score += 10; reasons.append("Ichimoku Trend")
            
            # Haber
            n_score, n_list = 0, []
            if mode == "PRO":
                n_score, n_list = self.intel.analyze_news("HISSE", ticker)
                score += n_score
                if n_score > 0: reasons.append("Haber+")

            # AI Müdahalesi
            features = {"RSI": last['RSI'], "MACD_Diff": last['MACD_12_26_9']-last['MACDs_12_26_9'], "VWAP_Diff": (last['Close']-last['VWAP'])/last['VWAP'], "News_Score": n_score}
            ai_conf = 0
            if self.is_trained:
                ai_prob = self.model.predict_proba(pd.DataFrame([features]))[0][1] * 100
                ai_conf = ai_prob
                if ai_prob > 70: score += 10; reasons.append(f"AI Onaylı (%{ai_prob:.0f})")
            
            score = max(0, min(100, score))
            signal, color = "NÖTR", "gray"
            if score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif score >= 60: signal, color = "AL 🌱", "blue"
            elif score <= 30: signal, color = "SAT 🔻", "red"

            return {
                "Hisse": ticker, "Fiyat": last['Close'], "Skor": int(score),
                "Sinyal": signal, "Renk": color, "RSI": last['RSI'],
                "Stop": last['Close']-(last['ATR']*1.5), "Hedef": last['Close']+(last['ATR']*3),
                "Yorumlar": reasons, "Haberler": n_list, "Data": df, "Features": features, "AI_Conf": ai_conf,
                "Temel": self.get_fundamentals(ticker) if mode=="PRO" else None
            }
        except: return None

    def analyze_batch(self, tickers):
        res = []
        try:
            d = yf.download([f"{t}.IS" for t in tickers], period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
            for t in tickers:
                try:
                    df = d[f"{t}.IS"].dropna()
                    if len(df)<50: continue
                    rsi = ta.rsi(df['Close'], 14).iloc[-1]
                    last = df['Close'].iloc[-1]
                    vwap = (df['Volume']*(df['High']+df['Low']+df['Close'])/3).cumsum()/df['Volume'].cumsum()
                    
                    sc = 50
                    if rsi<40 and last>vwap.iloc[-1]: sc=85
                    elif rsi>70: sc=20
                    
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
            btn = st.button("ANALİZ ET 🔍", type="primary")

        if btn and sembol:
            with st.spinner("Analiz ediliyor..."):
                res = engine.analyze(sembol)
                if res:
                    st.session_state['last_res'] = res
                    
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
                fig = go.Figure(data=[go.Candlestick(x=res['Data'].index, open=res['Data']['Open'], high=res['Data']['High'], low=res['Data']['Low'], close=res['Data']['Close'])])
                
                # HATA DÜZELTME: Dinamik kolon kontrolü
                bbu = next((c for c in res['Data'].columns if c.startswith('BBU')), None)
                if bbu: fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data'][bbu], line=dict(color='gray', dash='dot'), name='Bollinger'))
                
                fig.add_trace(go.Scatter(x=res['Data'].index, y=res['Data']['VWAP'], line=dict(color='orange'), name='VWAP'))
                fig.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig, use_container_width=True)
            
            with g2:
                if res['Renk']=='green': st.success(f"**{res['Sinyal']}**")
                else: st.warning(f"**{res['Sinyal']}**")
                st.write("#### 📝 Nedenler")
                for y in res['Yorumlar']: st.markdown(f"✅ {y}")
                
                st.markdown("---")
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
