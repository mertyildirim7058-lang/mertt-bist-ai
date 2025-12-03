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
from datetime import datetime, timedelta, timezone

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

# --- 2. ÇOKLU CANLI FİYAT MOTORU ---
def get_realtime_price(ticker):
    clean_ticker = ticker.replace('.IS', '')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={clean_ticker}"
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.content, "html.parser")
        fiyat = soup.find('span', {'id': 'ctl00_ctl58_g_76ae4504_9743_4791_98df_dce2ca95cc0d_lblSonFiyat'})
        if fiyat: return float(fiyat.text.replace(',', '.'))
    except: pass
    
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{clean_ticker}-detay/"
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.content, "html.parser")
        fiyat = soup.select_one('.price-arrow-down, .price-arrow-up, .text-2')
        if fiyat: return float(fiyat.text.strip().replace(',', '.'))
    except: pass
    return None

# --- 3. HABER MOTORU (GÜNCELLENDİ: TAZE HABER ZORUNLU) ---
class NewsEngine:
    def __init__(self):
        self.risk_keywords = ['savaş', 'kriz', 'düşüş', 'ceza', 'zarar', 'satış', 'enflasyon']
        self.tech_keywords = ['rekor', 'büyüme', 'onay', 'temettü', 'kar', 'anlaşma', 'yatırım', 'yapay zeka']

    def get_latest_news(self, ticker):
        news_list = []
        score = 0
        
        # Google News'e "when:1d" (Son 24 saat) parametresini ekledik
        # Ayrıca sorguyu genişlettik
        urls = [
            f"https://news.google.com/rss/search?q={ticker}+hisse+when:2d&hl=tr&gl=TR&ceid=TR:tr",
            f"https://news.google.com/rss/search?q={ticker}+kap+when:2d&hl=tr&gl=TR&ceid=TR:tr"
        ]
        
        for url in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:8]:
                    title = entry.title
                    link = entry.link
                    try:
                        # Tarih İşleme (Timezone sorununu çözmek için)
                        if hasattr(entry, 'published_parsed'):
                            # RSS tarihini UTC'ye çevirip üzerine 3 saat (TR Saati) ekliyoruz
                            published_time = datetime(*entry.published_parsed[:6])
                            # Basitçe şu anki zamanla kıyasla
                            now = datetime.utcnow() # Server saati genelde UTC'dir
                            diff = now - published_time
                            
                            # Eğer haber son 24 saat içindeyse SAAT göster
                            if diff.days < 1:
                                # TR Saati Gösterimi (UTC+3)
                                tr_time = published_time + timedelta(hours=3)
                                date_str = tr_time.strftime("%H:%M")
                            else:
                                date_str = published_time.strftime("%d.%m")

                            t_lower = title.lower()
                            impact = "Nötr"
                            if any(k in t_lower for k in self.tech_keywords): score += 10; impact="Pozitif"
                            if any(k in t_lower for k in self.risk_keywords): score -= 15; impact="Negatif"
                            
                            news_list.append({"Title": title, "Link": link, "Date": date_str, "Impact": impact, "Timestamp": time.mktime(entry.published_parsed)})
                    except: pass
            except: pass
        
        # Tekrarları Temizle ve En Yeniye Göre Sırala
        unique_news = []
        seen = set()
        # Timestamp'e göre tersten sırala (En yeni en üstte)
        news_list.sort(key=lambda x: x['Timestamp'], reverse=True)
        
        for n in news_list:
            if n['Title'] not in seen:
                unique_news.append(n)
                seen.add(n['Title'])
                
        return max(-30, min(30, score)), unique_news[:10]

# --- 4. DERİN TEKNİK ANALİZ MOTORU ---
class TechnicalEngine:
    def detect_patterns(self, df):
        patterns = []
        score = 0
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            body = abs(last['Close'] - last['Open'])
            wick_lower = min(last['Close'], last['Open']) - last['Low']
            
            if wick_lower > (body * 2): patterns.append("Hammer (Çekiç)"); score += 15
            
            if prev['Close'] < prev['Open'] and last['Close'] > last['Open']:
                if last['Close'] > prev['Open'] and last['Open'] < prev['Close']:
                    patterns.append("Yutan Boğa"); score += 20
            
            lows = df['Low'].tail(20).values
            if len(lows) > 10:
                min1 = np.min(lows[:10])
                min2 = np.min(lows[10:])
                if abs(min1 - min2) / min1 < 0.01: patterns.append("İkili Dip"); score += 15
        except: pass
        return patterns, score

    def analyze(self, df):
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            bb = ta.bbands(df['Close'], length=20)
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            
            if bb is not None: df = pd.concat([df, bb], axis=1)
            if kc is not None: df = pd.concat([df, kc], axis=1)
            df = pd.concat([df, ichimoku], axis=1)
            
            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            score = 50
            reasons = []
            
            if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']: score += 15; reasons.append("Ichimoku Bulutu Üstünde")
            
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']:
                score += 10
                if prev['MACD_12_26_9'] < prev['MACDs_12_26_9']: score += 10; reasons.append("MACD AL Kesişimi")
            
            try:
                bbl = next((c for c in df.columns if c.startswith('BBL')), None)
                if bbl and last['Close'] <= df[bbl].iloc[-1] * 1.01: score += 15; reasons.append("Bollinger Alt Bant")
            except: pass
            
            if df['OBV'].iloc[-1] > df['OBV'].iloc[-5]: score += 10
            
            if last['Close'] > last['VWAP']: score += 10
            if last['RSI'] < 30: score += 20; reasons.append("RSI Dip")
            elif last['RSI'] > 70 and score < 60: score -= 20; reasons.append("RSI Tepe")

            pats, pat_score = self.detect_patterns(df)
            score += pat_score
            for p in pats: reasons.append(f"Formasyon: {p}")

            return max(0, min(100, score)), reasons, df
        except: return 0, [], df

# --- ANA MOTOR ---
class TradingEngine:
    def __init__(self):
        self.tech = TechnicalEngine()
        self.news = NewsEngine()

    def get_fundamentals(self, ticker):
        try:
            stock = yf.Ticker(f"{ticker}.IS")
            info = stock.info
            fk = info.get('trailingPE', None)
            pddd = info.get('priceToBook', None)
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-"}
        except: return None

    def analyze(self, ticker):
        try:
            t = f"{ticker}.IS"
            df = yf.download(f"{ticker}.IS", period="6mo", interval="60m", progress=False)
            if df is None or len(df) < 50: return None
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Istanbul')
            
            live_price = get_realtime_price(ticker)
            is_live = False
            if live_price:
                last_close = df['Close'].iloc[-1]
                if abs(live_price - last_close) / last_close < 0.20:
                    df.iloc[-1, df.columns.get_loc('Close')] = live_price
                    df.iloc[-1, df.columns.get_loc('High')] = max(live_price, df.iloc[-1]['High'])
                    df.iloc[-1, df.columns.get_loc('Low')] = min(live_price, df.iloc[-1]['Low'])
                    is_live = True
            
            tech_score, tech_reasons, processed_df = self.tech.analyze(df)
            news_score, news_list = self.news.get_latest_news(ticker)
            fund = self.get_fundamentals(ticker)
            
            final_score = max(0, min(100, tech_score + news_score))
            
            signal, color = "NÖTR", "gray"
            if final_score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif final_score >= 60: signal, color = "AL 🌱", "blue"
            elif final_score <= 30: signal, color = "SAT 🔻", "red"
            
            last_close = processed_df['Close'].iloc[-1]
            atr = processed_df['ATR'].iloc[-1]

            return {
                "Hisse": ticker, "Fiyat": last_close, "Skor": int(final_score),
                "Sinyal": signal, "Renk": color, 
                "Stop": last_close - (atr * 1.5), "Hedef": last_close + (atr * 3.0),
                "Yorumlar": tech_reasons, "Haberler": news_list,
                "Temel": fund, "Data": processed_df, "Is_Live": is_live
            }
        except: return None
    
    def analyze_batch(self, tickers_list):
        results = []
        try:
            data = yf.download([f"{t}.IS" for t in tickers_list], period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
            for ticker in tickers_list:
                try:
                    df = data[f"{ticker}.IS"].copy().dropna()
                    if len(df) < 50: continue
                    rsi = ta.rsi(df['Close'], 14).iloc[-1]
                    vwap = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
                    last = df['Close'].iloc[-1]
                    
                    score = 50
                    if rsi < 40: score += 20
                    if last > vwap.iloc[-1]: score += 10
                    
                    signal = "NÖTR"
                    if score >= 80: signal = "GÜÇLÜ AL 🚀"
                    elif score <= 30: signal = "SAT 🔻"
                    
                    if signal != "NÖTR":
                        results.append({"Hisse": ticker, "Fiyat": last, "Sinyal": signal, "RSI": rsi, "Skor": score})
                except: continue
        except: pass
        return results

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.divider()
        menu = st.radio("Panel", ["💬 Hisse Sor / Analiz", "📡 Piyasa Radarı", "Çıkış"])
        if menu == "Çıkış": st.session_state['giris_yapildi'] = False; st.rerun()

    engine = TradingEngine()
    
    # Liste Çekici (Manuel Liste)
    tum_hisseler = [
        "THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS", 
        "HEKTS", "PETKM", "ISCTR", "SAHOL", "FROTO", "YKBNK", "EKGYO", "ODAS", "KOZAL", "KONTR", 
        "ASTOR", "EUPWR", "GUBRF", "OYAKC", "TCELL", "TTKOM", "ENKAI", "VESTL", "ARCLK", "TOASO",
        "PGSUS", "TAVHL", "MGROS", "SOKM", "AEFES", "AGHOL", "AHGAZ", "AKFGY", "AKSA", "ALARK",
        "ALFAS", "BRSAN", "CANTE", "CCOLA", "CIMSA", "DOAS", "EGEEN", "ENJSA", "GESAN", "GUBRF"
    ]

    if menu == "💬 Hisse Sor / Analiz":
        st.title("🤖 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu:", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("ANALİZ ET 🔍", type="primary")

        if btn and sembol:
            with st.spinner(f"{sembol} analiz ediliyor..."):
                res = engine.analyze(sembol)
                if res:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL", delta="Canlı" if res['Is_Live'] else "Gecikmeli")
                    k2.metric("Skor", f"{res['Skor']}/100")
                    k3.metric("Sinyal", res['Sinyal'])
                    temel = res['Temel']
                    k4.metric("Temel", f"F/K: {temel['FK']}" if temel else "-")
                    
                    st.divider()
                    
                    c_sol, c_sag = st.columns([2, 1])
                    with c_sol:
                        st.subheader("📊 Teknik Görünüm")
                        df = res['Data']
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
                        
                        try:
                            bbu = next((c for c in df.columns if c.startswith('BBU')), None)
                            if bbu: fig.add_trace(go.Scatter(x=df.index, y=df[bbu], line=dict(color='gray', dash='dot'), name='Bollinger'))
                            kcu = next((c for c in df.columns if c.startswith('KCU')), None)
                            if kcu: fig.add_trace(go.Scatter(x=df.index, y=df[kcu], line=dict(color='purple'), name='Keltner'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange'), name='VWAP'))
                        except: pass
                        
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with c_sag:
                        st.subheader("📝 Analiz Notları")
                        if res['Renk'] == 'green': st.success(f"**{res['Sinyal']}**")
                        else: st.warning(f"**{res['Sinyal']}**")
                        
                        st.info(f"Hedef: {res['Hedef']:.2f}")
                        st.error(f"Stop: {res['Stop']:.2f}")
                        
                        for y in res['Yorumlar']: st.markdown(f"✅ {y}")
                        
                        if res['Temel']:
                            st.markdown("---")
                            st.write(f"**F/K:** {res['Temel']['FK']}")
                            st.write(f"**PD/DD:** {res['Temel']['PD_DD']}")

                    st.markdown("---")
                    st.subheader("📰 Haber Akışı")
                    if res['Haberler']:
                        for n in res['Haberler']:
                            color = "🟢" if n['Impact'] == "Pozitif" else "🔴" if n['Impact'] == "Negatif" else "⚪"
                            st.markdown(f"{color} **[{n['Title']}]({n['Link']})**")
                            st.caption(f"⏰ {n['Date']}")
                    else: st.info("Önemli haber yok.")

                else: st.error("Analiz yapılamadı.")

    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
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
                st.success(f"Tarama Bitti! {len(df)} Fırsat Bulundu.")
                st.dataframe(df.style.format({"Fiyat": "{:.2f}", "RSI": "{:.0f}"}).background_gradient(subset=['Skor'], cmap='RdYlGn'), use_container_width=True)
            else: st.warning("Sinyal yok.")

if __name__ == "__main__":
    main()
