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
from datetime import datetime, timedelta

# --- 1. AYARLAR ---
LOGO_INTERNET_LINKI = "https://raw.githubusercontent.com/kullaniciadi/proje/main/logo.png"

st.set_page_config(page_title="MERTT AI Terminal", layout="wide", page_icon="🦅")

def logo_goster():
    try: st.image("logo.png", use_container_width=True)
    except: 
        try: st.image(LOGO_INTERNET_LINKI, use_container_width=True)
        except: st.header("🦅 MERTT AI")

# --- 2. ÇOKLU CANLI FİYAT MOTORU (HATASIZ) ---
def get_realtime_price(ticker):
    """
    3 Farklı Kaynaktan Fiyat Dener. En güncelini alır.
    """
    clean_ticker = ticker.replace('.IS', '')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # Kaynak 1: İş Yatırım
    try:
        url = f"https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse={clean_ticker}"
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.content, "html.parser")
        fiyat = soup.find('span', {'id': 'ctl00_ctl58_g_76ae4504_9743_4791_98df_dce2ca95cc0d_lblSonFiyat'})
        if fiyat: return float(fiyat.text.replace(',', '.'))
    except: pass

    # Kaynak 2: BigPara
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{clean_ticker}-detay/"
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.content, "html.parser")
        fiyat = soup.select_one('.price-arrow-down, .price-arrow-up, .text-2')
        if fiyat: return float(fiyat.text.strip().replace(',', '.'))
    except: pass
    
    # Kaynak 3: Google Finance (Yedek)
    try:
        url = f"https://www.google.com/finance/quote/{clean_ticker}:BIST"
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.content, "html.parser")
        fiyat = soup.find('div', {'class': 'YMlKec fxKbKc'})
        if fiyat: return float(fiyat.text.replace('₺', '').replace(',', ''))
    except: pass

    return None

# --- 3. GELİŞMİŞ HABER MOTORU (30 DK FİLTRELİ) ---
class NewsEngine:
    def __init__(self):
        self.risk_keywords = ['savaş', 'kriz', 'düşüş', 'ceza', 'zarar', 'satış', 'enflasyon']
        self.tech_keywords = ['rekor', 'büyüme', 'onay', 'temettü', 'kar', 'anlaşma', 'yatırım', 'yapay zeka']

    def get_latest_news(self, ticker):
        """Sadece son 30 dakika - 24 saat içindeki haberleri getirir"""
        news_list = []
        score = 0
        
        # RSS Kaynakları
        urls = [
            f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr",
            "https://www.trthaber.com/xml/ekonomi.xml"
        ]
        
        for url in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    title = entry.title
                    link = entry.link
                    
                    # Zaman Kontrolü
                    try:
                        if hasattr(entry, 'published_parsed'):
                            dt = datetime(*entry.published_parsed[:6])
                            diff = datetime.now() - dt
                            
                            # 24 Saatten eski haberleri alma
                            if diff.days < 1:
                                date_str = dt.strftime("%H:%M")
                                is_hot = "🔥" if diff.seconds < 1800 else "" # 30 dk'dan yeniyse Ateş koy
                                
                                # Puanlama
                                t_lower = title.lower()
                                impact = "Nötr"
                                if any(k in t_lower for k in self.tech_keywords): 
                                    score += 10; impact="Pozitif"
                                if any(k in t_lower for k in self.risk_keywords): 
                                    score -= 15; impact="Negatif"
                                    
                                news_list.append({
                                    "Title": f"{is_hot} {title}", 
                                    "Link": link, 
                                    "Date": date_str,
                                    "Impact": impact
                                })
                    except: pass
            except: pass
            
        return max(-30, min(30, score)), news_list

# --- 4. DERİN TEKNİK ANALİZ MOTORU ---
class TechnicalEngine:
    def analyze(self, df):
        """Tüm İndikatörleri Hesaplar ve Sinyal Üretir"""
        try:
            # İndikatörler
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # MACD
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1) # MACD_12_26_9
            
            # Bollinger & Keltner (Sıkışma için)
            bb = ta.bbands(df['Close'], length=20)
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            if bb is not None: df = pd.concat([df, bb], axis=1)
            if kc is not None: df = pd.concat([df, kc], axis=1)
            
            # Ichimoku
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
            df = pd.concat([df, ichimoku], axis=1)
            
            # VWAP
            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # --- GELİŞMİŞ PUANLAMA (0-100) ---
            score = 50
            reasons = []
            
            # 1. Ichimoku (Trendin Kralı)
            # Fiyat Bulutun Üstünde mi? (Span A ve B)
            if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']:
                score += 15
                reasons.append("Ichimoku: Fiyat Bulutun Üstünde (Güçlü Trend)")
            elif last['Close'] < last['ISA_9'] and last['Close'] < last['ISB_26']:
                score -= 15
                
            # 2. MACD (Momentum)
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']:
                score += 10
                if prev['MACD_12_26_9'] < prev['MACDs_12_26_9']: # Yeni Kesişim
                    score += 10
                    reasons.append("MACD: Yeni AL Sinyali")
            
            # 3. Bollinger & Keltner (Sıkışma - Patlama)
            # Bollinger Bantları, Keltner Kanallarının içine girdiyse "Sıkışma" vardır (Squeeze)
            # Dinamik sütun isimlerini bulalım
            try:
                bbu = df[[c for c in df.columns if c.startswith('BBU')][0]].iloc[-1]
                bbl = df[[c for c in df.columns if c.startswith('BBL')][0]].iloc[-1]
                kcu = df[[c for c in df.columns if c.startswith('KCU')][0]].iloc[-1]
                kcl = df[[c for c in df.columns if c.startswith('KCL')][0]].iloc[-1]
                
                if bbu < kcu and bbl > kcl:
                    reasons.append("Squeeze: Fiyat Sıkıştı, Patlama Yakın!")
                    score += 5 # Nötr ama dikkat çekici
            except: pass
            
            # 4. RSI ve VWAP (Teyit)
            if last['Close'] > last['VWAP']: score += 10
            
            if last['RSI'] < 30: 
                score += 20; reasons.append("RSI: Aşırı Satım (Dip)")
            elif last['RSI'] > 70: 
                # Eğer trend güçlüyse (Ichimoku üstü) RSI yüksekliği iyidir, düşürme.
                if score < 60: score -= 20; reasons.append("RSI: Aşırı Alım (Risk)")

            # 5. Mum Formasyonları
            # Yutan Boğa (Bullish Engulfing)
            if prev['Close'] < prev['Open'] and last['Close'] > last['Open']:
                if last['Close'] > prev['Open'] and last['Open'] < prev['Close']:
                    score += 20
                    reasons.append("Formasyon: Yutan Boğa (Bullish Engulfing)")

            return max(0, min(100, score)), reasons, df
            
        except Exception as e: 
            print(e)
            return 0, [], df

# --- ANA MOTOR ---
class TradingEngine:
    def __init__(self):
        self.tech = TechnicalEngine()
        self.news = NewsEngine()

    def get_fundamentals(self, ticker):
        """Temel Analiz"""
        try:
            stock = yf.Ticker(f"{ticker}.IS")
            info = stock.info
            fk = info.get('trailingPE', None)
            pddd = info.get('priceToBook', None)
            
            # Değerleme
            yorum = "NÖTR"
            puan = 0
            if fk and fk < 8: puan += 1
            if pddd and pddd < 1.5: puan += 1
            
            if puan == 2: yorum = "KELEPİR (UCUZ)"
            elif fk and fk > 30: yorum = "PRİMLİ (PAHALI)"
            
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-", "Yorum": yorum}
        except: return None

    def analyze(self, ticker):
        try:
            # 1. Geçmiş Veri (6 Ay - Ichimoku için şart)
            df = yf.download(f"{ticker}.IS", period="6mo", interval="60m", progress=False)
            if df is None or len(df) < 100: return None
            
            # MultiIndex Temizliği
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            
            # Saat Dilimi
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Istanbul')
            
            # 2. CANLI FİYAT YAMASI (Çoklu Kaynak)
            live_price = get_realtime_price(ticker)
            is_live = False
            
            if live_price:
                last_close = df['Close'].iloc[-1]
                # %20 Sapma Kontrolü (Hatalı veriyi engelle)
                if abs(live_price - last_close) / last_close < 0.20:
                    df.iloc[-1, df.columns.get_loc('Close')] = live_price
                    df.iloc[-1, df.columns.get_loc('High')] = max(live_price, df.iloc[-1]['High'])
                    df.iloc[-1, df.columns.get_loc('Low')] = min(live_price, df.iloc[-1]['Low'])
                    is_live = True
            
            # 3. TEKNİK ANALİZ
            tech_score, tech_reasons, processed_df = self.tech.analyze(df)
            
            # 4. HABER ANALİZİ
            news_score, news_list = self.news.get_latest_news(ticker)
            
            # 5. TEMEL ANALİZ
            fund = self.get_fundamentals(ticker)
            
            # FİNAL SKOR
            final_score = tech_score + news_score
            final_score = max(0, min(100, final_score))
            
            # Sinyal
            signal = "NÖTR"
            color = "gray"
            if final_score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif final_score >= 60: signal, color = "AL 🌱", "blue"
            elif final_score <= 30: signal, color = "SAT 🔻", "red"
            
            # Hedefler
            last_close = processed_df['Close'].iloc[-1]
            atr = processed_df['ATR'].iloc[-1]
            stop = last_close - (atr * 1.5)
            target = last_close + (atr * 3.0)

            return {
                "Hisse": ticker, "Fiyat": last_close, "Skor": int(final_score),
                "Sinyal": signal, "Renk": color, 
                "Stop": stop, "Hedef": target,
                "Yorumlar": tech_reasons, "Haberler": news_list,
                "Temel": fund, "Data": processed_df, "Is_Live": is_live
            }

        except Exception as e: 
            print(f"Hata: {e}")
            return None

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.divider()
        menu = st.radio("Panel", ["💬 Hisse Sor / Analiz", "📡 Piyasa Radarı", "Çıkış"])
        if menu == "Çıkış": st.session_state['giris_yapildi'] = False; st.rerun()

    engine = TradingEngine()

    if menu == "💬 Hisse Sor / Analiz":
        st.title("🤖 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu (Örn: THYAO):", "").upper()
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
                    k4.metric("Temel", res['Temel']['Yorum'] if res['Temel'] else "-")
                    
                    st.divider()
                    
                    # Grafik ve Detaylar
                    c_sol, c_sag = st.columns([2, 1])
                    with c_sol:
                        st.subheader("📊 Teknik Görünüm")
                        df = res['Data']
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
                        
                        # İndikatörleri Çiz (Ichimoku, Bollinger)
                        try:
                            fig.add_trace(go.Scatter(x=df.index, y=df['ISA_9'], line=dict(color='rgba(0, 255, 0, 0.3)'), name='Ichimoku A', visible='legendonly'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['ISB_26'], line=dict(color='rgba(255, 0, 0, 0.3)'), name='Ichimoku B', visible='legendonly'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange'), name='VWAP'))
                        except: pass
                        
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with c_sag:
                        st.subheader("📝 Analiz Notları")
                        if res['Renk'] == 'green': st.success(f"**{res['Sinyal']}**")
                        elif res['Renk'] == 'red': st.error(f"**{res['Sinyal']}**")
                        
                        st.info(f"Hedef: {res['Hedef']:.2f}")
                        st.error(f"Stop: {res['Stop']:.2f}")
                        
                        for y in res['Yorumlar']: st.markdown(f"✅ {y}")
                        
                        if res['Temel']:
                            st.markdown("---")
                            st.write(f"**F/K:** {res['Temel']['FK']}")
                            st.write(f"**PD/DD:** {res['Temel']['PD_DD']}")

                    st.markdown("---")
                    st.subheader("📰 Haber Akışı (Son 24 Saat)")
                    if res['Haberler']:
                        for n in res['Haberler']:
                            color = "🟢" if n['Impact'] == "Pozitif" else "🔴" if n['Impact'] == "Negatif" else "⚪"
                            st.markdown(f"{color} **[{n['Title']}]({n['Link']})** - *{n['Date']}*")
                    else: st.info("Son 24 saatte önemli haber yok.")

                else: st.error("Veri alınamadı.")

    elif menu == "📡 Piyasa Radarı":
        st.title("📡 Piyasa Radarı")
        
        # BIST 30 + Önemli Hisseler Listesi (Hız için 50 hisse)
        tickers = ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS", "HEKTS", "PETKM", "ISCTR", "SAHOL", "FROTO", "YKBNK", "EKGYO", "ODAS", "KOZAL", "KONTR", "ASTOR", "EUPWR", "GUBRF", "OYAKC", "TCELL", "TTKOM", "ENKAI", "VESTL", "ARCLK", "TOASO", "PGSUS", "TAVHL", "MGROS", "SOKM", "AEFES", "AGHOL", "AHGAZ", "AKFGY", "AKSA", "ALARK", "ALFAS", "BRSAN", "CANTE", "CCOLA", "CIMSA", "DOAS", "EGEEN", "ENJSA", "GESAN", "GUBRF"]
        
        if st.button("TARAMAYI BAŞLAT 🚀", type="primary"):
            results = []
            bar = st.progress(0)
            
            # Batch Tarama (Döngü)
            for i, t in enumerate(tickers):
                res = engine.analyze(t, mode="PRO")
                if res and (res['Sinyal'] == "GÜÇLÜ AL 🚀" or res['Sinyal'] == "SAT 🔻"):
                     # KART GÖRÜNÜMÜ (Senin istediğin gibi)
                     with st.expander(f"{res['Sinyal']} | {res['Hisse']} - {res['Fiyat']:.2f} TL (Skor: {res['Skor']})"):
                         c1, c2, c3 = st.columns(3)
                         c1.metric("Sinyal", res['Sinyal'])
                         c2.metric("RSI", f"{res['RSI']:.0f}")
                         fk = res['Temel']['FK'] if res['Temel'] else "-"
                         c3.metric("F/K", fk)
                         st.write(f"**Neden:** {', '.join(res['Yorumlar'])}")
                         if res['Haberler']: st.markdown(f"📰 **Son Haber:** {res['Haberler'][0]['Title']}")

                bar.progress((i+1)/len(tickers))
            
            bar.empty()
            st.success("Tarama Tamamlandı.")

if __name__ == "__main__":
    main()
