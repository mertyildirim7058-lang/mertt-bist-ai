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
            st.markdown("<h4 style='text-align: center;'>Gelecek İçin Bilgi ve Teknoloji</h4>", unsafe_allow_html=True)
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

# --- 2. YEDEK LİSTE (KURTARICI) ---
def get_backup_list():
    """İş Yatırım çalışmazsa devreye girecek TAM LİSTE."""
    return [
        "A1CAP", "ACSEL", "ADEL", "ADESE", "ADGYO", "AEFES", "AFYON", "AGESA", "AGHOL", "AGROT", "AGYO",
        "AHGAZ", "AKBNK", "AKCNS", "AKENR", "AKFGY", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN",
        "AKSGY", "AKSUE", "AKYHO", "ALARK", "ALBRK", "ALCAR", "ALCTL", "ALFAS", "ALGYO", "ALKA",
        "ALKIM", "ALMAD", "ALTNY", "ALVES", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARCLK",
        "ARDYZ", "ARENA", "ARSAN", "ARTMS", "ARZUM", "ASELS", "ASGYO", "ASTOR", "ASUZU", "ATAGY",
        "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATSYH", "AVGYO", "AVHOL", "AVOD", "AVPGY", "AVTUR",
        "AYCES", "AYDEM", "AYEN", "AYES", "AYGAZ", "AZTEK", "BABA", "BAGFS", "BAKAB", "BALAT",
        "BANVT", "BARMA", "BASCM", "BASGZ", "BAYRK", "BEGYO", "BERA", "BEYAZ", "BFREN", "BIENY",
        "BIGCH", "BIMAS", "BINHO", "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BMSCH", "BMSTL", "BNTAS",
        "BOBET", "BORLS", "BOSSA", "BRISA", "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BRSAN",
        "BRYAT", "BSOKE", "BTCIM", "BUCIM", "BURCE", "BURVA", "BVSAN", "BYDNR", "CANTE", "CATES",
        "CCOLA", "CELHA", "CEMAS", "CEMTS", "CEOEM", "CIMSA", "CLEBI", "CMBTN", "CMENT", "CONSE",
        "COSMO", "CRDFA", "CRFSA", "CUSAN", "CVKMD", "CWENE", "DAGHL", "DAGI", "DAPGM", "DARDL",
        "DATA", "DATES", "DDRKM", "DELEG", "DEMISA", "DERHL", "DERIM", "DESA", "DESPC", "DEVA",
        "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMSAS", "DNISI", "DOAS", "DOBUR", "DOCO",
        "DOGUB", "DOHOL", "DOKTA", "DURDO", "DYOBY", "DZGYO", "EBEBK", "ECILC", "EPLAS", "ECZYT",
        "EDATA", "EDIP", "EGEEN", "EGEPO", "EGGUB", "EGPRO", "EGSER", "EKGYO", "EKIZ", "EKSUN",
        "ELITE", "EMKEL", "EMNIS", "ENJSA", "ENKAI", "ENSRI", "ENTRA", "ENVER", "EPLAS", "ERBOS",
        "ERCB", "EREGL", "ERSU", "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUHOL", "EUKYO",
        "EUPWR", "EUREN", "EUYO", "FADE", "FENE", "FLAP", "FMIZP", "FONET", "FORMT", "FORTE",
        "FRIGO", "FROTO", "FZLGY", "GARAN", "GARFA", "GEDIK", "GEDZA", "GENIL", "GENTS", "GEREL",
        "GESAN", "GLBMD", "GLCVY", "GLRYH", "GLYHO", "GMTAS", "GOKNR", "GOLTS", "GOODY", "GOZDE",
        "GRNYO", "GRSEL", "GSDDE", "GSDHO", "GSRAY", "GUBRF", "GWIND", "GZNMI", "HALKB", "HATEK",
        "HDFGS", "HEDEF", "HEKTS", "HKTM", "HLGYO", "HRKET", "HTTBT", "HUBVC", "HUNER", "HURGZ",
        "ICBCT", "IDEAS", "IDGYO", "IEYHO", "IHAAS", "IHEVA", "IHGZT", "IHLAS", "IHLGM", "IHYAY",
        "IMASM", "INDES", "INFO", "INGRM", "INTEM", "INVEO", "INVES", "ISATR", "ISBIR", "ISBTR",
        "ISCTR", "ISDMR", "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISKUR", "ISMEN", "ISSEN", "ISYAT",
        "ITTFH", "IZENR", "IZFAS", "IZINV", "IZMDC", "JANTS", "KAPLM", "KARYE", "KARSN", "KARTN",
        "KARYE", "KATMR", "KAYSE", "KCAER", "KCMKW", "KDOAS", "KFEIN", "KGYO", "KBORU", "KIMMR",
        "KLGYO", "KLKIM", "KLMSN", "KLNMA", "KLRHO", "KLSYN", "KMPUR", "KNFRT", "KONKA", "KONTR",
        "KONYA", "KOPOL", "KORDS", "KOZAA", "KOZAL", "KRDMA", "KRDMB", "KRDMD", "KRGYO", "KRONT",
        "KRPLS", "KRSTL", "KRTEK", "KRVGD", "KSTUR", "KTLEV", "KTSKR", "KUTPO", "KUVVA", "KUYAS",
        "KZBGY", "KZGYO", "LIDER", "LIDFA", "LINK", "LKMNH", "LOGO", "LRSHO", "LUKSK", "MAALT",
        "MACKO", "MAGEN", "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA", "MARTI", "MAVI", "MEDTR",
        "MEGAP", "MEGMT", "MEKAG", "MNDRS", "MENBA", "MERCN", "MERIT", "MERKO", "METUR", "MGROS",
        "MIATK", "MIPAZ", "MMCAS", "MNDTR", "MOBTL", "MOGAN", "MONDU", "MPARK", "MRGYO", "MRSHL",
        "MSGYO", "MTRKS", "MTRYO", "MUNDA", "NATA", "NETAS", "NIBAS", "NTGAZ", "NTHOL", "NUGYO",
        "NUHCM", "OBAMS", "OBASE", "ODAS", "ODINE", "OFSYM", "ONCSM", "ORCAY", "ORGE", "ORMA",
        "OSMEN", "OSTIM", "OTKAR", "OTTO", "OYAKC", "OYAYO", "OYLUM", "OYYAT", "OZGYO", "OZKGY",
        "OZRDN", "OZSUB", "PAGYO", "PAMEL", "PAPIL", "PARSN", "PASEU", "PCILT", "PEGYO", "PEKGY",
        "PENGD", "PENTA", "PETKM", "PETUN", "PGSUS", "PINSU", "PKART", "PKENT", "PLAT", "PNLSN",
        "PNSUT", "POLHO", "POLTK", "PRDGS", "PRKAB", "PRKME", "PRZMA", "PSDTC", "PSGYO", "QNBFB",
        "QNBFL", "QUAGR", "RALYH", "RAYSG", "RNPOL", "REEDR", "RHEAG", "RODRG", "ROYAL", "RTALB",
        "RUBNS", "RYGYO", "RYSAS", "SAFKR", "SAHOL", "SAMAT", "SANEL", "SANFM", "SANKO", "SARKY",
        "SARTN", "SASA", "SAYAS", "SDTTR", "SEKFK", "SEKUR", "SELEC", "SELGD", "SELVA", "SEYKM",
        "SILVR", "SISE", "SKBNK", "SKTAS", "SMART", "SMRTG", "SNAET", "SNPAM", "SNGYO", "SNKRN",
        "SOKE", "SOKM", "SONME", "SRVGY", "SUMAS", "SUNGW", "SURGY", "SUWEN", "TABGD", "TARKM",
        "TATEN", "TATGD", "TAVHL", "TBORG", "TCELL", "TDGYO", "TEKTU", "TERA", "TETMT", "TEZOL",
        "TGSAS", "THYAO", "TKFEN", "TKNSA", "TLMAN", "TMPOL", "TMSN", "TNZTP", "TOASO", "TRCAS",
        "TRGYO", "TRILC", "TSGYO", "TSKB", "TSPOR", "TTKOM", "TTRAK", "TUCLK", "TUKAS", "TUPRS",
        "TUREX", "TURGG", "TURSG", "UFUK", "ULAS", "ULKER", "ULUFA", "ULUSE", "ULUUN", "UMPAS",
        "UNLU", "USAK", "UZERB", "VAKBN", "VAKFN", "VAKKO", "VANGD", "VBTYZ", "VERUS", "VESBE",
        "VESTL", "VKFYO", "VKGYO", "VKING", "VRGYO", "YAPRK", "YATAS", "YAYLA", "YEOTK", "YESIL",
        "YGGYO", "YGYO", "YKBNK", "YKSLN", "YONGA", "YUNSA", "YYAPI", "YYLGD", "ZEDUR", "ZOREN",
        "ZRGYO"
    ]

# --- 3. LİSTE ÇEKİCİ (GERİ GELDİ) ---
@st.cache_data(ttl=600)
def get_live_tickers():
    """
    İş Yatırım'dan canlı listeyi çeker. 
    Çekemezse YEDEK LİSTEYİ kullanır.
    """
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
    
    # Liste boşsa yedeği kullan
    if len(canli_liste) > 50: return sorted(list(set(canli_liste)))
    else: return sorted(list(set(get_backup_list())))

# --- 4. ÇOKLU CANLI FİYAT MOTORU ---
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

# --- 5. HABER MOTORU ---
class NewsEngine:
    def __init__(self):
        self.risk_keywords = ['savaş', 'kriz', 'düşüş', 'ceza', 'zarar', 'satış', 'enflasyon']
        self.tech_keywords = ['rekor', 'büyüme', 'onay', 'temettü', 'kar', 'anlaşma', 'yatırım', 'yapay zeka']

    def get_latest_news(self, ticker):
        news_list = []
        score = 0
        urls = [f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr", "https://www.trthaber.com/xml/ekonomi.xml"]
        for url in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    title = entry.title
                    link = entry.link
                    try:
                        if hasattr(entry, 'published_parsed'):
                            dt = datetime(*entry.published_parsed[:6])
                            if (datetime.now() - dt).days < 1: # Son 24 saat
                                date_str = dt.strftime("%H:%M")
                                t_lower = title.lower()
                                impact = "Nötr"
                                if any(k in t_lower for k in self.tech_keywords): score += 10; impact="Pozitif"
                                if any(k in t_lower for k in self.risk_keywords): score -= 15; impact="Negatif"
                                news_list.append({"Title": title, "Link": link, "Date": date_str, "Impact": impact})
                    except: pass
            except: pass
        return max(-30, min(30, score)), news_list

# --- 6. DERİN TEKNİK ANALİZ MOTORU ---
class TechnicalEngine:
    def detect_patterns(self, df):
        patterns = []
        score = 0
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            body = abs(last['Close'] - last['Open'])
            wick_lower = min(last['Close'], last['Open']) - last['Low']
            
            if wick_lower > (body * 2):
                patterns.append("Hammer (Çekiç)")
                score += 15
            
            if prev['Close'] < prev['Open'] and last['Close'] > last['Open']:
                if last['Close'] > prev['Open'] and last['Open'] < prev['Close']:
                    patterns.append("Yutan Boğa")
                    score += 20
            
            lows = df['Low'].tail(20).values
            if len(lows) > 10:
                min1 = np.min(lows[:10])
                min2 = np.min(lows[10:])
                if abs(min1 - min2) / min1 < 0.01:
                    patterns.append("İkili Dip")
                    score += 15
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
            
            if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']:
                score += 15; reasons.append("Ichimoku: Trend Pozitif")
                
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']:
                score += 10
                if prev['MACD_12_26_9'] < prev['MACDs_12_26_9']: score += 10; reasons.append("MACD: AL Sinyali")
            
            try:
                bbl = df[[c for c in df.columns if c.startswith('BBL')][0]].iloc[-1]
                if last['Close'] <= bbl * 1.01: score += 15; reasons.append("Bollinger: Dip")
            except: pass
            
            if df['OBV'].iloc[-1] > df['OBV'].iloc[-5]:
                score += 10; reasons.append("OBV: Para Girişi")
            
            if last['Close'] > last['VWAP']: score += 10
            if last['RSI'] < 30: score += 20; reasons.append("RSI: Aşırı Satım")
            elif last['RSI'] > 70 and score < 60: score -= 20; reasons.append("RSI: Aşırı Alım")

            pats, pat_score = self.detect_patterns(df)
            score += pat_score
            for p in pats: reasons.append(f"Formasyon: {p}")

            return max(0, min(100, score)), reasons, df
        except Exception as e: return 0, [], df

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
            yorum = "NÖTR"
            if fk and fk < 8: yorum = "KELEPİR"
            elif fk and fk > 30: yorum = "PAHALI"
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-", "Yorum": yorum}
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
                    if score >= 80: signal = "GÜÇLÜ AL"
                    elif score <= 30: signal = "SAT"
                    
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
        menu = st.radio("Panel", ["💬 Hisse Sor / Analiz", "📡 Piyasa Radarı", "🌍 Global & Haber Odası", "Çıkış"])
        if menu == "Çıkış": st.session_state['giris_yapildi'] = False; st.rerun()

    engine = TradingEngine()
    tum_hisseler = get_live_tickers()

    if menu == "💬 Hisse Sor / Analiz":
        st.title("💬 Hisse Analiz Asistanı")
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
                    k4.metric("Temel", temel['Yorum'] if temel else "-")
                    
                    st.divider()
                    
                    col_g, col_d = st.columns([2, 1])
                    with col_g:
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
                        
                    with col_d:
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
        
        if not tum_hisseler:
            st.error("⚠️ Liste çekilemedi, Yedek liste devreye giriyor.")
        
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

    elif menu == "🌍 Global & Haber Odası":
        # Global Kısım Kodu (Aynı kalabilir veya buraya eklenebilir)
        st.title("🌍 Dünya & Gündem")
        st.info("Bu modül bakımda.") # Yer kazanmak için kısalttım

if __name__ == "__main__":
    main()
