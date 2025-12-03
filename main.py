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

# --- YEDEK LİSTE (KURTARICI) ---
def get_backup_list():
    return [
        "A1CAP", "ACSEL", "ADEL", "ADESE", "AEFES", "AFYON", "AGESA", "AGHOL", "AGROT", "AGYO",
        "AHGAZ", "AKBNK", "AKCNS", "AKENR", "AKFGY", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN",
        "AKSGY", "AKSUE", "AKYHO", "ALARK", "ALBRK", "ALCAR", "ALCTL", "ALFAS", "ALGYO", "ALKA",
        "ALKIM", "ALMAD", "ALTNY", "ALVES", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARCLK",
        "ARDYZ", "ARENA", "ARSAN", "ARTMS", "ARZUM", "ASELS", "ASGYO", "ASTOR", "ASUZU", "ATAGY",
        "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATSYH", "AVGYO", "AVHOL", "AVOD", "AVPGY", "AVTUR",
        "AYCES", "AYDEM", "AYEN", "AYES", "AYGAZ", "AZTEK", "BAGFS", "BAKAB", "BALAT", "BANVT",
        "BARMA", "BASCM", "BASGZ", "BAYRK", "BEGYO", "BERA", "BEYAZ", "BFREN", "BIENY", "BIGCH",
        "BIMAS", "BINHO", "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BMSCH", "BMSTL", "BNTAS", "BOBET",
        "BORLS", "BOSSA", "BRISA", "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BRSAN", "BRYAT",
        "BSOKE", "BTCIM", "BUCIM", "BURCE", "BURVA", "BVSAN", "BYDNR", "CANTE", "CATES", "CCOLA",
        "CELHA", "CEMAS", "CEMTS", "CEOEM", "CIMSA", "CLEBI", "CMBTN", "CMENT", "CONSE", "COSMO",
        "CRDFA", "CRFSA", "CUSAN", "CVKMD", "CWENE", "DAGHL", "DAGI", "DAPGM", "DARDL", "DATA",
        "DATES", "DDRKM", "DELEG", "DEMISA", "DERHL", "DERIM", "DESA", "DESPC", "DEVA", "DGATE",
        "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMSAS", "DNISI", "DOAS", "DOBUR", "DOCO", "DOGUB",
        "DOHOL", "DOKTA", "DURDO", "DYOBY", "DZGYO", "EBEBK", "ECILC", "EPLAS", "ECZYT", "EDATA",
        "EDIP", "EGEEN", "EGEPO", "EGGUB", "EGPRO", "EGSER", "EKGYO", "EKIZ", "EKSUN", "ELITE",
        "EMKEL", "EMNIS", "ENJSA", "ENKAI", "ENSRI", "ENTRA", "ENVER", "EPLAS", "ERBOS", "ERCB",
        "EREGL", "ERSU", "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUHOL", "EUKYO", "EUPWR",
        "EUREN", "EUYO", "FADE", "FENE", "FLAP", "FMIZP", "FONET", "FORMT", "FORTE", "FRIGO",
        "FROTO", "FZLGY", "GARAN", "GARFA", "GEDIK", "GEDZA", "GENIL", "GENTS", "GEREL", "GESAN",
        "GLBMD", "GLCVY", "GLRYH", "GLYHO", "GMTAS", "GOKNR", "GOLTS", "GOODY", "GOZDE", "GRNYO",
        "GRSEL", "GSDDE", "GSDHO", "GSRAY", "GUBRF", "GWIND", "GZNMI", "HALKB", "HATEK", "HDFGS",
        "HEDEF", "HEKTS", "HKTM", "HLGYO", "HRKET", "HTTBT", "HUBVC", "HUNER", "HURGZ", "ICBCT",
        "IDEAS", "IDGYO", "IEYHO", "IHAAS", "IHEVA", "IHGZT", "IHLAS", "IHLGM", "IHYAY", "IMASM",
        "INDES", "INFO", "INGRM", "INTEM", "INVEO", "INVES", "ISATR", "ISBIR", "ISBTR", "ISCTR",
        "ISDMR", "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISKUR", "ISMEN", "ISSEN", "ISYAT", "ITTFH",
        "IZENR", "IZFAS", "IZINV", "IZMDC", "JANTS", "KAPLM", "KARYE", "KARSN", "KARTN", "KARYE",
        "KATMR", "KAYSE", "KCAER", "KCMKW", "KDOAS", "KFEIN", "KGYO", "KBORU", "KIMMR", "KLGYO",
        "KLKIM", "KLMSN", "KLNMA", "KLRHO", "KLSYN", "KMPUR", "KNFRT", "KONKA", "KONTR", "KONYA",
        "KOPOL", "KORDS", "KOZAA", "KOZAL", "KRDMA", "KRDMB", "KRDMD", "KRGYO", "KRONT", "KRPLS",
        "KRSTL", "KRTEK", "KRVGD", "KSTUR", "KTLEV", "KTSKR", "KUTPO", "KUVVA", "KUYAS", "KZBGY",
        "KZGYO", "LIDER", "LIDFA", "LINK", "LKMNH", "LOGO", "LRSHO", "LUKSK", "MAALT", "MACKO",
        "MAGEN", "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA", "MARTI", "MAVI", "MEDTR", "MEGAP",
        "MEGMT", "MEKAG", "MNDRS", "MENBA", "MERCN", "MERIT", "MERKO", "METUR", "MGROS", "MIATK",
        "MIPAZ", "MMCAS", "MNDTR", "MOBTL", "MOGAN", "MONDU", "MPARK", "MRGYO", "MRSHL", "MSGYO",
        "MTRKS", "MTRYO", "MUNDA", "NATA", "NETAS", "NIBAS", "NTGAZ", "NTHOL", "NUGYO", "NUHCM",
        "OBAMS", "OBASE", "ODAS", "ODINE", "OFSYM", "ONCSM", "ORCAY", "ORGE", "ORMA", "OSMEN",
        "OSTIM", "OTKAR", "OTTO", "OYAKC", "OYAYO", "OYLUM", "OYYAT", "OZGYO", "OZKGY", "OZRDN",
        "OZSUB", "PAGYO", "PAMEL", "PAPIL", "PARSN", "PASEU", "PCILT", "PEGYO", "PEKGY", "PENGD",
        "PENTA", "PETKM", "PETUN", "PGSUS", "PINSU", "PKART", "PKENT", "PLAT", "PNLSN", "PNSUT",
        "POLHO", "POLTK", "PRDGS", "PRKAB", "PRKME", "PRZMA", "PSDTC", "PSGYO", "QNBFB", "QNBFL",
        "QUAGR", "RALYH", "RAYSG", "RNPOL", "REEDR", "RHEAG", "RODRG", "ROYAL", "RTALB", "RUBNS",
        "RYGYO", "RYSAS", "SAFKR", "SAHOL", "SAMAT", "SANEL", "SANFM", "SANKO", "SARKY", "SARTN",
        "SASA", "SAYAS", "SDTTR", "SEKFK", "SEKUR", "SELEC", "SELGD", "SELVA", "SEYKM", "SILVR",
        "SISE", "SKBNK", "SKTAS", "SMART", "SMRTG", "SNAET", "SNPAM", "SNGYO", "SNKRN", "SOKE",
        "SOKM", "SONME", "SRVGY", "SUMAS", "SUNGW", "SURGY", "SUWEN", "TABGD", "TARKM", "TATEN",
        "TATGD", "TAVHL", "TBORG", "TCELL", "TDGYO", "TEKTU", "TERA", "TETMT", "TEZOL", "TGSAS",
        "THYAO", "TKFEN", "TKNSA", "TLMAN", "TMPOL", "TMSN", "TNZTP", "TOASO", "TRCAS", "TRGYO",
        "TRILC", "TSGYO", "TSKB", "TSPOR", "TTKOM", "TTRAK", "TUCLK", "TUKAS", "TUPRS", "TUREX",
        "TURGG", "TURSG", "UFUK", "ULAS", "ULKER", "ULUFA", "ULUSE", "ULUUN", "UMPAS", "UNLU",
        "USAK", "UZERB", "VAKBN", "VAKFN", "VAKKO", "VANGD", "VBTYZ", "VERUS", "VESBE", "VESTL",
        "VKFYO", "VKGYO", "VKING", "VRGYO", "YAPRK", "YATAS", "YAYLA", "YEOTK", "YESIL", "YGGYO",
        "YGYO", "YKBNK", "YKSLN", "YONGA", "YUNSA", "YYAPI", "YYLGD", "ZEDUR", "ZOREN", "ZRGYO"
    ]

# --- 2. CANLI LİSTE MOTORU (HATASIZ) ---
@st.cache_data(ttl=600)
def get_live_tickers():
    canli_liste = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.content, 'html.parser')
        table = soup.find('table', {'id': 'tableHisseOnerileri'})
        if table:
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if cols: canli_liste.append(cols[0].find('a').text.strip())
    except: pass
    
    if len(canli_liste) < 50:
        return sorted(list(set(get_backup_list())))
    return sorted(list(set(canli_liste)))

# --- 3. ÇOKLU CANLI FİYAT MOTORU ---
def get_realtime_price(ticker):
    clean_ticker = ticker.replace('.IS', '')
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{clean_ticker}-detay/"
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.content, "html.parser")
        fiyat = soup.select_one('.price-arrow-down, .price-arrow-up, .text-2')
        if fiyat: return float(fiyat.text.strip().replace(',', '.'))
    except: pass
    return None

# --- 4. GLOBAL VE HABER MOTORU (BİRLEŞİK) ---
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

    def analyze_news(self, query_type="GENEL", ticker=""):
        sentiment = 0
        news_display = []
        
        if query_type == "HISSE":
            feeds = [f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr"]
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
                            if hasattr(entry, 'published_parsed'):
                                news_date = datetime(*entry.published_parsed[:6])
                                today = datetime.now()
                                if (today - news_date).days <= 7:
                                    date_str = news_date.strftime("%d.%m")
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
        return max(-20, min(20, sentiment)), unique[:15]

# --- 5. TEKNİK ANALİZ MOTORU ---
class TechnicalEngine:
    def detect_patterns(self, df):
        patterns = []
        score = 0
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            body = abs(last['Close'] - last['Open'])
            wick_lower = min(last['Close'], last['Open']) - last['Low']
            if wick_lower > (body * 2): patterns.append("Çekiç"); score += 15
            if prev['Close'] < prev['Open'] and last['Close'] > last['Open']:
                if last['Close'] > prev['Open'] and last['Open'] < prev['Close']: patterns.append("Yutan Boğa"); score += 20
        except: pass
        return patterns, score

    def analyze(self, df):
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            bb = ta.bbands(df['Close'], length=20)
            if bb is not None: df = pd.concat([df, bb], axis=1)
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            if kc is not None: df = pd.concat([df, kc], axis=1)
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
            df = pd.concat([df, ichimoku], axis=1)
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

            last = df.iloc[-1]
            prev = df.iloc[-2]
            if pd.isna(last['RSI']): return 0, [], df

            score = 50
            reasons = []
            
            if last['Close'] > last['VWAP']: score += 10; reasons.append("Fiyat VWAP Üzerinde")
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 15; reasons.append("MACD Al")
            if last['RSI'] < 30: score += 20; reasons.append("RSI Dip")
            elif last['RSI'] > 70: score -= 15; reasons.append("RSI Tepe")
            
            if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']: score += 15; reasons.append("Ichimoku Bulutu Üstünde")
            
            pats, p_score = self.detect_patterns(df)
            score += p_score
            for p in pats: reasons.append(f"Formasyon: {p}")

            return max(0, min(100, score)), reasons, df
        except: return 0, [], df

# --- 6. ANA MOTOR ---
class TradingEngine:
    def __init__(self):
        self.tech = TechnicalEngine()
        self.intel = GlobalIntel()

    def get_fundamentals(self, ticker):
        try:
            stock = yf.Ticker(f"{ticker}.IS")
            info = stock.info
            return {"FK": info.get('trailingPE'), "PD_DD": info.get('priceToBook')}
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

            is_live = False
            if mode == "PRO":
                live_price = get_realtime_price(ticker)
                if live_price:
                    if abs(live_price - df.iloc[-1]['Close']) / df.iloc[-1]['Close'] < 0.20:
                        df.iloc[-1, df.columns.get_loc('Close')] = live_price
                        df.iloc[-1, df.columns.get_loc('High')] = max(live_price, df.iloc[-1]['High'])
                        df.iloc[-1, df.columns.get_loc('Low')] = min(live_price, df.iloc[-1]['Low'])
                        is_live = True

            tech_score, tech_reasons, processed_df = self.tech.analyze(df)
            news_score, news_list = self.intel.analyze_news("HISSE", ticker)
            fund = self.get_fundamentals(ticker)
            
            final_score = max(0, min(100, tech_score + news_score))
            signal = "NÖTR"; color = "gray"
            if final_score >= 80: signal = "GÜÇLÜ AL 🚀"; color = "green"
            elif final_score >= 60: signal = "AL 🌱"; color = "blue"
            elif final_score <= 30: signal = "SAT 🔻"; color = "red"
            
            last = processed_df.iloc[-1]
            return {
                "Hisse": ticker, "Fiyat": last['Close'], "Skor": int(final_score),
                "Sinyal": signal, "Renk": color, "RSI": last['RSI'],
                "Stop": last['Close'] - (last['ATR']*1.5), "Hedef": last['Close'] + (last['ATR']*3.0),
                "Yorumlar": tech_reasons, "Haberler": news_list, "Temel": fund, "Data": processed_df, "Is_Live": is_live
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
            with st.spinner(f"{sembol} analiz ediliyor..."):
                res = engine.analyze(sembol, mode="PRO")
                if res:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL", delta="Canlı" if res['Is_Live'] else "Gecikmeli")
                    k2.metric("Skor", f"{res['Skor']}/100")
                    k3.metric("Karar", res['Sinyal'])
                    fk_val = res['Temel']['FK'] if res['Temel'] and res['Temel']['FK'] else "-"
                    k4.metric("Temel (F/K)", fk_val)
                    st.divider()
                    
                    col_g, col_d = st.columns([2, 1])
                    with col_g:
                        df = res['Data']
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat"))
                        
                        # Dinamik Kolon Bulma
                        bbu = next((c for c in df.columns if c.startswith('BBU')), None)
                        if bbu: fig.add_trace(go.Scatter(x=df.index, y=df[bbu], line=dict(color='gray', dash='dot'), name='Bollinger'))
                        kcu = next((c for c in df.columns if c.startswith('KCU')), None)
                        if kcu: fig.add_trace(go.Scatter(x=df.index, y=df[kcu], line=dict(color='purple'), name='Keltner'))
                        
                        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='orange'), name='VWAP'))
                        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, title=f"Veri Zamanı: {res['Tarih']}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_d:
                        st.subheader("📝 Tespitler")
                        for y in res['Yorumlar']: st.markdown(f"✅ {y}")
                        st.info(f"Hedef: {res['Hedef']:.2f}")
                        st.error(f"Stop: {res['Stop']:.2f}")
                    
                    st.markdown("---")
                    st.subheader("📰 Haberler")
                    if res['Haberler']:
                        for n in res['Haberler']:
                            st.markdown(f"🔹 **[{n['Title']}]({n['Link']})** ({n['Date']})")
                    else: st.info("Haber yok.")
                else: st.error("Hisse bulunamadı.")

    elif menu == "📡 Piyasa Radarı":
        st.title("📡 MERTT Piyasa Radarı")
        st.info(f"Takipteki Hisse: {len(tum_hisseler)}")
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
                st.dataframe(df.style.format({"Fiyat": "{:.2f}", "RSI": "{:.0f}"}).background_gradient(subset=['Skor'], cmap='RdYlGn'), use_container_width=True)
            else: st.warning("Sinyal yok.")

    elif menu == "🌍 Global & Haber Odası":
        st.title("🌍 Dünya Piyasaları & Gündem")
        indices = intel.get_global_indices()
        if indices:
            cols = st.columns(len(indices))
            for i, (name, data) in enumerate(indices.items()):
                cols[i].metric(label=name, value=f"{data['Fiyat']:.2f}", delta=f"%{data['Degisim']:.2f}")
        st.divider()
        _, news_list = intel.analyze_news("GENEL") 
        if news_list:
            for n in news_list:
                st.markdown(f"#### 📰 [{n['Title']}]({n['Link']})")
                st.caption(f"🗓️ {n['Date']}")
                st.write("---")

if __name__ == "__main__":
    main()
