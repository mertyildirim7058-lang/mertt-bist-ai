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
                    else: st.error("⛔ Yetkisiz Erişim!")
                except: st.error("Sistem Hatası: Şifre tanımlı değil.")
        return False
    return True

if not guvenlik_kontrolu(): st.stop()

# --- YEDEK TAM LİSTE ---
def get_backup_list():
    return [
        "A1CAP", "ACSEL", "ADEL", "ADESE", "ADGYO", "AEFES", "AFYON", "AGESA", "AGHOL", "AGROT", "AGYO", "AHGAZ", "AKBNK", "AKCNS", "AKENR", "AKFGY", "AKFYE", "AKGRT", "AKMGY", "AKSA", "AKSEN", "AKSGY", "AKSUE", "AKYHO", "ALARK", "ALBRK", "ALCAR", "ALCTL", "ALFAS", "ALGYO", "ALKA", "ALKIM", "ALMAD", "ALTNY", "ALVES", "ANELE", "ANGEN", "ANHYT", "ANSGR", "ARASE", "ARCLK", "ARDYZ", "ARENA", "ARSAN", "ARTMS", "ARZUM", "ASELS", "ASGYO", "ASTOR", "ASUZU", "ATAGY", "ATAKP", "ATATP", "ATEKS", "ATLAS", "ATSYH", "AVGYO", "AVHOL", "AVOD", "AVPGY", "AVTUR", "AYCES", "AYDEM", "AYEN", "AYES", "AYGAZ", "AZTEK", "BABA", "BAGFS", "BAKAB", "BALAT", "BANVT", "BARMA", "BASCM", "BASGZ", "BAYRK", "BEGYO", "BERA", "BEYAZ", "BFREN", "BIENY", "BIGCH", "BIMAS", "BINHO", "BIOEN", "BIZIM", "BJKAS", "BLCYT", "BMSCH", "BMSTL", "BNTAS", "BOBET", "BORLS", "BOSSA", "BRISA", "BRKO", "BRKSN", "BRKVY", "BRLSM", "BRMEN", "BRSAN", "BRYAT", "BSOKE", "BTCIM", "BUCIM", "BURCE", "BURVA", "BVSAN", "BYDNR", "CANTE", "CATES", "CCOLA", "CELHA", "CEMAS", "CEMTS", "CEOEM", "CIMSA", "CLEBI", "CMBTN", "CMENT", "CONSE", "COSMO", "CRDFA", "CRFSA", "CUSAN", "CVKMD", "CWENE", "DAGHL", "DAGI", "DAPGM", "DARDL", "DATA", "DATES", "DDRKM", "DELEG", "DEMISA", "DERHL", "DERIM", "DESA", "DESPC", "DEVA", "DGATE", "DGGYO", "DGNMO", "DIRIT", "DITAS", "DMSAS", "DNISI", "DOAS", "DOBUR", "DOCO", "DOGUB", "DOHOL", "DOKTA", "DURDO", "DYOBY", "DZGYO", "EBEBK", "ECILC", "EPLAS", "ECZYT", "EDATA", "EDIP", "EGEEN", "EGEPO", "EGGUB", "EGPRO", "EGSER", "EKGYO", "EKIZ", "EKSUN", "ELITE", "EMKEL", "EMNIS", "ENJSA", "ENKAI", "ENSRI", "ENTRA", "ENVER", "EPLAS", "ERBOS", "ERCB", "EREGL", "ERSU", "ESCAR", "ESCOM", "ESEN", "ETILR", "ETYAT", "EUHOL", "EUKYO", "EUPWR", "EUREN", "EUYO", "FADE", "FENE", "FLAP", "FMIZP", "FONET", "FORMT", "FORTE", "FRIGO", "FROTO", "FZLGY", "GARAN", "GARFA", "GEDIK", "GEDZA", "GENIL", "GENTS", "GEREL", "GESAN", "GLBMD", "GLCVY", "GLRYH", "GLYHO", "GMTAS", "GOKNR", "GOLTS", "GOODY", "GOZDE", "GRNYO", "GRSEL", "GSDDE", "GSDHO", "GSRAY", "GUBRF", "GWIND", "GZNMI", "HALKB", "HATEK", "HDFGS", "HEDEF", "HEKTS", "HKTM", "HLGYO", "HRKET", "HTTBT", "HUBVC", "HUNER", "HURGZ", "ICBCT", "IDEAS", "IDGYO", "IEYHO", "IHAAS", "IHEVA", "IHGZT", "IHLAS", "IHLGM", "IHYAY", "IMASM", "INDES", "INFO", "INGRM", "INTEM", "INVEO", "INVES", "ISATR", "ISBIR", "ISBTR", "ISCTR", "ISDMR", "ISFIN", "ISGSY", "ISGYO", "ISKPL", "ISKUR", "ISMEN", "ISSEN", "ISYAT", "ITTFH", "IZENR", "IZFAS", "IZINV", "IZMDC", "JANTS", "KAPLM", "KARYE", "KARSN", "KARTN", "KARYE", "KATMR", "KAYSE", "KCAER", "KCMKW", "KDOAS", "KFEIN", "KGYO", "KBORU", "KIMMR", "KLGYO", "KLKIM", "KLMSN", "KLNMA", "KLRHO", "KLSYN", "KMPUR", "KNFRT", "KONKA", "KONTR", "KONYA", "KOPOL", "KORDS", "KOZAA", "KOZAL", "KRDMA", "KRDMB", "KRDMD", "KRGYO", "KRONT", "KRPLS", "KRSTL", "KRTEK", "KRVGD", "KSTUR", "KTLEV", "KTSKR", "KUTPO", "KUVVA", "KUYAS", "KZBGY", "KZGYO", "LIDER", "LIDFA", "LINK", "LKMNH", "LOGO", "LRSHO", "LUKSK", "MAALT", "MACKO", "MAGEN", "MAKIM", "MAKTK", "MANAS", "MARBL", "MARKA", "MARTI", "MAVI", "MEDTR", "MEGAP", "MEGMT", "MEKAG", "MNDRS", "MENBA", "MERCN", "MERIT", "MERKO", "METUR", "MGROS", "MIATK", "MIPAZ", "MMCAS", "MNDTR", "MOBTL", "MOGAN", "MONDU", "MPARK", "MRGYO", "MRSHL", "MSGYO", "MTRKS", "MTRYO", "MUNDA", "NATA", "NETAS", "NIBAS", "NTGAZ", "NTHOL", "NUGYO", "NUHCM", "OBAMS", "OBASE", "ODAS", "ODINE", "OFSYM", "ONCSM", "ORCAY", "ORGE", "ORMA", "OSMEN", "OSTIM", "OTKAR", "OTTO", "OYAKC", "OYAYO", "OYLUM", "OYYAT", "OZGYO", "OZKGY", "OZRDN", "OZSUB", "PAGYO", "PAMEL", "PAPIL", "PARSN", "PASEU", "PCILT", "PEGYO", "PEKGY", "PENGD", "PENTA", "PETKM", "PETUN", "PGSUS", "PINSU", "PKART", "PKENT", "PLAT", "PNLSN", "PNSUT", "POLHO", "POLTK", "PRDGS", "PRKAB", "PRKME", "PRZMA", "PSDTC", "PSGYO", "QNBFB", "QNBFL", "QUAGR", "RALYH", "RAYSG", "RNPOL", "REEDR", "RHEAG", "RODRG", "ROYAL", "RTALB", "RUBNS", "RYGYO", "RYSAS", "SAFKR", "SAHOL", "SAMAT", "SANEL", "SANFM", "SANKO", "SARKY", "SARTN", "SASA", "SAYAS", "SDTTR", "SEKFK", "SEKUR", "SELEC", "SELGD", "SELVA", "SEYKM", "SILVR", "SISE", "SKBNK", "SKTAS", "SMART", "SMRTG", "SNAET", "SNPAM", "SNGYO", "SNKRN", "SOKE", "SOKM", "SONME", "SRVGY", "SUMAS", "SUNGW", "SURGY", "SUWEN", "TABGD", "TARKM", "TATEN", "TATGD", "TAVHL", "TBORG", "TCELL", "TDGYO", "TEKTU", "TERA", "TETMT", "TEZOL", "TGSAS", "THYAO", "TKFEN", "TKNSA", "TLMAN", "TMPOL", "TMSN", "TNZTP", "TOASO", "TRCAS", "TRGYO", "TRILC", "TSGYO", "TSKB", "TSPOR", "TTKOM", "TTRAK", "TUCLK", "TUKAS", "TUPRS", "TUREX", "TURGG", "TURSG", "UFUK", "ULAS", "ULKER", "ULUFA", "ULUSE", "ULUUN", "UMPAS", "UNLU", "USAK", "UZERB", "VAKBN", "VAKFN", "VAKKO", "VANGD", "VBTYZ", "VERUS", "VESBE", "VESTL", "VKFYO", "VKGYO", "VKING", "VRGYO", "YAPRK", "YATAS", "YAYLA", "YEOTK", "YESIL", "YGGYO", "YGYO", "YKBNK", "YKSLN", "YONGA", "YUNSA", "YYAPI", "YYLGD", "ZEDUR", "ZOREN", "ZRGYO"
    ]

# --- 3. CANLI LİSTE ---
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

# --- 2. CANLI FİYAT ---
def get_realtime_price(ticker):
    time.sleep(random.uniform(0.2, 0.5))
    try:
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{ticker.replace('.IS','')}-detay/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(resp.content, "html.parser")
        price_span = soup.find("span", {"class": "text-2"})
        if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
        if price_span: return float(price_span.text.strip().replace(',', '.'))
    except: return None

# --- 3. GLOBAL & HABER ---
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
            feeds = [f"https://news.google.com/rss/search?q={ticker}+hisse+kap+haberleri&hl=tr&gl=TR&ceid=TR:tr", f"https://news.google.com/rss/search?q={ticker}+borsa&hl=tr&gl=TR&ceid=TR:tr"]
        else:
            feeds = ["https://news.google.com/rss/search?q=Borsa+İstanbul+Gündem&hl=tr&gl=TR&ceid=TR:tr", "https://www.trthaber.com/xml/ekonomi.xml"]
            
        for url in feeds:
            try:
                r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=5)
                if r.status_code == 200:
                    feed = feedparser.parse(r.content)
                    for entry in feed.entries[:10]: 
                        title = entry.title.replace(" - Haberler", "")
                        link = entry.link
                        try:
                            if hasattr(entry, 'published_parsed'):
                                news_date = datetime(*entry.published_parsed[:6])
                                today = datetime.now()
                                # 15 Güne kadar haberleri al
                                if (today - news_date).days <= 15:
                                    date_str = news_date.strftime("%d.%m %H:%M")
                                    t_lower = title.lower()
                                    imp = "Nötr"; color = "gray"; score_delta = 0
                                    for w in self.tech: 
                                        if w in t_lower: score_delta += 2; imp="Pozitif"; color="green"
                                    for w in self.risk: 
                                        if w in t_lower: score_delta -= 3; imp="Negatif"; color="red"
                                    
                                    sentiment += score_delta
                                    # Gösterim (Bugün ve Dün)
                                    if (today - news_date).days <= 1:
                                        news_display.append({"Title": title, "Link": link, "Date": date_str, "Color": color})
                        except: pass
            except: pass
        return max(-20, min(20, sentiment)), news_display[:15]

# --- 4. ANALİZ MOTORU ---
class TradingEngine:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        self.intel = GlobalIntel()

    def get_fundamentals(self, ticker):
        try:
            info = yf.Ticker(f"{ticker}.IS").info
            fk = info.get('trailingPE', None)
            pddd = info.get('priceToBook', None)
            return {"FK": round(fk, 2) if fk else "-", "PD_DD": round(pddd, 2) if pddd else "-"}
        except: return None

    def calculate_fibonacci(self, df):
        """Fibonacci Destek/Direnç"""
        try:
            recent_high = df['High'].tail(60).max()
            recent_low = df['Low'].tail(60).min()
            diff = recent_high - recent_low
            
            levels = {
                "0.236": recent_high - 0.236 * diff,
                "0.382": recent_high - 0.382 * diff,
                "0.5": recent_high - 0.5 * diff,
                "0.618": recent_high - 0.618 * diff
            }
            return levels
        except: return {}

    def analyze(self, ticker, mode="PRO"):
        try:
            t = f"{ticker}.IS"
            # EMA 200 için 1 yıllık veri
            df = yf.download(t, period="1y", interval="60m", progress=False)
            if df is None or len(df) < 200: return None
            if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
            df = df.ffill().bfill()
            
            # TR Saati
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Istanbul')

            # Canlı Yama
            is_live = False
            if mode == "PRO":
                live = get_realtime_price(ticker)
                if live and abs(live - df.iloc[-1]['Close'])/df.iloc[-1]['Close'] < 0.2:
                    df.iloc[-1, df.columns.get_loc('Close')] = live
                    is_live = True

            # İndikatörler
            df['RSI'] = ta.rsi(df['Close'], 14)
            df['EMA_9'] = ta.ema(df['Close'], 9)
            df['EMA_200'] = ta.ema(df['Close'], 200)
            df = pd.concat([df, ta.macd(df['Close'])], axis=1)
            
            # Bollinger & Keltner
            bb = ta.bbands(df['Close'], 20)
            if bb is not None: df = pd.concat([df, bb], axis=1)
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            if kc is not None: df = pd.concat([df, kc], axis=1)
            
            # Ichimoku
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
            df = pd.concat([df, ichimoku], axis=1)
            
            # PSAR
            psar = ta.psar(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, psar], axis=1)
            
            # Hacim
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['VWAP'] = (df['Volume']*(df['High']+df['Low']+df['Close'])/3).cumsum()/df['Volume'].cumsum()
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], 14)

            last = df.iloc[-1]
            if pd.isna(last['RSI']): return None

            # --- PUANLAMA ---
            score = 50
            reasons = []
            
            # 1. Trend (EMA 200 & 9)
            if last['Close'] > last['EMA_200']: 
                score += 10; reasons.append("Fiyat EMA 200 Üstünde (Uzun Vade Boğa)")
            else: score -= 20 # Ayı piyasası
            
            if last['Close'] > last['EMA_9']: score += 5

            # 2. Ichimoku
            if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']:
                score += 15; reasons.append("Ichimoku Bulut Üstü (Güçlü Trend)")
            
            # 3. Fibonacci (Destek Dönüşü)
            fibs = self.calculate_fibonacci(df)
            if fibs:
                # Eğer fiyat 0.618 veya 0.5 seviyesine %1 yakınsa ve yükseliyorsa
                if abs(last['Close'] - fibs['0.618'])/fibs['0.618'] < 0.01:
                    score += 15; reasons.append("Fibonacci 0.618 Desteğinden Tepki")
            
            # 4. PSAR
            psar_col = next((c for c in df.columns if c.startswith('PSAR')), None)
            if psar_col and df[psar_col].iloc[-1] < last['Close']: score += 10

            # 5. Osilatörler
            if last['MACD_12_26_9'] > last['MACDs_12_26_9']: score += 10; reasons.append("MACD Al")
            if last['RSI'] < 30: score += 20; reasons.append("RSI Dip")
            elif last['RSI'] > 75: score -= 15
            if df['OBV'].iloc[-1] > df['OBV'].iloc[-5]: score += 5

            # 6. Haberler
            n_sc, n_lst = 0, []
            if mode == "PRO":
                n_sc, n_lst = self.intel.analyze_news("HISSE", ticker)
                score += n_sc
                if n_sc > 0: reasons.append("Haberler Pozitif")

            score = max(0, min(100, score))
            signal, color = "NÖTR", "gray"
            if score >= 80: signal, color = "GÜÇLÜ AL 🚀", "green"
            elif score >= 60: signal, color = "AL 🌱", "blue"
            elif score <= 30: signal, color = "SAT 🔻", "red"

            stop = last['Close'] - (last['ATR']*1.5)
            hedef = last['Close'] + (last['ATR']*3.0)
            temel = self.get_fundamentals(ticker)

            return {
                "Hisse": ticker, "Fiyat": last['Close'], "Skor": int(score),
                "Sinyal": signal, "Renk": color, "RSI": last['RSI'],
                "Stop": stop, "Hedef": hedef, "Yorumlar": reasons, "Haberler": n_lst, 
                "Data": df, "Tarih": df.index[-1].strftime('%d %B %H:%M'),
                "Is_Live": is_live, "Temel": temel
            }
        except: return None

    def analyze_batch(self, tickers):
        res = []
        try:
            d = yf.download([f"{t}.IS" for t in tickers], period="6mo", interval="60m", group_by='ticker', progress=False, threads=True)
            for t in tickers:
                try:
                    df = d[f"{t}.IS"].dropna()
                    if len(df)<100: continue
                    rsi = ta.rsi(df['Close'], 14).iloc[-1]
                    ema200 = ta.ema(df['Close'], 200).iloc[-1]
                    last = df['Close'].iloc[-1]
                    
                    sc = 50
                    if last > ema200: sc += 10
                    else: sc -= 30 # Düşüş trendindekileri ele
                    
                    if rsi < 40: sc += 30
                    
                    if sc >= 80: 
                        res.append({"Hisse":t, "Fiyat":last, "Sinyal":"GÜÇLÜ AL 🚀", "RSI":rsi, "Skor":sc})
                except: continue
        except: pass
        return res

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
        st.title("🤖 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu:", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("ANALİZ ET 🔍", type="primary")

        if btn and sembol:
            with st.spinner("EMA 200, Fibonacci ve Haberler taranıyor..."):
                res = engine.analyze(sembol)
                if res:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL", delta="Canlı" if res['Is_Live'] else "Gecikmeli")
                    k2.metric("Skor", f"{res['Skor']}/100")
                    k3.metric("Karar", res['Sinyal'])
                    fk = res['Temel']['FK'] if res['Temel'] else "-"
                    k4.metric("F/K", fk)
                    st.divider()
                    
                    g, d = st.columns([2, 1])
                    with g:
                        df = res['Data']
                        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Fiyat")])
                        
                        # İndikatörleri Ekle
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='blue', width=2), name='EMA 200'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], line=dict(color='cyan', width=1), name='EMA 9', visible='legendonly'))
                        
                        bbu = next((c for c in df.columns if c.startswith('BBU')), None)
                        if bbu: fig.add_trace(go.Scatter(x=df.index, y=df[bbu], line=dict(color='gray', dash='dot'), name='Bollinger'))
                        
                        psar_col = next((c for c in df.columns if c.startswith('PSAR')), None)
                        if psar_col: fig.add_trace(go.Scatter(x=df.index, y=df[psar_col], mode='markers', name='PSAR'))

                        fig.update_layout(template="plotly_dark", height=500, title=f"Son Veri: {res['Tarih']}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with d:
                        if res['Renk']=='green': st.success(f"**{res['Sinyal']}**")
                        else: st.warning(f"**{res['Sinyal']}**")
                        st.info(f"Hedef: {res['Hedef']:.2f}")
                        st.error(f"Stop: {res['Stop']:.2f}")
                        for y in res['Yorumlar']: st.markdown(f"✅ {y}")

                    if res['Haberler']:
                        st.markdown("### 📰 Haber Akışı")
                        for n in res['Haberler']:
                            col = "🟢" if n['Color']=="green" else "🔴" if n['Color']=="red" else "⚪"
                            st.markdown(f"{col} **[{n['Title']}]({n['Link']})** ({n['Date']})")

                else: st.error("Veri yok.")

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

    elif menu == "🌍 Global & Haber Odası":
        st.title("🌍 Piyasalar")
        idx = intel.get_global_indices()
        c = st.columns(len(idx))
        for i, (n, d) in enumerate(idx.items()): c[i].metric(n, f"{d['Fiyat']:.2f}", f"%{d['Degisim']:.2f}")
        st.divider()
        _, nws = intel.analyze_news("GENEL")
        for n in nws: st.markdown(f"#### 📰 [{n['Title']}]({n['Link']})")

if __name__ == "__main__":
    main()
