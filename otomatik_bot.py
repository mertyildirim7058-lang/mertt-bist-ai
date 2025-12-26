import sys
import time
import random
import os
import concurrent.futures
import numpy as np

# --- KÜTÜPHANE KONTROLÜ ---
try:
    import yfinance as yf
    import pandas as pd
    import pandas_ta as ta
    import requests
    import feedparser
    from bs4 import BeautifulSoup
    import xgboost as xgb
except ImportError as e:
    print(f"Kütüphane eksik: {e}")
    sys.exit(0)

# --- AYARLAR ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- İNSAN TAKLİDİ (Headers) ---
def get_random_header():
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
    ]
    return {'User-Agent': random.choice(user_agents)}

def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
            requests.post(url, json=payload)
        except: pass

# --- 1. DEVASA YEDEK LİSTE (BIST TÜM) ---
def get_full_bist_list():
    """
    İş Yatırım çalışmazsa devreye girecek 580+ Hisselik Liste.
    """
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

# --- 2. LİSTE ÇEKME MANTIĞI ---
def get_scan_list():
    canli_liste = []
    try:
        # Önce Canlıyı Dene
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        r = requests.get(url, headers=get_random_header(), timeout=10)
        soup = BeautifulSoup(r.content, 'html.parser')
        table = soup.find('table', {'id': 'tableHisseOnerileri'})
        if table:
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if cols: canli_liste.append(cols[0].find('a').text.strip())
    except: pass
    
    # Eğer canlı liste boşsa, YEDEK LİSTEYİ kullan
    if len(canli_liste) < 50:
        return sorted(list(set(get_full_bist_list())))
        
    return sorted(list(set(canli_liste)))

# --- 3. ANLIK FİYAT (CANLI YAMA) ---
def get_realtime_price(ticker):
    """
    İş Yatırım / BigPara üzerinden anlık fiyat çeker.
    Bu sayede Yahoo'nun 15dk gecikmesini kapatırız.
    """
    # Bot olduğu anlaşılmasın diye rastgele bekle
    time.sleep(random.uniform(1.0, 2.5))
    try:
        clean = ticker.replace('.IS', '')
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{clean}-detay/"
        r = requests.get(url, headers=get_random_header(), timeout=5)
        soup = BeautifulSoup(r.content, "html.parser")
        
        # Fiyatı bul
        price_span = soup.find("span", {"class": "text-2"})
        if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
        
        if price_span:
            return float(price_span.text.strip().replace(',', '.'))
    except: pass
    return None

# --- 4. HABER DUYGU ANALİZİ ---
class NewsEngine:
    def analyze_sentiment(self, ticker):
        score = 0
        try:
            # RSS ile Google News Tara
            url = f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr"
            feed = feedparser.parse(url)
            
            pos = ['kar', 'büyüme', 'rekor', 'onay', 'temettü', 'anlaşma', 'geri alım']
            neg = ['zarar', 'düşüş', 'ceza', 'iptal', 'kriz', 'soruşturma']
            
            for entry in feed.entries[:3]: # Son 3 haber
                title = entry.title.lower()
                for w in pos: 
                    if w in title: score += 5
                for w in neg: 
                    if w in title: score -= 5
        except: pass
        return max(-20, min(20, score))

# --- 5. HİBRİT ANALİZ (ANA BEYİN) ---
def analyze_hybrid(ticker, history_data):
    """
    Yahoo verisiyle Teknik Analiz yapar, 
    Potansiyel varsa Canlı Fiyatı çeker,
    En son Haberlere bakar.
    """
    news_engine = NewsEngine()
    try:
        # A. Veriyi Al
        try: df = history_data[f"{ticker}.IS"].copy()
        except: return None
        
        if df.empty or df['Close'].isnull().all(): return None
        df = df.dropna()
        if len(df) < 50: return None # Yeni arzları ele
        
        # B. ÖN ELEME (Hız İçin)
        current_rsi = ta.rsi(df['Close'], length=14).iloc[-1]
        if current_rsi > 40 and current_rsi < 70: return None 
        
        # C. CANLI FİYAT YAMASI (Sadece Adaylar İçin)
        live_price = get_realtime_price(ticker)
        if live_price and live_price > 0:
            last_yahoo = df['Close'].iloc[-1]
            if abs(live_price - last_yahoo) / last_yahoo < 0.20:
                df.iloc[-1, df.columns.get_loc('Close')] = live_price
        
        # D. TEKNİK ANALİZ
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None: df = pd.concat([df, bb], axis=1)
        
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        
        last = df.iloc[-1]
        
        # E. PUANLAMA
        score = 50
        reasons = []
        
        # Teknik Puanlar
        if last['Close'] > last['VWAP']: score += 10
        if last['MACD_12_26_9'] > last['MACDs_12_26_9']: 
            score += 15
            reasons.append("MACD Al")
            
        if last['RSI'] < 30: 
            score += 25
            reasons.append(f"RSI Dip ({last['RSI']:.0f})")
        elif last['RSI'] > 75: 
            score -= 20
            
        # Bollinger Dip
        bbl = next((c for c in df.columns if c.startswith('BBL')), None)
        if bbl and last['Close'] <= last[bbl] * 1.01:
            score += 20
            reasons.append("Bollinger Dip")
            
        # F. HABER ANALİZİ
        if score >= 70:
            n_score = news_engine.analyze_sentiment(ticker)
            score += n_score
            if n_score > 0: reasons.append("Haber+")
            
        score = max(0, min(100, score))
        
        # G. KARAR
        if score >= 85:
            return {
                "Hisse": ticker,
                "Fiyat": last['Close'],
                "Skor": int(score),
                "Neden": ", ".join(reasons)
            }
            
    except: return None
    return None

# --- ANA DÖNGÜ ---
def main():
    print("🚀 Bot Başlatılıyor...")
    tickers = get_scan_list() # Yedekli Liste
    print(f"📋 {len(tickers)} hisse taranacak.")
    
    firsatlar = []
    
    # 30'lu Paketler
    chunk_size = 30
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        symbols = [f"{t}.IS" for t in chunk]
        try:
            # Toplu İndirme
            data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
            
            # Analiz
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(analyze_hybrid, t, data): t for t in chunk}
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res: firsatlar.append(res)
                    
        except Exception as e: print(f"Hata: {e}")
        
        # Paket arası bekleme
        time.sleep(random.uniform(3.0, 6.0))
        
    # Raporlama
    if firsatlar:
        firsatlar.sort(key=lambda x: x['Skor'], reverse=True)
        top = firsatlar[:8]
        
        msg = "🦅 **MERTT AI: Canlı Fırsatlar** 🦅\n\n"
        for f in top:
            msg += f"🟢 *{f['Hisse']}* | {f['Fiyat']:.2f} TL\n"
            msg += f"   └ Skor: {f['Skor']} | {f['Neden']}\n\n"
            
        send_telegram(msg)
    else:
        print("Fırsat bulunamadı.")

if __name__ == "__main__":
    main()
