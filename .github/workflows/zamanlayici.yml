import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import feedparser
from bs4 import BeautifulSoup
import os
import time
import random
import numpy as np

# --- AYARLAR ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- İNSAN TAKLİDİ İÇİN BAŞLIKLAR ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1'
]

def get_random_header():
    return {'User-Agent': random.choice(USER_AGENTS)}

def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
            requests.post(url, json=payload)
        except: pass

# --- 1. CANLI LİSTE ---
def get_live_tickers():
    canli_liste = []
    try:
        url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
        # Liste çekerken de insan gibi davranalım
        time.sleep(2)
        r = requests.get(url, headers=get_random_header(), timeout=10)
        soup = BeautifulSoup(r.content, 'html.parser')
        table = soup.find('table', {'id': 'tableHisseOnerileri'})
        if table:
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if cols: canli_liste.append(cols[0].find('a').text.strip())
    except: pass
    
    if len(canli_liste) < 10:
        return ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS"]
    return sorted(list(set(canli_liste)))

# --- 2. ANLIK FİYAT (İŞ YATIRIM) ---
def get_realtime_price(ticker):
    """
    Sadece potansiyel hisseler için çalışır.
    İş Yatırım'dan anlık fiyat çeker.
    """
    # İstenilen Gecikme: 2 ile 5 saniye arası rastgele bekle (Ban yememek için)
    delay = random.uniform(2.0, 5.0)
    time.sleep(delay)
    
    try:
        clean_ticker = ticker.replace('.IS', '')
        url = f"https://bigpara.hurriyet.com.tr/borsa/hisse-fiyatlari/{clean_ticker}-detay/"
        
        resp = requests.get(url, headers=get_random_header(), timeout=5)
        soup = BeautifulSoup(resp.content, "html.parser")
        
        price_span = soup.find("span", {"class": "text-2"})
        if not price_span: price_span = soup.select_one('.price-arrow-down, .price-arrow-up')
        
        if price_span:
            return float(price_span.text.strip().replace(',', '.'))
    except: pass
    return None

# --- 3. HABER MOTORU ---
class NewsEngine:
    def analyze_sentiment(self, ticker):
        score = 0
        try:
            query = f"{ticker} hisse kap"
            url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"
            feed = feedparser.parse(url)
            
            pos = ['kar', 'büyüme', 'rekor', 'onay', 'temettü', 'anlaşma', 'geri alım']
            neg = ['zarar', 'düşüş', 'ceza', 'iptal', 'kriz', 'soruşturma']
            
            # Haber okurken de bekleme yapalım
            time.sleep(random.uniform(0.5, 1.5))
            
            for entry in feed.entries[:2]:
                title = entry.title.lower()
                for w in pos: 
                    if w in title: score += 5
                for w in neg: 
                    if w in title: score -= 5
        except: pass
        return max(-15, min(15, score))

# --- 4. HİBRİT ANALİZ (Main.py Kadar Zeki) ---
def analyze_hybrid(ticker, history_data):
    """
    history_data: Yahoo'dan gelen toplu veri (DataFrame)
    """
    news_engine = NewsEngine()
    try:
        # 1. Veriyi Al (Yahoo Geçmişi)
        try:
            df = history_data[f"{ticker}.IS"].copy()
        except: return None # Veri yoksa geç

        if df.empty or df['Close'].isnull().all(): return None
        df = df.dropna()
        if len(df) < 50: return None # Veri azsa geç
        
        # --- ÖN ELEME (Hız İçin) ---
        # Sadece teknik olarak umut vaat edenlere Canlı Fiyat soracağız.
        # Yoksa 600 hisseye canlı sormak 1 saat sürer.
        temp_rsi = ta.rsi(df['Close'], length=14).iloc[-1]
        if temp_rsi > 50 and temp_rsi < 70: return None # Nötr bölgedeyse hiç uğraşma
        
        # 2. Canlı Fiyatı Çek (Sadece Adaylar İçin)
        live_price = get_realtime_price(ticker)
        
        # 3. Veriyi Yamala
        if live_price and live_price > 0:
            last_yahoo = df['Close'].iloc[-1]
            # %20 sapma kontrolü
            if abs(live_price - last_yahoo) / last_yahoo < 0.20:
                df.iloc[-1, df.columns.get_loc('Close')] = live_price
                if live_price > df.iloc[-1]['High']: df.iloc[-1, df.columns.get_loc('High')] = live_price
                if live_price < df.iloc[-1]['Low']: df.iloc[-1, df.columns.get_loc('Low')] = live_price

        # 4. Detaylı Teknik Analiz (Main.py ile aynı)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None: df = pd.concat([df, bb], axis=1)
        
        psar = ta.psar(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, psar], axis=1)
        # PSAR sütununu bul
        psar_col = next((c for c in df.columns if c.startswith('PSAR')), None)
        
        ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
        df = pd.concat([df, ichimoku], axis=1)
        
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        
        last = df.iloc[-1]
        if pd.isna(last['RSI']): return None
        
        # 5. Puanlama
        score = 50
        reasons = []
        
        # Trend
        if last['Close'] > last['VWAP']: score += 10
        if last['MACD_12_26_9'] > last['MACDs_12_26_9']: 
            score += 15
            reasons.append("MACD Al")
            
        # Ichimoku & PSAR
        if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']: score += 10
        if psar_col and df[psar_col].iloc[-1] < last['Close']: score += 10
        
        # Osilatörler
        if last['RSI'] < 30: 
            score += 25
            reasons.append(f"RSI Dip ({last['RSI']:.0f})")
        elif last['RSI'] > 70: score -= 20
        
        # Bollinger Alt Bant
        bbl = next((c for c in df.columns if c.startswith('BBL')), None)
        if bbl and last['Close'] <= last[bbl] * 1.01:
            score += 20
            reasons.append("Bollinger Dip")
            
        # 6. Haber Analizi (Sadece Yüksek Puanlılar İçin)
        if score >= 70:
            n_score = news_engine.analyze_sentiment(ticker)
            score += n_score
            if n_score > 0: reasons.append("Haber+")
            
        score = max(0, min(100, score))
        
        # 7. Sinyal
        if score >= 85:
            return {
                "Hisse": ticker,
                "Fiyat": last['Close'],
                "Skor": int(score),
                "Neden": ", ".join(reasons)
            }
            
    except: return None
    return None

def main():
    print("Bot Görevde...")
    tickers = get_live_tickers()
    if not tickers: return
    
    print(f"{len(tickers)} hisse taranacak.")
    firsatlar = []
    
    # Batch İndirme (Yahoo Geçmişi)
    # 50'şerli paketler halinde indirip işleyeceğiz
    chunk_size = 50
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for chunk in chunks:
        symbols = [f"{t}.IS" for t in chunk]
        try:
            # Yahoo'dan geçmişi toplu çek
            data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False)
            
            # Her hisseyi işle (Canlı Fiyat + Analiz)
            for t in chunk:
                res = analyze_hybrid(t, data)
                if res: firsatlar.append(res)
                
        except: pass
        
        # Paketler arası bekleme (Yahoo engeli yememek için)
        time.sleep(random.uniform(2.0, 5.0))
        
    if firsatlar:
        firsatlar.sort(key=lambda x: x['Skor'], reverse=True)
        top_picks = firsatlar[:10]
        
        msg = "🦅 **MERTT AI: Canlı Sniper Sinyalleri** 🦅\n"
        msg += f"🕒 {time.strftime('%H:%M')}\n\n"
        
        for f in top_picks:
            msg += f"🟢 *{f['Hisse']}* | {f['Fiyat']:.2f} TL\n"
            msg += f"   └ Puan: {f['Skor']} | {f['Neden']}\n\n"
            
        send_telegram(msg)
    else:
        print("Fırsat yok.")

if __name__ == "__main__":
    main()
