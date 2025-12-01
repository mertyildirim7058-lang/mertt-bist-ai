import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import feedparser
from bs4 import BeautifulSoup
import os
import concurrent.futures
import time
import random

# --- GITHUB SECRETS (AYARLAR) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# --- 1. HABER İSTİHBARAT MOTORU ---
class NewsEngine:
    def __init__(self):
        self.pozitif = ['kar', 'artış', 'büyüme', 'rekor', 'onay', 'temettü', 'anlaşma', 'ihale', 'geri alım', 'yatırım']
        self.negatif = ['zarar', 'düşüş', 'ceza', 'iptal', 'kriz', 'soruşturma', 'borç', 'satış', 'azalış']
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7'
        }

    def analyze_sentiment(self, ticker):
        """Sadece potansiyel hisseler için haber tarar (Hız için)"""
        score = 0
        try:
            query = f"{ticker} hisse kap borsa"
            url = f"https://news.google.com/rss/search?q={query}&hl=tr&gl=TR&ceid=TR:tr"
            
            # Requests ile çekip Feedparser'a veriyoruz (Engel yememek için)
            response = requests.get(url, headers=self.headers, timeout=5)
            feed = feedparser.parse(response.content)
            
            haber_ozeti = []
            
            for entry in feed.entries[:3]: # Son 3 habere bak
                title = entry.title.lower()
                impact = 0
                
                for w in self.pozitif: 
                    if w in title: impact += 1
                for w in self.negatif: 
                    if w in title: impact -= 2 # Kötü haber daha etkilidir
                
                if impact != 0:
                    haber_ozeti.append(f"• {entry.title[:30]}...")
                    score += (impact * 5) # Her etki 5 puan
                    
            return max(-20, min(20, score)), haber_ozeti
        except: return 0, []

# --- 2. CANLI LİSTE MOTORU (YEDEKSİZ) ---
def get_live_tickers():
    canli_liste = []
    url = "https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/default.aspx"
    
    # 1. Pandas ile dene
    try:
        df = pd.read_html(url)[0]
        raw = df.iloc[:, 0].tolist()
        canli_liste = [str(x).strip() for x in raw if str(x).isalnum()]
    except: pass
    
    # 2. Requests ile dene
    if not canli_liste:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.content, 'html.parser')
            table = soup.find('table', {'id': 'tableHisseOnerileri'})
            if table:
                rows = table.find('tbody').find_all('tr')
                for r in rows:
                    cols = r.find_all('td')
                    if cols: canli_liste.append(cols[0].find('a').text.strip())
        except: pass
        
    return sorted(list(set(canli_liste)))

# --- 3. TEKNİK ANALİZ MOTORU ---
def analyze_technical(df):
    """Tüm indikatörleri hesaplar ve puan verir"""
    try:
        # İndikatörler
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None: df = pd.concat([df, bb], axis=1)
        
        psar = ta.psar(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, psar], axis=1)
        # PSAR sütun adını bul
        psar_col = next((c for c in df.columns if c.startswith('PSAR')), None)
        
        ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
        df = pd.concat([df, ichimoku], axis=1)
        
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        if pd.isna(last['RSI']): return 0, []
        
        # --- PUANLAMA ---
        score = 50
        reasons = []
        
        # 1. Trend
        if last['Close'] > last['VWAP']: score += 10
        if last['MACD_12_26_9'] > last['MACDs_12_26_9']: 
            score += 15
            reasons.append("MACD Al")
        
        # 2. Ichimoku & PSAR
        if last['Close'] > last['ISA_9'] and last['Close'] > last['ISB_26']:
            score += 10
        if psar_col and df[psar_col].iloc[-1] < last['Close']:
            score += 10
            
        # 3. Osilatör & Volatilite
        # Bollinger (Dinamik İsim Kontrolü)
        bbl = next((c for c in df.columns if c.startswith('BBL')), None)
        if bbl and last['Close'] <= last[bbl] * 1.01:
            score += 20
            reasons.append("Bollinger Dip")
            
        if last['RSI'] < 30: 
            score += 25
            reasons.append(f"RSI Ucuz ({last['RSI']:.0f})")
        elif last['RSI'] > 70: 
            score -= 20
        
        return score, reasons
    except: return 0, []

# --- 4. TOPLU TARAMA VE FİLTRELEME ---
def process_batch(tickers):
    results = []
    symbols = [f"{t}.IS" for t in tickers]
    news_engine = NewsEngine()
    
    try:
        # Toplu Veri İndirme (Hız için 3 aylık yeterli)
        data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
        
        for ticker in tickers:
            try:
                try: df = data[f"{ticker}.IS"].copy()
                except: continue
                
                if df.empty or df['Close'].isnull().all(): continue
                df = df.dropna()
                if len(df) < 50: continue # Yeni arzları ele
                
                last_price = df['Close'].iloc[-1]
                if last_price <= 0: continue

                # 1. Teknik Puanlama
                tech_score, tech_reasons = analyze_technical(df)
                
                # Sadece Teknik Puanı 65 üzeri olanlara Haber Analizi yap (API kotası ve hız için)
                news_score = 0
                news_notes = []
                
                if tech_score >= 65:
                    news_score, news_notes = news_engine.analyze_sentiment(ticker)
                
                final_score = tech_score + news_score
                final_score = max(0, min(100, final_score))
                
                # Karar
                signal = None
                if final_score >= 80: signal = "GÜÇLÜ AL 🚀"
                elif final_score <= 25: signal = "SAT 🔻" # Satış uyarısı da iyidir
                
                if signal:
                    results.append({
                        "Hisse": ticker,
                        "Fiyat": last_price,
                        "Sinyal": signal,
                        "Skor": final_score,
                        "Nedenler": tech_reasons + ([f"Haber: {n}" for n in news_notes] if news_notes else [])
                    })
                    
            except: continue
    except: pass
    return results

def send_report(opportunities):
    if not opportunities: return
    
    # Skora göre sırala
    opportunities.sort(key=lambda x: x['Skor'], reverse=True)
    
    msg = "🦅 **MERTT AI: Otomatik İstihbarat** 🦅\n"
    msg += f"🕒 {time.strftime('%d.%m %H:%M')}\n\n"
    
    for op in opportunities[:10]: # En iyi 10 tanesi
        icon = "🟢" if "AL" in op['Sinyal'] else "🔴"
        detay = ", ".join(op['Nedenler'][:2]) # İlk 2 sebebi yaz
        
        msg += f"{icon} *{op['Hisse']}* | {op['Fiyat']:.2f} TL\n"
        msg += f"   └ Skor: {op['Skor']} | {op['Sinyal']}\n"
        msg += f"   └ _{detay}_\n\n"
        
    msg += "_Otomatik analizdir, yatırım tavsiyesi değildir._"
    
    # Mesaj çok uzunsa böl (Telegram limiti)
    if len(msg) > 4000:
        send_telegram(msg[:4000])
        send_telegram(msg[4000:])
    else:
        send_telegram(msg)

# --- MAIN ---
def main():
    print("Bot uyanıyor...")
    tickers = get_live_tickers()
    if not tickers:
        print("Liste çekilemedi.")
        return
        
    print(f"{len(tickers)} hisse taranacak.")
    all_opportunities = []
    
    # 50'lik paketler
    chunk_size = 50
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        print(f"Paket {i+1} işleniyor...")
        res = process_batch(chunk)
        all_opportunities.extend(res)
        time.sleep(2) # Yahoo'ya nezaket
        
    if all_opportunities:
        print(f"{len(all_opportunities)} fırsat bulundu, gönderiliyor.")
        send_report(all_opportunities)
    else:
        print("Fırsat yok.")

if __name__ == "__main__":
    main()
                          
