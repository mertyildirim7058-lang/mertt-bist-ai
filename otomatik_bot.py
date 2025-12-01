import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import feedparser
from bs4 import BeautifulSoup
import os
import concurrent.futures
import time

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram(message):
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"})
        except: pass

def get_live_tickers():
    # ... (Aynı Canlı Liste Fonksiyonu) ...
    # Kısaltmak için buraya önceki kodlardaki listeyi koyabilirsin veya boş dönerse BIST 30 kullanır
    return ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL"] # Örnek

class NewsEngine:
    def analyze_sentiment(self, ticker):
        score = 0
        try:
            url = f"https://news.google.com/rss/search?q={ticker}+hisse+kap&hl=tr&gl=TR&ceid=TR:tr"
            feed = feedparser.parse(url)
            pos = ['kar', 'büyüme', 'rekor', 'onay', 'temettü', 'anlaşma']
            neg = ['zarar', 'düşüş', 'ceza', 'iptal', 'kriz']
            news_found = []
            
            for entry in feed.entries[:2]:
                title = entry.title.lower()
                impact = 0
                for w in pos: 
                    if w in title: impact += 1
                for w in neg: 
                    if w in title: impact -= 2
                
                if impact != 0:
                    score += (impact * 5)
                    news_found.append(entry.title)
                    
            return max(-20, min(20, score)), news_found
        except: return 0, []

def analyze_batch(tickers):
    results = []
    symbols = [f"{t}.IS" for t in tickers]
    news_engine = NewsEngine()
    
    try:
        data = yf.download(symbols, period="2mo", interval="60m", group_by='ticker', progress=False)
        for ticker in tickers:
            try:
                df = data[f"{ticker}.IS"].copy().dropna()
                if len(df) < 50: continue
                
                rsi = ta.rsi(df['Close'], length=14).iloc[-1]
                vwap = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
                last_price = df['Close'].iloc[-1]
                
                score = 50
                tech_note = ""
                
                if rsi < 40 and last_price > vwap.iloc[-1]: 
                    score = 75
                    tech_note = "Teknik Güçlü"
                
                # Sadece Teknik Puanı İyi Olanlara Haber Bak (Hız İçin)
                news_note = []
                if score >= 70:
                    n_score, n_list = news_engine.analyze_sentiment(ticker)
                    score += n_score
                    news_note = n_list
                
                if score >= 85:
                    results.append({
                        "Hisse": ticker, "Fiyat": last_price, "Skor": score,
                        "Not": tech_note, "Haber": news_note[0] if news_note else "Haber yok"
                    })
            except: continue
    except: pass
    return results

def main():
    # ... (Ana döngü, Telegram gönderimi) ...
    # Burası önceki otomatik_bot ile aynı, sadece analyze_batch fonksiyonunu yukarıdaki gibi güncelle.
    pass

if __name__ == "__main__":
    main()
