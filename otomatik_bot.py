import yfinance as yf
import pandas_ta as ta
import requests
import pandas as pd
import concurrent.futures
import os

# --- GITHUB SECRETS'TAN ALINACAK BİLGİLER ---
# Kodun içine Token yazmıyoruz! GitHub'a ekleyeceğiz.
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") 
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram(message):
    """Telegram'a Güvenli Mesaj Atar"""
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": CHAT_ID, 
                "text": message, 
                "parse_mode": "Markdown"
            }
            requests.post(url, json=payload)
            print("Mesaj gönderildi.")
        except Exception as e:
            print(f"Telegram hatası: {e}")
    else:
        print("HATA: Token veya Chat ID bulunamadı!")

def analyze(ticker):
    try:
        t = f"{ticker}.IS"
        # 15 dakikalık veride analiz (Daha az gürültü)
        df = yf.download(t, period="5d", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        
        if len(df) < 50: return None
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        
        last = df.iloc[-1]
        
        # STRATEJİ: RSI < 40 (Ucuz) VE Fiyat > VWAP (Alıcılar Gelmiş)
        if last['RSI'] < 40 and last['Close'] > last['VWAP']:
            return f"📈 *{ticker}*\nFiyat: {last['Close']:.2f} TL\nRSI: {last['RSI']:.0f}"
        return None
    except: return None

def main():
    print("Otomatik Tarama Başlıyor...")
    # Buraya en önemli 30 hisseyi ekle
    tickers = ["THYAO", "ASELS", "KCHOL", "GARAN", "AKBNK", "SASA", "SISE", "EREGL", "TUPRS", "BIMAS", "HEKTS", "PETKM", "ISCTR", "SAHOL", "FROTO", "YKBNK", "EKGYO", "ODAS", "KOZAL", "KONTR", "ASTOR", "EUPWR", "GUBRF", "OYAKC", "TCELL", "TTKOM", "ENKAI", "VESTL", "ARCLK", "TOASO"]
    
    firsatlar = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(analyze, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: firsatlar.append(res)
            
    if firsatlar:
        header = "🦅 **MERTT AI Sinyalleri** 🦅\n\n"
        body = "\n-------------------\n".join(firsatlar)
        footer = "\n\n_Bu bir yatırım tavsiyesi değildir._"
        send_telegram(header + body + footer)
    else:
        print("Fırsat yok.")

if __name__ == "__main__":
    main()
      
