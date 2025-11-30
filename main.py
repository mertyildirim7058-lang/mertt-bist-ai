import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import numpy as np
from PIL import Image
import time

# --- 1. AYARLAR ---
LOGO_INTERNET_LINKI = "https://raw.githubusercontent.com/kullaniciadi/proje/main/logo.png"

st.set_page_config(
    page_title="MERTT AI", 
    layout="wide", 
    page_icon="🛡️"  
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
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            logo_goster()
            st.markdown("<h4 style='text-align: center;'>Gelecek İçin Bilgi ve Teknoloji</h4>", unsafe_allow_html=True)
            st.divider()
            sifre = st.text_input("Erişim Anahtarı:", type="password")
            if st.button("Sisteme Giriş Yap", type="primary", use_container_width=True):
                try:
                    if sifre == st.secrets["GIRIS_SIFRESI"]: 
                        st.session_state['giris_yapildi'] = True
                        st.rerun()
                    else: st.error("⛔ Yetkisiz Erişim!")
                except: st.error("Sistem Hatası: Şifre tanımlı değil.")
        return False
    return True

if not guvenlik_kontrolu(): st.stop()

# --- TAM HİSSE LİSTESİ (MANUEL VE KESİN) ---
def get_all_tickers():
    """BIST Tüm Hisse Senetleri (Kasım 2025)"""
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

# --- TEK HİSSE ANALİZİ (Manuel Sorgu İçin) ---
def analyze_single(ticker):
    try:
        t = f"{ticker}.IS"
        df = yf.download(t, period="2mo", interval="60m", progress=False)
        if df is None or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = [col[0] for col in df.columns]
        
        # İndikatörler
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['VWAP'] = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        last = df.iloc[-1]
        
        # Basit Kontrol
        if pd.isna(last['RSI']): return None
        
        signal = "NÖTR"
        color = "gray"
        if last['RSI'] < 45 and last['Close'] > last['VWAP']: 
            signal = "GÜÇLÜ AL"
            color = "green"
        elif last['RSI'] > 70:
            signal = "SAT"
            color = "red"
            
        return {
            "Fiyat": last['Close'], "RSI": last['RSI'], 
            "Sinyal": signal, "Renk": color, 
            "Stop": last['Close'] - last['ATR']*1.5,
            "Hedef": last['Close'] + last['ATR']*3,
            "Data": df
        }
    except: return None

# --- TOPLU ANALİZ MOTORU (Batch Processing) ---
def analyze_batch(tickers_list):
    """50'li paketler halinde indirir ve işler (ÇOK HIZLI VE GÜVENLİ)"""
    results = []
    
    # 1. Liste Hazırlığı (.IS ekle)
    symbols = [f"{t}.IS" for t in tickers_list]
    
    try:
        # 2. TOPLU İNDİRME (Tek İstek!)
        # group_by='ticker' çok önemlidir, veriyi hisse hisse ayırır.
        data = yf.download(symbols, period="2mo", interval="60m", group_by='ticker', progress=False, threads=True)
        
        # 3. Her hisse için döngü
        for ticker in tickers_list:
            try:
                # Veriyi çek (MultiIndex'ten)
                df = data[f"{ticker}.IS"].copy()
                
                # Veri Kontrolü (Boş mu?)
                if df.empty or df['Close'].isnull().all(): continue
                
                # NaN Temizliği
                df = df.dropna()
                if len(df) < 50: continue # Yeni arzsa atla
                
                # --- İNDİKATÖRLER ---
                # Pandas TA bazen toplu indirmede hata verebilir, manuel hesaplama daha güvenli olabilir ama deneyelim:
                rsi = ta.rsi(df['Close'], length=14)
                vwap = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
                
                last_close = df['Close'].iloc[-1]
                last_rsi = rsi.iloc[-1]
                last_vwap = vwap.iloc[-1]
                
                # --- FİLTRELEME ---
                # Fiyat 0 olamaz, RSI NaN olamaz
                if last_close <= 0 or pd.isna(last_rsi): continue
                
                # --- KARAR MEKANİZMASI ---
                signal = "NÖTR"
                skor = 50
                
                # Basit ve Etkili Strateji
                if last_rsi < 45 and last_close > last_vwap:
                    signal = "GÜÇLÜ AL"
                    skor = 85
                elif last_rsi > 75:
                    signal = "AŞIRI ALIM (SAT)"
                    skor = 20
                elif last_close < last_vwap and last_rsi < 50:
                    signal = "DÜŞÜŞ TRENDİ"
                    skor = 30
                
                # SADECE SİNYAL OLANLARI KAYDET
                if "AL" in signal or "SAT" in signal or "DÜŞÜŞ" in signal:
                    results.append({
                        "Hisse": ticker,
                        "Fiyat": last_close,
                        "Sinyal": signal,
                        "RSI": last_rsi,
                        "Skor": skor
                    })
            except: continue
            
    except Exception as e:
        st.error(f"Toplu indirme hatası: {e}")
        
    return results

# --- ARAYÜZ ---
def main():
    with st.sidebar:
        logo_goster()
        st.markdown("<h3 style='text-align: center;'>Yapay Zeka Üssü</h3>", unsafe_allow_html=True)
        st.divider()
        menu = st.radio("Panel", ["💬 Hisse Sor", "📡 Piyasa Radarı (Batch)", "Çıkış"])
        if menu == "Çıkış":
            st.session_state['giris_yapildi'] = False
            st.rerun()

    if menu == "💬 Hisse Sor":
        st.title("🤖 Hisse Analiz Asistanı")
        c1, c2 = st.columns([3,1])
        with c1: sembol = st.text_input("Hisse Kodu (Örn: THYAO):", "").upper()
        with c2: 
            st.markdown("<br>", unsafe_allow_html=True)
            btn = st.button("Analiz Et", type="primary")

        if btn and sembol:
            with st.spinner("Analiz ediliyor..."):
                res = analyze_single(sembol)
                if res:
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f}")
                    k2.metric("Sinyal", res['Sinyal'])
                    k3.metric("RSI", f"{res['RSI']:.0f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=res['Data'].index, open=res['Data']['Open'], high=res['Data']['High'], low=res['Data']['Low'], close=res['Data']['Close']))
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Veri bulunamadı.")

    elif menu == "📡 Piyasa Radarı (Batch)":
        st.title("📡 MERTT Piyasa Radarı")
        tum_hisseler = get_all_tickers()
        st.info(f"Takipteki Hisse Sayısı: {len(tum_hisseler)}")
        
        if st.button("TÜM BORSAYI TARA (Turbo Mod) 🚀", type="primary"):
            all_results = []
            
            # LİSTEYİ 50'şerli PARÇALARA BÖL (Chunking)
            chunk_size = 50
            chunks = [tum_hisseler[i:i + chunk_size] for i in range(0, len(tum_hisseler), chunk_size)]
            
            bar = st.progress(0)
            status = st.empty()
            
            # Her parçayı işle
            for i, chunk in enumerate(chunks):
                status.text(f"Paket {i+1}/{len(chunks)} işleniyor... ({chunk[0]} - {chunk[-1]})")
                batch_res = analyze_batch(chunk)
                all_results.extend(batch_res)
                bar.progress((i + 1) / len(chunks))
                # Yahoo'yu kızdırmamak için minik bekleme
                time.sleep(1)
            
            bar.empty()
            status.empty()
            
            if all_results:
                df = pd.DataFrame(all_results)
                try:
                    st.success(f"Tarama Bitti! {len(df)} Fırsat Bulundu.")
                    st.dataframe(
                        df.style.format({"Fiyat": "{:.2f}", "RSI": "{:.0f}", "Skor": "{:.0f}"})
                        .background_gradient(subset=['Skor'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                except: st.dataframe(df)
            else:
                st.warning("Hiçbir sinyal bulunamadı.")

if __name__ == "__main__":
    main()
    
