streamlit
yfinance
pandas
pandas_ta
xgboost
requests
beautifulsoup4
plotly
lxml
openpyxl
scikit-learn
matplotlib
        elif last['RSI'] > 70:
            signal = "SAT"
            color = "red"
            # RSI 70'i geçtikçe satış baskısı skoru artar
            skor = min(90, (last['RSI'] - 50) * 2)
            
        return {
            "Fiyat": last['Close'], "RSI": last['RSI'], 
            "Sinyal": signal, "Renk": color, "Skor": int(skor),
            "Stop": last['Close'] - last['ATR']*1.5,
            "Hedef": last['Close'] + last['ATR']*3,
            "Data": df
        }
    except: return None

# --- TOPLU ANALİZ (Batch) ---
def analyze_batch(tickers_list):
    results = []
    symbols = [f"{t}.IS" for t in tickers_list]
    
    try:
        data = yf.download(symbols, period="3mo", interval="60m", group_by='ticker', progress=False, threads=True)
        
        for ticker in tickers_list:
            try:
                try: df = data[f"{ticker}.IS"].copy()
                except: continue

                if df.empty or df['Close'].isnull().all(): continue
                df = df.dropna()
                if len(df) < 50: continue 
                
                rsi = ta.rsi(df['Close'], length=14)
                vwap = (df['Volume'] * (df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
                
                last_close = df['Close'].iloc[-1]
                last_rsi = rsi.iloc[-1]
                last_vwap = vwap.iloc[-1]
                
                if last_close <= 0 or pd.isna(last_rsi): continue
                
                signal = "NÖTR"
                skor = 50 # Varsayılan Nötr Skoru
                
                # --- DİNAMİK SKOR HESAPLAMA ---
                if last_rsi < 45 and last_close > last_vwap:
                    signal = "GÜÇLÜ AL"
                    # RSI ne kadar düşükse skor o kadar yüksek (Max 99)
                    skor = 50 + ((50 - last_rsi) * 2)
                    if skor > 99: skor = 99
                    
                elif last_rsi > 75:
                    signal = "SAT"
                    # RSI ne kadar yüksekse düşüş ihtimali o kadar yüksek
                    skor = (last_rsi - 50) * 2
                    if skor > 95: skor = 95
                    
                elif last_close < last_vwap and last_rsi < 50:
                    signal = "DÜŞÜŞ TRENDİ"
                    skor = 30
                
                # Sadece önemli sinyalleri al
                if "AL" in signal or "SAT" in signal or "DÜŞÜŞ" in signal:
                    results.append({
                        "Hisse": ticker,
                        "Fiyat": last_close,
                        "Sinyal": signal,
                        "RSI": last_rsi,
                        "Skor": int(skor) # Tam sayıya çevir
                    })
            except: continue
    except: pass
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

    tum_hisseler = get_live_tickers()

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
                    k1.metric("Fiyat", f"{res['Fiyat']:.2f} TL")
                    k2.metric("Sinyal", res['Sinyal'], delta=f"Güven: %{res['Skor']}")
                    k3.metric("RSI", f"{res['RSI']:.0f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=res['Data'].index, open=res['Data']['Open'], high=res['Data']['High'], low=res['Data']['Low'], close=res['Data']['Close']))
                    fig.update_layout(template="plotly_dark", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Veri bulunamadı.")

    elif menu == "📡 Piyasa Radarı (Batch)":
        st.title("📡 MERTT Piyasa Radarı")
        
        if not tum_hisseler:
            st.error("⚠️ HATA: Canlı borsa listesine ulaşılamıyor!")
            st.stop()
            
        st.info(f"Canlı Takipteki Hisse Sayısı: {len(tum_hisseler)}")
        
        if st.button("TÜM BORSAYI TARA (Turbo Mod) 🚀", type="primary"):
            all_results = []
            chunk_size = 50 
            chunks = [tum_hisseler[i:i + chunk_size] for i in range(0, len(tum_hisseler), chunk_size)]
            
            bar = st.progress(0)
            status = st.empty()
            
            for i, chunk in enumerate(chunks):
                status.text(f"Analiz Paketi {i+1}/{len(chunks)} işleniyor...")
                batch_res = analyze_batch(chunk)
                all_results.extend(batch_res)
                bar.progress((i + 1) / len(chunks))
                time.sleep(1) 
            
            bar.empty()
            status.empty()
            
            if all_results:
                df = pd.DataFrame(all_results)
                
                st.success(f"Tarama Bitti! {len(df)} Sinyal Bulundu.")
                
                # --- PROFESYONEL TABLO TASARIMI ---
                st.dataframe(
                    df,
                    column_config={
                        "Hisse": st.column_config.TextColumn("Hisse Kodu"),
                        "Fiyat": st.column_config.NumberColumn("Fiyat (TL)", format="%.2f TL"),
                        "Sinyal": st.column_config.TextColumn("AI Kararı"),
                        "RSI": st.column_config.NumberColumn("RSI Gücü", format="%.0f"),
                        "Skor": st.column_config.ProgressColumn(
                            "AI Güven Skoru",
                            help="Yapay Zeka'nın sinyale olan güven derecesi",
                            format="%d",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("Hiçbir sinyal bulunamadı.")

if __name__ == "__main__":
    main()
    
