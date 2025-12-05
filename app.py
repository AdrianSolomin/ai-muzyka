import streamlit as st
import librosa
import numpy as np
import joblib

# 1. Wczytanie modelu (mózgu AI)
try:
    model = joblib.load("moj_model_muzyczny.pkl")
except:
    st.error("Błąd: Nie znaleziono pliku 'moj_model_muzyczny.pkl'. Uruchom najpierw trenowanie.")
    st.stop()

# Funkcja do obróbki dźwięku (identyczna jak przy trenowaniu)
def przetworz_audio(plik_audio):
    # Wczytujemy 30 sekund nagrania
    y, sr = librosa.load(plik_audio, duration=30)
    # Wyciągamy cechy MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Uśredniamy wynik
    return np.mean(mfcc.T, axis=0)

# 2. Wygląd strony
st.set_page_config(page_title="Music Genre AI", page_icon="🎵")
st.title("🎵 Rozpoznawanie Gatunków Muzycznych")
st.write("Wgraj plik .wav, a sztuczna inteligencja odgadnie jego gatunek.")

# 3. Wgrywanie pliku
plik = st.file_uploader("Wrzuć plik tutaj:", type=["wav", "mp3"])

if plik is not None:
    # Wyświetlamy odtwarzacz audio
    st.audio(plik)
    
    if st.button("Analizuj utwór"):
        with st.spinner("Słucham i analizuję..."):
            try:
                # --- ANALIZA ---
                # Zamiana dźwięku na liczby
                cechy = przetworz_audio(plik)
                cechy = cechy.reshape(1, -1) # Formatowanie pod model (1 wiersz)
                
                # --- PREDYKCJA ---
                # Jaki to gatunek? (np. 'rock')
                wynik = model.predict(cechy)[0]
                # Z jaką pewnością? (np. [0.1, 0.8, 0.1...])
                prawdopodobienstwa = model.predict_proba(cechy)[0]
                
                # Obliczamy maksymalną pewność w procentach
                pewnosc_procent = np.max(prawdopodobienstwa) * 100
                
                # --- WYNIKI ---
                
                # 1. Główny komunikat na zielonym pasku
                st.success(f"To brzmi jak: **{wynik.upper()}** ")
                
                # 2. Duży licznik (wygląda profesjonalnie)
                st.metric(
                    label="Zidentyfikowany gatunek", 
                    value=wynik.upper(), 
                    delta=f"{pewnosc_procent:.2f}% pewności"
                )
                
                # 3. Wykres słupkowy dla wszystkich gatunków
                st.write("---")
                st.write("Szczegółowy rozkład prawdopodobieństwa:")
                # Tworzymy słownik {gatunek: procent} dla wykresu
                dane_wykresu = dict(zip(model.classes_, prawdopodobienstwa))
                st.bar_chart(dane_wykresu)
                
            except Exception as e:
                st.error(f"Wystąpił błąd podczas analizy: {e}")