import streamlit as st
import pandas as pd
import openai
import os
import json
import boto3
from botocore.client import Config
import io
from dotenv import load_dotenv
import pandera as pa
from pandera.errors import SchemaError
# Ostateczna, poprawna wersja importu
from langfuse import Langfuse
from pycaret.regression import load_model, predict_model

# Wczytaj zmienne środowiskowe z pliku .env
load_dotenv()

# --- Konfiguracja ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
DO_SPACES_KEY = os.getenv('DO_SPACES_KEY')
DO_SPACES_SECRET = os.getenv('DO_SPACES_SECRET')
DO_SPACES_ENDPOINT_URL = os.getenv('DO_SPACES_ENDPOINT_URL', 'https://fra1.digitaloceanspaces.com')
DO_SPACES_BUCKET = os.getenv('DO_SPACES_BUCKET')
MODEL_FILE_KEY = 'models/halfmarathon_pipeline.pkl'

# Inicjalizacja klientów
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Inicjalizacja Langfuse.
# Jeśli klucze nie zostaną podane, obiekt nie zostanie utworzony.
langfuse = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host="https://cloud.langfuse.com"
    )

# --- Schemat walidacji Pandera ---
llm_output_schema = pa.DataFrameSchema({
    "wiek": pa.Column(int, checks=pa.Check.in_range(1, 100), nullable=False),
    "plec": pa.Column(str, checks=pa.Check.isin(['K', 'M']), nullable=False),
    "tempo_5km": pa.Column(str, checks=pa.Check.str_matches(r'^\d{1,2}:\d{2}$'), nullable=False),
})

# --- Funkcje pomocnicze ---
@st.cache_resource
def get_boto_client():
    session = boto3.session.Session()
    return session.client('s3', config=Config(s3={'addressing_style': 'path'}), region_name=DO_SPACES_ENDPOINT_URL.split('//')[1].split('.')[0], endpoint_url=DO_SPACES_ENDPOINT_URL, aws_access_key_id=DO_SPACES_KEY, aws_secret_access_key=DO_SPACES_SECRET)

@st.cache_resource
def load_model_from_spaces():
    """Pobiera i wczytuje pipeline PyCaret z DigitalOcean Spaces."""
    try:
        client = get_boto_client()
        
        # Wracamy do niezawodnej metody zapisu do pliku tymczasowego,
        # ponieważ ładowanie z `io.BytesIO` nie jest wspierane przez `load_model` w ten sposób.
        local_path_with_ext = 'downloaded_model.pkl'
        local_path_no_ext = 'downloaded_model'
        
        client.download_file(DO_SPACES_BUCKET, MODEL_FILE_KEY, local_path_with_ext)
        
        # Przekazujemy do funkcji load_model nazwę pliku BEZ rozszerzenia .pkl
        model = load_model(local_path_no_ext)
        
        # Usuwamy plik tymczasowy po załadowaniu modelu do pamięci
        os.remove(local_path_with_ext)
        return model
    except Exception as e:
        st.error(f"Nie udało się załadować modelu: {e}")
        return None

def time_str_to_seconds(time_str):
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except: return None

def format_time_from_seconds(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def extract_data_with_llm(user_input):
    """Używa LLM do ekstrakcji danych z tekstu użytkownika."""
    trace = None
    if langfuse:
        trace = langfuse.trace(
            name="data-extraction",
            input={"user_description": user_input}
        )

    if not OPENAI_API_KEY:
        st.error("Klucz API OpenAI nie jest skonfigurowany.")
        return None
    system_prompt = """
    Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyekstrahowanie trzech informacji z tekstu podanego przez użytkownika: wieku, płci oraz tempa biegu na 5km.
    Zwróć odpowiedź wyłącznie w formacie JSON.
    - Wiek (`wiek`) powinien być liczbą całkowitą.
    - Płeć (`plec`) powinna być jedną z dwóch wartości: 'M' (mężczyzna) lub 'K' (kobieta).
    - Tempo na 5km (`tempo_5km`) powinno być w formacie "MM:SS".
    Jeśli którejś informacji brakuje, ustaw dla niej wartość null.
    """
    try:
        response = openai.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}], response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        if trace:
            trace.update(output=result)
        return result
    except Exception as e:
        if trace:
            trace.update(output={"error": str(e)})
        st.error(f"Błąd podczas komunikacji z OpenAI: {e}")
        return None

# --- Główna aplikacja Streamlit ---
st.set_page_config(page_title="Szacowanie Czasu Półmaratonu", layout="wide")

st.title("🏃‍♂️ Estymator Czasu Ukończenia Półmaratonu")
st.markdown("Opisz siebie, a my oszacujemy Twój czas! Podaj swój **wiek**, **płeć** oraz **średnie tempo na 5 km**.")

# --- ZMIENIONY OBRAZEK ---
# Używamy ścieżki do "surowego" pliku na GitHubie.
st.image("https://mod-9-homework.fra1.digitaloceanspaces.com/Image_19f2o619f2o619f2.png", caption="Walczymy o nowe rekordy!", use_column_width=True)


pipeline = load_model_from_spaces()

user_description = st.text_area("Przedstaw się:", "Cześć, mam 33 lata, jestem mężczyzną. Biegam 5km w 24 minuty i 15 sekund.", height=100)

if st.button("Szacuj czas", type="primary"):
    if not user_description:
        st.warning("Proszę, opisz siebie w polu tekstowym.")
    elif pipeline is None:
        st.error("Model predykcyjny nie jest dostępny. Skontaktuj się z administratorem.")
    else:
        with st.spinner("Analizuję Twoje dane i liczę..."):
            extracted_data = extract_data_with_llm(user_description)
            if not extracted_data:
                st.error("Nie udało się przetworzyć Twojego opisu.")
            else:
                st.subheader("🤖 Dane wyekstrahowane przez AI:")
                st.json(extracted_data)
                try:
                    validation_df = pd.DataFrame([extracted_data])
                    llm_output_schema.validate(validation_df)
                    st.info("Dane wejściowe poprawne. Przystępuję do predykcji.")
                    
                    wiek = extracted_data["wiek"]
                    plec = extracted_data["plec"]
                    tempo_5km_str = extracted_data["tempo_5km"]
                    czas_5km_s = time_str_to_seconds(tempo_5km_str)
                    tempo_1km_s = czas_5km_s / 5

                    # --- OPTYMALIZACJA PAMIĘCI ---
                    # Definiujemy bardziej oszczędne typy danych dla DataFrame'u.
                    input_data = {
                        'wiek': [wiek], 
                        'plec': [plec], 
                        'tempo_5km_s_na_km': [tempo_1km_s]
                    }
                    input_df = pd.DataFrame(input_data).astype({
                        'wiek': 'int16',
                        'plec': 'category',
                        'tempo_5km_s_na_km': 'float32'
                    })
                    
                    predictions = predict_model(pipeline, data=input_df)
                    prediction_s = predictions['prediction_label'].iloc[0]
                    predicted_time_str = format_time_from_seconds(prediction_s)

                    st.success("Oszacowanie gotowe!")
                    st.metric(label="Przewidywany czas netto na mecie półmaratonu", value=predicted_time_str)

                except SchemaError as err:
                    st.error(f"Błąd walidacji danych: {err.failure_cases['failure_case'][0]}")
                except Exception as e:
                    st.error(f"Wystąpił nieoczekiwany błąd: {e}")

st.markdown("---")
st.info("Aplikacja wykorzystuje najlepszy model wybrany automatycznie przez PyCaret (AutoML) oraz model LLM (OpenAI GPT-3.5) do analizy tekstu.")
