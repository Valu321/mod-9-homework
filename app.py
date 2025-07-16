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
from langfuse import Langfuse
from pycaret.regression import load_model, predict_model

# --- Konfiguracja strony ---
st.set_page_config(
    page_title="Estymator Czasu Półmaratonu",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Wczytywanie zmiennych i inicjalizacja ---
load_dotenv()

# Konfiguracja kluczy API
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

langfuse = None
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host="https://cloud.langfuse.com"
    )

# --- Style CSS dla lepszego wyglądu ---
st.markdown("""
<style>
    .stButton > button {
        border-radius: 20px;
        border: 2px solid #1E88E5;
        color: #1E88E5;
        background-color: transparent;
        font-weight: bold;
    }
    .stButton > button:hover {
        border-color: #0D47A1;
        color: #0D47A1;
    }
    .stMetric {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# --- Schemat walidacji Pandera ---
llm_output_schema = pa.DataFrameSchema({
    "wiek": pa.Column(int, checks=pa.Check.in_range(1, 100), nullable=False, error="Wiek musi być liczbą od 1 do 100."),
    "plec": pa.Column(str, checks=pa.Check.isin(['K', 'M']), nullable=False, error="Płeć musi być określona jako 'K' lub 'M'."),
    "tempo_5km": pa.Column(str, checks=pa.Check.str_matches(r'^\d{1,2}:\d{2}$'), nullable=False, error="Tempo na 5km musi być w formacie MM:SS."),
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
        local_path_with_ext = 'downloaded_model.pkl'
        local_path_no_ext = 'downloaded_model'
        client.download_file(DO_SPACES_BUCKET, MODEL_FILE_KEY, local_path_with_ext)
        model = load_model(local_path_no_ext)
        os.remove(local_path_with_ext)
        return model
    except Exception as e:
        st.error(f"Nie udało się załadować modelu: {e}", icon="🚨")
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
        trace = langfuse.trace(name="data-extraction", input={"user_description": user_input})

    if not OPENAI_API_KEY:
        st.error("Klucz API OpenAI nie jest skonfigurowany.", icon="🔑")
        return None
        
    system_prompt = """
    Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyekstrahowanie trzech informacji z tekstu podanego przez użytkownika: wieku, płci oraz tempa biegu na 5km.
    Zwróć odpowiedź wyłącznie w formacie JSON.
    - Wiek (`wiek`) powinien być liczbą całkowitą.
    - Płeć (`plec`) powinna być jedną z dwóch wartości: 'M' (mężczyzna) lub 'K' (kobieta).
    - Tempo na 5km (`tempo_5km`) powinno być w formacie "MM:SS".
    Jeśli którejś informacji brakuje, ustaw dla niej wartość null. Upewnij się, że odpowiedź to poprawny obiekt JSON.
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
        st.error(f"Błąd podczas komunikacji z OpenAI: {e}", icon="🔥")
        return None

# --- Inicjalizacja stanu sesji ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = "Cześć, mam 33 lata, jestem mężczyzną. Biegam 5km w 24 minuty i 15 sekund."


# --- Główna aplikacja Streamlit ---

# --- Nagłówek ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://mod-9-homework.fra1.digitaloceanspaces.com/Image_19f2o619f2o619f2.png", width=150)
with col2:
    st.title("Estymator Czasu Półmaratonu")
    st.markdown("Opisz siebie w polu poniżej, a ja oszacuję Twój przewidywany czas na mecie! Podaj swój **wiek**, **płeć** oraz **tempo na 5 km**.")


# --- Ładowanie modelu (w tle) ---
pipeline = load_model_from_spaces()


# --- Formularz wejściowy ---
st.write("### 💬 Krok 1: Opowiedz nam o sobie")
user_description = st.text_area(
    "Przedstaw się:",
    value=st.session_state.user_input,
    height=100,
    label_visibility="collapsed"
)

col1, col2, _ = st.columns([1, 1, 3])
predict_button = col1.button("Szacuj czas", type="primary", use_container_width=True)
clear_button = col2.button("Wyczyść", use_container_width=True)

if clear_button:
    st.session_state.prediction_result = None
    st.session_state.user_input = ""
    st.rerun()

if predict_button:
    st.session_state.user_input = user_description
    if not user_description:
        st.warning("Proszę, opisz siebie w polu tekstowym.", icon="⚠️")
    elif pipeline is None:
        st.error("Model predykcyjny nie jest dostępny. Skontaktuj się z administratorem.", icon="🚨")
    else:
        with st.spinner("Analizuję Twoje dane i liczę... 🤖"):
            extracted_data = extract_data_with_llm(user_description)
            
            if not extracted_data or all(value is None for value in extracted_data.values()):
                st.error("Nie udało mi się znaleźć potrzebnych informacji w Twoim opisie. Upewnij się, że podałeś/aś swój **wiek**, **płeć** oraz **tempo na 5km**.", icon="😟")
                st.session_state.prediction_result = None
            else:
                try:
                    validation_df = pd.DataFrame([extracted_data])
                    llm_output_schema.validate(validation_df)
                    
                    wiek = extracted_data["wiek"]
                    plec = extracted_data["plec"]
                    tempo_5km_str = extracted_data["tempo_5km"]
                    czas_5km_s = time_str_to_seconds(tempo_5km_str)
                    tempo_1km_s = czas_5km_s / 5

                    input_data = {'wiek': [wiek], 'plec': [plec], 'tempo_5km_s_na_km': [tempo_1km_s]}
                    input_df = pd.DataFrame(input_data).astype({'wiek': 'int16', 'plec': 'category', 'tempo_5km_s_na_km': 'float32'})
                    
                    predictions = predict_model(pipeline, data=input_df)
                    prediction_s = predictions['prediction_label'].iloc[0]
                    
                    st.session_state.prediction_result = {
                        "extracted_data": extracted_data,
                        "predicted_time_str": format_time_from_seconds(prediction_s)
                    }

                except SchemaError as err:
                    st.error(f"Znalazłem błąd w podanych danych: **{err.failure_cases['failure_case'][0]}** Popraw swój opis i spróbuj ponownie.", icon="🔎")
                    st.session_state.prediction_result = None
                except Exception as e:
                    st.error(f"Wystąpił nieoczekiwany błąd: {e}", icon="💥")
                    st.session_state.prediction_result = None

# --- Wyświetlanie wyników ---
if st.session_state.prediction_result:
    st.write("---")
    st.write("### 📈 Krok 2: Analiza i wynik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Dane zinterpretowane przez AI:")
        data = st.session_state.prediction_result["extracted_data"]
        st.info(f"""
        - **Wiek:** {data['wiek']} lat
        - **Płeć:** {'Mężczyzna' if data['plec'] == 'M' else 'Kobieta'}
        - **Tempo na 5km:** {data['tempo_5km']}
        """)

    with col2:
        st.write("#### Twój przewidywany czas netto:")
        st.metric(label="Półmaraton (21.0975 km)", value=st.session_state.prediction_result["predicted_time_str"])
        st.success("Powodzenia na starcie!", icon="🎉")

st.markdown("---")
st.info("Aplikacja wykorzystuje model AutoML (PyCaret) oraz model LLM (OpenAI) do analizy tekstu. Pamiętaj, że jest to tylko estymacja!", icon="ℹ️")
#Koniec aplikacji