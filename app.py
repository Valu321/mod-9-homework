import streamlit as st
import pandas as pd
import openai
import os
import json
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import io
from dotenv import load_dotenv
import pandera as pa
from pandera.errors import SchemaError
from langfuse import Langfuse
from pycaret.regression import load_model, predict_model

# --- Stałe (Constants) ---
APP_TITLE = "Estymator Czasu Półmaratonu"
APP_ICON = "🏃‍♂️"
DO_SPACES_ENDPOINT_URL_DEFAULT = 'https://fra1.digitaloceanspaces.com'
MODEL_FILE_KEY = 'models/halfmarathon_pipeline.pkl'
DOWNLOADED_MODEL_FILENAME = 'downloaded_model.pkl'
DOWNLOADED_MODEL_NAME = 'downloaded_model'
OPENAI_MODEL = "gpt-3.5-turbo-0125"
LANGFUSE_HOST = "https://cloud.langfuse.com"
DEFAULT_USER_INPUT = "Cześć, mam 33 lata, jestem mężczyzną. Biegam 5km w 24 minuty i 15 sekund."
IMAGE_URL = "https://mod-9-homework.fra1.digitaloceanspaces.com/Image_19f2o619f2o619f2.png"

# --- Konfiguracja strony ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Wczytywanie konfiguracji i inicjalizacja ---
@st.cache_data
def load_app_config():
    """Wczytuje konfigurację z pliku .env i zwraca jako słownik."""
    load_dotenv()
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGFUSE_PUBLIC_KEY": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": os.getenv("LANGFUSE_SECRET_KEY"),
        "DO_SPACES_KEY": os.getenv('DO_SPACES_KEY'),
        "DO_SPACES_SECRET": os.getenv('DO_SPACES_SECRET'),
        "DO_SPACES_ENDPOINT_URL": os.getenv('DO_SPACES_ENDPOINT_URL', DO_SPACES_ENDPOINT_URL_DEFAULT),
        "DO_SPACES_BUCKET": os.getenv('DO_SPACES_BUCKET'),
    }

config = load_app_config()

# Inicjalizacja klientów
if config.get("OPENAI_API_KEY"):
    openai.api_key = config["OPENAI_API_KEY"]

langfuse = None
if config.get("LANGFUSE_PUBLIC_KEY") and config.get("LANGFUSE_SECRET_KEY"):
    langfuse = Langfuse(
        public_key=config["LANGFUSE_PUBLIC_KEY"],
        secret_key=config["LANGFUSE_SECRET_KEY"],
        host=LANGFUSE_HOST
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
    "wiek": pa.Column(int, checks=pa.Check.in_range(1, 100), nullable=False),
    "plec": pa.Column(str, checks=pa.Check.isin(['K', 'M']), nullable=False),
    "tempo_5km": pa.Column(str, checks=pa.Check.str_matches(r'^\d{1,2}:\d{2}$'), nullable=False),
})

# --- Funkcje pomocnicze ---
@st.cache_resource
def get_boto_client(_config: dict):
    """Tworzy i zwraca klienta boto3 do interakcji z DigitalOcean Spaces."""
    session = boto3.Session()
    return session.client(
        's3',
        config=Config(s3={'addressing_style': 'path'}),
        region_name=_config["DO_SPACES_ENDPOINT_URL"].split('//')[1].split('.')[0],
        endpoint_url=_config["DO_SPACES_ENDPOINT_URL"],
        aws_access_key_id=_config["DO_SPACES_KEY"],
        aws_secret_access_key=_config["DO_SPACES_SECRET"]
    )

@st.cache_resource
def load_model_from_spaces(_config: dict):
    """Pobiera i wczytuje pipeline PyCaret z DigitalOcean Spaces."""
    try:
        client = get_boto_client(_config)
        client.download_file(_config["DO_SPACES_BUCKET"], MODEL_FILE_KEY, DOWNLOADED_MODEL_FILENAME)
        model = load_model(DOWNLOADED_MODEL_NAME)
        os.remove(DOWNLOADED_MODEL_FILENAME)
        return model
    except ClientError as e:
        if e.response['Error']['Code'] == '403':
            st.error("Błąd autoryzacji: Sprawdź klucze API i uprawnienia do bucketu.", icon="🚨")
        else:
            st.error(f"Nie udało się załadować modelu (błąd klienta): {e}", icon="🚨")
        return None
    except Exception as e:
        st.error(f"Nie udało się załadować modelu (błąd ogólny): {e}", icon="🚨")
        return None

def time_str_to_seconds(time_str: str) -> int | None:
    """Konwertuje czas w formacie 'MM:SS' na sekundy."""
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except (ValueError, AttributeError):
        return None

def format_time_from_seconds(total_seconds: float) -> str:
    """Formatuje sekundy do formatu 'HH:MM:SS'."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def extract_data_with_llm(user_input: str, _config: dict, lf_client: Langfuse | None):
    """Używa LLM do ekstrakcji danych z tekstu użytkownika."""
    trace = lf_client.trace(name="data-extraction", input=user_input) if lf_client else None
    
    if not _config.get("OPENAI_API_KEY"):
        st.error("Klucz API OpenAI nie jest skonfigurowany.", icon="🔑")
        return None

    system_prompt = """Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyekstrahowanie trzech informacji z tekstu użytkownika: wieku, płci oraz tempa biegu na 5km. Zwróć odpowiedź wyłącznie w formacie JSON.
- Wiek (`wiek`) powinien być liczbą całkowitą.
- Płeć (`plec`) powinna być jedną z dwóch wartości: 'M' (mężczyzna) lub 'K' (kobieta).
- Tempo na 5km (`tempo_5km`) powinno być w formacie "MM:SS".
Jeśli którejś informacji brakuje, ustaw dla niej wartość null. Upewnij się, że odpowiedź to poprawny obiekt JSON."""
    
    try:
        generation = trace.generation(name="llm-extraction", input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}], model=OPENAI_MODEL) if trace else None
        response = openai.chat.completions.create(model=OPENAI_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}], response_format={"type": "json_object"})
        
        content = response.choices[0].message.content
        if not content:
            if trace: trace.score(name="data-extraction-quality", value=0, comment="LLM returned empty content.")
            return None
        
        result = json.loads(content)
        if generation: generation.end(output=result, usage=response.usage)
        return result
        
    except Exception as e:
        st.error(f"Błąd podczas komunikacji z OpenAI: {e}", icon="🔥")
        if trace: trace.score(name="data-extraction-quality", value=0, comment=f"Exception: {e}")
        return None

def run_prediction_pipeline(user_description: str, model, _config: dict, lf_client: Langfuse | None):
    """Orkiestruje cały proces od ekstrakcji danych po predykcję, zwracając wynik lub błąd."""
    extracted_data = extract_data_with_llm(user_description, _config, lf_client)
    if not extracted_data or all(value is None for value in extracted_data.values()):
        return {"error": "Nie udało mi się znaleźć potrzebnych informacji w Twoim opisie. Upewnij się, że podałeś/aś swój **wiek**, **płeć** oraz **tempo na 5km**."}

    try:
        validation_df = pd.DataFrame([extracted_data])
        llm_output_schema.validate(validation_df)
        
        czas_5km_s = time_str_to_seconds(extracted_data["tempo_5km"])
        if czas_5km_s is None:
            raise ValueError("Niepoprawny format tempa na 5km. Oczekiwano 'MM:SS'.")
        
        tempo_1km_s = czas_5km_s / 5
        input_data = {'wiek': [extracted_data["wiek"]], 'plec': [extracted_data["plec"]], 'tempo_5km_s_na_km': [tempo_1km_s]}
        input_df = pd.DataFrame(input_data).astype({'wiek': 'int16', 'plec': 'category', 'tempo_5km_s_na_km': 'float32'})
        
        predictions = predict_model(model, data=input_df)
        prediction_s = predictions['prediction_label'].iloc[0]
        
        return {"extracted_data": extracted_data, "predicted_time_str": format_time_from_seconds(prediction_s)}

    except SchemaError as err:
        # err.failure_cases może być None, więc dodajemy sprawdzenie, aby uniknąć błędu.
        if err.failure_cases is not None and not err.failure_cases.empty:
            error_details = err.failure_cases['failure_case'].iloc[0]
            return {"error": f"Znalazłem błąd w podanych danych: {error_details}"}
        # Zapewniamy ogólny komunikat błędu jako fallback.
        return {"error": f"Błąd walidacji danych. Sprawdź format (np. tempo 'MM:SS'). Szczegóły: {err}"}
    except Exception as e:
        return {"error": f"Wystąpił nieoczekiwany błąd: {e}"}

# --- Inicjalizacja stanu sesji ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = DEFAULT_USER_INPUT

# --- Główna aplikacja Streamlit ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image(IMAGE_URL, width=150)
with col2:
    st.title(APP_TITLE)
    st.markdown("Opisz siebie w polu poniżej, a ja oszacuję Twój przewidywany czas na mecie! Podaj swój **wiek**, **płeć** oraz **tempo na 5 km**.")

# --- Ładowanie modelu (w tle) ---
pipeline = load_model_from_spaces(config)

# --- Formularz wejściowy ---
st.write("### 🏃 Krok 1: Opowiedz nam o sobie")
user_description = st.text_area("Przedstaw się:", value=st.session_state.user_input, height=100, label_visibility="collapsed")

col1_form, col2_form, _ = st.columns([1, 1, 3])
predict_button = col1_form.button("Szacuj czas", type="primary", use_container_width=True)
clear_button = col2_form.button("Wyczyść", use_container_width=True)

if clear_button:
    st.session_state.prediction_result = None
    st.session_state.user_input = ""
    st.rerun()

if predict_button:
    st.session_state.user_input = user_description
    if not user_description:
        st.warning("Proszę, opisz siebie w polu tekstowym.", icon="⚠️")
    elif pipeline is None:
        st.error("Model predykcyjny nie jest dostępny. Sprawdź komunikaty powyżej lub skontaktuj się z administratorem.", icon="🚨")
    else:
        with st.spinner("Analizuję Twoje dane i liczę... 🤖"):
            result = run_prediction_pipeline(user_description, pipeline, config, langfuse)
            if "error" in result:
                st.error(result["error"], icon="😟")
                st.session_state.prediction_result = None
            else:
                st.session_state.prediction_result = result

# --- Wyświetlanie wyników ---
if st.session_state.prediction_result:
    st.write("---")
    st.write("### 📈 Krok 2: Analiza i wynik")
    
    result_data = st.session_state.prediction_result
    # Dodajemy sprawdzenie, czy klucze na pewno istnieją w słowniku
    if "extracted_data" in result_data and "predicted_time_str" in result_data:
        col1_res, col2_res = st.columns(2)
        
        with col1_res:
            st.write("#### Dane zinterpretowane przez AI:")
            data = result_data["extracted_data"]
            # Użycie .get() jest bezpieczniejsze i rozwiązuje problemy lintera
            plec = data['plec'] if isinstance(data, dict) else data
            plec_str = 'Mężczyzna' if plec == 'M' else 'Kobieta' if plec == 'K' else 'B/D'
            
            st.info(f"""
            - **Wiek:** {data.get('wiek', 'B/D') if isinstance(data, dict) else data} lat
            - **Płeć:** {plec_str}
            - **Tempo na 5km:** {data.get('tempo_5km', 'B/D') if isinstance(data, dict) else 'B/D'}
            """)

        with col2_res:
            st.write("#### Twój przewidywany czas netto:")
            st.metric(label="Półmaraton (21.0975 km)", value=result_data["predicted_time_str"])
            st.success("Powodzenia na starcie!", icon="🎉")

st.markdown("---")
st.info("Aplikacja wykorzystuje model AutoML (PyCaret) oraz model LLM (OpenAI) do analizy tekstu. Pamiętaj, że jest to tylko estymacja!", icon="ℹ️")
