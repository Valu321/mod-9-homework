"""
Estymator Czasu Półmaratonu - Aplikacja Streamlit

Aplikacja wykorzystująca model AutoML (PyCaret) oraz model LLM (OpenAI) 
do analizy tekstu i predykcji czasu półmaratonu na podstawie wieku, 
płci i tempa na 5km.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import boto3
import openai
import pandas as pd
import pandera as pa
import streamlit as st
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langfuse import Langfuse
from pandera.errors import SchemaError
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

# --- Wyjątki aplikacji ---
class AppConfigError(Exception):
    """Wyjątek rzucany gdy konfiguracja aplikacji jest nieprawidłowa."""
    pass

class ModelLoadError(Exception):
    """Wyjątek rzucany gdy nie można załadować modelu."""
    pass

class DataExtractionError(Exception):
    """Wyjątek rzucany gdy nie można wyekstrahować danych z tekstu."""
    pass

# --- Klasy konfiguracji ---
@dataclass
class AppConfig:
    """Klasa przechowująca konfigurację aplikacji."""
    openai_api_key: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_secret_key: Optional[str]
    do_spaces_key: Optional[str]
    do_spaces_secret: Optional[str]
    do_spaces_endpoint_url: str
    do_spaces_bucket: Optional[str]
    
    @classmethod
    def from_env(cls) -> AppConfig:
        """Tworzy instancję konfiguracji z zmiennych środowiskowych."""
        load_dotenv()
        
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            do_spaces_key=os.getenv('DO_SPACES_KEY'),
            do_spaces_secret=os.getenv('DO_SPACES_SECRET'),
            do_spaces_endpoint_url=os.getenv('DO_SPACES_ENDPOINT_URL', DO_SPACES_ENDPOINT_URL_DEFAULT),
            do_spaces_bucket=os.getenv('DO_SPACES_BUCKET'),
        )
    
    def validate(self) -> None:
        """Sprawdza czy konfiguracja jest kompletna."""
        required_fields = [
            ('openai_api_key', 'OPENAI_API_KEY'),
            ('do_spaces_key', 'DO_SPACES_KEY'),
            ('do_spaces_secret', 'DO_SPACES_SECRET'),
            ('do_spaces_bucket', 'DO_SPACES_BUCKET'),
        ]
        
        missing_fields = []
        for field, env_name in required_fields:
            if not getattr(self, field):
                missing_fields.append(env_name)
        
        if missing_fields:
            raise AppConfigError(
                f"Brakujące wymagane zmienne środowiskowe: {', '.join(missing_fields)}"
            )

@dataclass
class PredictionResult:
    """Klasa przechowująca wynik predykcji."""
    extracted_data: Dict[str, Any]
    predicted_time_str: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PredictionResult:
        """Tworzy instancję z słownika."""
        return cls(
            extracted_data=data["extracted_data"],
            predicted_time_str=data["predicted_time_str"]
        )

# --- Schemat walidacji Pandera ---
llm_output_schema = pa.DataFrameSchema({
    "wiek": pa.Column(int, checks=pa.Check.in_range(1, 100), nullable=False),
    "plec": pa.Column(str, checks=pa.Check.isin(['K', 'M']), nullable=False),
    "tempo_5km": pa.Column(str, checks=pa.Check.str_matches(r'^\d{1,2}:\d{2}$'), nullable=False),
})

# --- Style CSS ---
CSS_STYLES = """
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
"""

# --- Funkcje pomocnicze ---
def time_str_to_seconds(time_str: str) -> Optional[int]:
    """
    Konwertuje czas w formacie 'MM:SS' na sekundy.
    
    Args:
        time_str: Czas w formacie 'MM:SS'
        
    Returns:
        Liczba sekund lub None jeśli format jest nieprawidłowy
    """
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except (ValueError, AttributeError):
        return None

def format_time_from_seconds(total_seconds: float) -> str:
    """
    Formatuje sekundy do formatu 'HH:MM:SS'.
    
    Args:
        total_seconds: Liczba sekund
        
    Returns:
        Czas w formacie 'HH:MM:SS'
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_plec_display_name(plec: str) -> str:
    """
    Konwertuje kod płci na nazwę wyświetlaną.
    
    Args:
        plec: Kod płci ('M' lub 'K')
        
    Returns:
        Nazwa płci do wyświetlenia
    """
    return 'Mężczyzna' if plec == 'M' else 'Kobieta' if plec == 'K' else 'B/D'

# --- Klasy serwisów ---
class DigitalOceanService:
    """Serwis do interakcji z DigitalOcean Spaces."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        """Tworzy i zwraca klienta boto3."""
        if self._client is None:
            session = boto3.Session()
            self._client = session.client(
                's3',
                config=Config(s3={'addressing_style': 'path'}),
                region_name=self.config.do_spaces_endpoint_url.split('//')[1].split('.')[0],
                endpoint_url=self.config.do_spaces_endpoint_url,
                aws_access_key_id=self.config.do_spaces_key,
                aws_secret_access_key=self.config.do_spaces_secret
            )
        return self._client
    
    def download_model(self) -> str:
        """
        Pobiera model z DigitalOcean Spaces.
        
        Returns:
            Ścieżka do pobranego pliku
            
        Raises:
            ModelLoadError: Gdy nie można pobrać modelu
        """
        try:
            self.client.download_file(
                self.config.do_spaces_bucket, 
                MODEL_FILE_KEY, 
                DOWNLOADED_MODEL_FILENAME
            )
            return DOWNLOADED_MODEL_FILENAME
        except ClientError as e:
            if e.response['Error']['Code'] == '403':
                raise ModelLoadError("Błąd autoryzacji: Sprawdź klucze API i uprawnienia do bucketu.")
            else:
                raise ModelLoadError(f"Nie udało się pobrać modelu: {e}")
        except Exception as e:
            raise ModelLoadError(f"Nieoczekiwany błąd podczas pobierania modelu: {e}")

class LLMService:
    """Serwis do interakcji z modelem językowym."""
    
    def __init__(self, config: AppConfig, langfuse_client: Optional[Langfuse] = None):
        self.config = config
        self.langfuse_client = langfuse_client
        
        if config.openai_api_key:
            openai.api_key = config.openai_api_key
    
    def extract_data(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Ekstrahuje dane z tekstu użytkownika używając LLM.
        
        Args:
            user_input: Tekst wprowadzony przez użytkownika
            
        Returns:
            Słownik z wyekstrahowanymi danymi lub None w przypadku błędu
        """
        if not self.config.openai_api_key:
            raise DataExtractionError("Klucz API OpenAI nie jest skonfigurowany.")
        
        trace = None
        if self.langfuse_client:
            trace = self.langfuse_client.trace(name="data-extraction", input=user_input)
        
        system_prompt = """Jesteś ekspertem w analizie tekstu. Twoim zadaniem jest wyekstrahowanie trzech informacji z tekstu użytkownika: wieku, płci oraz tempa biegu na 5km. Zwróć odpowiedź wyłącznie w formacie JSON.
- Wiek (`wiek`) powinien być liczbą całkowitą.
- Płeć (`plec`) powinna być jedną z dwóch wartości: 'M' (mężczyzna) lub 'K' (kobieta).
- Tempo na 5km (`tempo_5km`) powinno być w formacie "MM:SS".
Jeśli którejś informacji brakuje, ustaw dla niej wartość null. Upewnij się, że odpowiedź to poprawny obiekt JSON."""
        
        try:
            generation = None
            if trace:
                generation = trace.generation(
                    name="llm-extraction",
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    model=OPENAI_MODEL
                )
            
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not content:
                if trace:
                    trace.score(name="data-extraction-quality", value=0, comment="LLM returned empty content.")
                return None
            
            result = json.loads(content)
            
            if generation:
                generation.end(output=result, usage=response.usage)
            
            return result
            
        except Exception as e:
            if trace:
                trace.score(name="data-extraction-quality", value=0, comment=f"Exception: {e}")
            raise DataExtractionError(f"Błąd podczas komunikacji z OpenAI: {e}")

class PredictionService:
    """Serwis do wykonywania predykcji."""
    
    def __init__(self, model, llm_service: LLMService):
        self.model = model
        self.llm_service = llm_service
    
    def predict(self, user_description: str) -> Union[PredictionResult, Dict[str, str]]:
        """
        Wykonuje pełny pipeline predykcji.
        
        Args:
            user_description: Opis użytkownika
            
        Returns:
            Wynik predykcji lub słownik z błędem
        """
        try:
            extracted_data = self.llm_service.extract_data(user_description)
            if not extracted_data or all(value is None for value in extracted_data.values()):
                return {"error": "Nie udało mi się znaleźć potrzebnych informacji w Twoim opisie. Upewnij się, że podałeś/aś swój **wiek**, **płeć** oraz **tempo na 5km**."}
            
            # Walidacja danych
            validation_df = pd.DataFrame([extracted_data])
            llm_output_schema.validate(validation_df)
            
            # Konwersja tempa
            czas_5km_s = time_str_to_seconds(extracted_data["tempo_5km"])
            if czas_5km_s is None:
                raise ValueError("Niepoprawny format tempa na 5km. Oczekiwano 'MM:SS'.")
            
            tempo_1km_s = czas_5km_s / 5
            
            # Przygotowanie danych do predykcji
            input_data = {
                'wiek': [extracted_data["wiek"]],
                'plec': [extracted_data["plec"]],
                'tempo_5km_s_na_km': [tempo_1km_s]
            }
            input_df = pd.DataFrame(input_data).astype({
                'wiek': 'int16',
                'plec': 'category',
                'tempo_5km_s_na_km': 'float32'
            })
            
            # Predykcja
            predictions = predict_model(self.model, data=input_df)
            prediction_s = predictions['prediction_label'].iloc[0]
            
            return PredictionResult(
                extracted_data=extracted_data,
                predicted_time_str=format_time_from_seconds(prediction_s)
            )
            
        except SchemaError as err:
            if err.failure_cases is not None and not err.failure_cases.empty:
                error_details = err.failure_cases['failure_case'].iloc[0]
                return {"error": f"Znalazłem błąd w podanych danych: {error_details}"}
            return {"error": f"Błąd walidacji danych. Sprawdź format (np. tempo 'MM:SS'). Szczegóły: {err}"}
        except Exception as e:
            return {"error": f"Wystąpił nieoczekiwany błąd: {e}"}

# --- Inicjalizacja aplikacji ---
@st.cache_data
def load_app_config() -> AppConfig:
    """Wczytuje i waliduje konfigurację aplikacji."""
    config = AppConfig.from_env()
    try:
        config.validate()
    except AppConfigError as e:
        st.error(f"Błąd konfiguracji: {e}", icon="🔧")
        st.stop()
    return config

@st.cache_resource
def load_model_from_spaces(config: AppConfig):
    """Pobiera i wczytuje pipeline PyCaret z DigitalOcean Spaces."""
    try:
        do_service = DigitalOceanService(config)
        model_path = do_service.download_model()
        model = load_model(DOWNLOADED_MODEL_NAME)
        os.remove(model_path)
        return model
    except ModelLoadError as e:
        st.error(str(e), icon="🚨")
        return None
    except Exception as e:
        st.error(f"Nie udało się załadować modelu: {e}", icon="🚨")
        return None

def initialize_services(config: AppConfig, model) -> tuple[LLMService, PredictionService]:
    """Inicjalizuje serwisy aplikacji."""
    langfuse_client = None
    if config.langfuse_public_key and config.langfuse_secret_key:
        langfuse_client = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=LANGFUSE_HOST
        )
    
    llm_service = LLMService(config, langfuse_client)
    prediction_service = PredictionService(model, llm_service)
    
    return llm_service, prediction_service

# --- Główna aplikacja Streamlit ---
def main():
    """Główna funkcja aplikacji."""
    # Konfiguracja strony
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Style CSS
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    # Inicjalizacja konfiguracji i modelu
    config = load_app_config()
    model = load_model_from_spaces(config)
    
    if model is None:
        st.error("Nie można załadować modelu. Sprawdź konfigurację i spróbuj ponownie.", icon="🚨")
        st.stop()
    
    # Inicjalizacja serwisów
    llm_service, prediction_service = initialize_services(config, model)
    
    # Inicjalizacja stanu sesji
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = DEFAULT_USER_INPUT
    
    # Interfejs użytkownika
    render_header()
    render_input_form(prediction_service)
    render_results()

def render_header():
    """Renderuje nagłówek aplikacji."""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(IMAGE_URL, width=150)
    with col2:
        st.title(APP_TITLE)
        st.markdown("Opisz siebie w polu poniżej, a ja oszacuję Twój przewidywany czas na mecie! Podaj swój **wiek**, **płeć** oraz **tempo na 5 km**.")

def render_input_form(prediction_service: PredictionService):
    """Renderuje formularz wejściowy."""
    st.write("### 🏃 Krok 1: Opowiedz nam o sobie")
    user_description = st.text_area(
        "Przedstaw się:",
        value=st.session_state.user_input,
        height=100,
        label_visibility="collapsed"
    )
    
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
        else:
            with st.spinner("Analizuję Twoje dane i liczę... 🤖"):
                result = prediction_service.predict(user_description)
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"], icon="😟")
                    st.session_state.prediction_result = None
                else:
                    st.session_state.prediction_result = result

def render_results():
    """Renderuje wyniki predykcji."""
    if st.session_state.prediction_result:
        st.write("---")
        st.write("### 📈 Krok 2: Analiza i wynik")
        
        result_data = st.session_state.prediction_result
        if isinstance(result_data, PredictionResult):
            col1_res, col2_res = st.columns(2)
            
            with col1_res:
                st.write("#### Dane zinterpretowane przez AI:")
                data = result_data.extracted_data
                plec_str = get_plec_display_name(data.get('plec', ''))
                
                st.info(f"""
                - **Wiek:** {data.get('wiek', 'B/D')} lat
                - **Płeć:** {plec_str}
                - **Tempo na 5km:** {data.get('tempo_5km', 'B/D')}
                """)
            
            with col2_res:
                st.write("#### Twój przewidywany czas netto:")
                st.metric(label="Półmaraton (21.0975 km)", value=result_data.predicted_time_str)
                st.success("Powodzenia na starcie!", icon="🎉")
    
    st.markdown("---")
    st.info("Aplikacja wykorzystuje model AutoML (PyCaret) oraz model LLM (OpenAI) do analizy tekstu. Pamiętaj, że jest to tylko estymacja!", icon="ℹ️")

if __name__ == "__main__":
    main()
