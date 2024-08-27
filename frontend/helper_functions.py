import os
import datetime

from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(r'.env', override=True)

client = AzureOpenAI(
   azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
)

deployment_chat=os.environ['AZURE_DEPLOYMENT_CHAT']
deployment_emb=os.environ['AZURE_OPENAI_DEPLOYMENT_ID_EMBEDDINGS']

model_chat = os.environ['AZURE_DEPLOYMENT_CHAT']
model_emb = os.environ['AZURE_OPENAI_EMBEDDING_MODEL_NAME']

def obtener_saludo():
    # Determina el saludo apropiado según la hora del día
    hora_actual = datetime.datetime.now().hour
    if 6 <= hora_actual < 12:
        return "¡Buenos días!"
    elif 12 <= hora_actual < 18:
        return "¡Buenas tardes!"
    else:
        return "¡Buenas noches!"