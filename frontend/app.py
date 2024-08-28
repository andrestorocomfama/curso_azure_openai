import streamlit as st
from streamlit_chat import message
from PIL import Image
from helper_functions import *  # Aquí se tienen las funciones auxiliares desarrolladas por cada equipo
import time
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Cargar las variables de entorno desde un archivo .env
load_dotenv(find_dotenv(), override=True)

# Configuración de los parámetros de conexión a Azure AI Search
search_endpoint: str = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
search_api_key: str = os.environ["AZURE_AI_SEARCH_API_KEY"]
index_name: str = os.environ["AZURE_AI_SEARCH_INDEX"]

# Configuración del retriever de LangChain con Azure AI Search
retriever = AzureAISearchRetriever(
    content_key="content",
    top_k=3,
    index_name=index_name,
    api_key=search_api_key,
    service_name=search_endpoint,
)

# Configuración del modelo de lenguaje Azure OpenAI
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
    deployment_name=os.environ["AZURE_DEPLOYMENT_CHAT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# Definición del prompt del sistema para el asistente
system_prompt = """Eres un asistente para tareas de preguntas y respuestas. Usa los siguientes fragmentos de contexto recuperado para responder la pregunta. Si no sabes la respuesta, simplemente di que no la sabes. Usa un máximo de tres oraciones y mantén la respuesta concisa. Debes responder en español.
\n\n
{context}
"""

# Ejemplos de mensajes para el asistente (pocos ejemplos)
examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "1 🦜 0", "output": "1"},
]

# Configuración del prompt de ejemplo para el few-shot learning
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Configuración del prompt para few-shot learning
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Definición de los mensajes para el prompt de chat
messages = [
    ("system", system_prompt),
    few_shot_prompt,
    MessagesPlaceholder("chat_history"),
    ("user", "{input}")
]

# Creación del prompt template para el modelo de lenguaje
prompt_template = ChatPromptTemplate.from_messages(messages)

# Configuración de la cadena de pregunta-respuesta utilizando LangChain
question_answer_chain = create_stuff_documents_chain(model, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


############################## FRONTEND ###########################################

# Configuración de la interfaz de usuario de Streamlit
img = Image.open('frontend/styles/logo.png')
st.set_page_config(
    page_title="Asistente virtual Comfama",
    page_icon=img,
    initial_sidebar_state="auto",
    layout='wide',
)

# Espacio para el logo y saludo inicial
placeholder = st.empty()
col1, col2, col3 = st.columns([2, 5, 2])

with col1:
    placeholder.image("frontend/styles/logo.png", width=120)

with col2:
    # Definición de estilos de la página
    page_styles = """
    <style>
        .stApp { background-color: #F0F0F0; }
        h1, h2, h3 { color: #fb6903; }
        p, li, label, footer { color: #565656; }
        button { background-color: #5CFFE0; color: white; border: none; }
        .streamlit-chat-footer { background-color: #3E6A27; color: white; }
        .streamlit-chat-message-container { background-color: #DCDCDC; }
        .streamlit-chat-input-container { background-color: #3E6A27; color: white; }
        .streamlit-chat-message { background-color: #565656; color: white; }
        div[data-testid="stSidebar"] { background-color: #f0f0f0; }
        div[data-testid="stButton"] > button { background-color: #5CFFE0; color: white; }
    </style>
    """
    st.markdown(page_styles, unsafe_allow_html=True)

    # Mostrar saludo inicial
    saludo = obtener_saludo()
    st.image('frontend/styles/euler.png', width=130)
    st.write(f"**{saludo}**\n Soy Euler, el asistente virtual de Comfama.\n Estoy acá para responder sus preguntas sobre los artículos de investigación que hay en mi base de conocimientos.\n\n ¿En qué puedo ayudarte hoy?")

    # Botón para iniciar una nueva conversación
    if st.button("Iniciar nueva conversación"):
        st.session_state.messages = []

    # Inicializar variables de sesión
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_interaction" not in st.session_state:
        st.session_state.last_interaction = time.time()

    # Función para actualizar el tiempo de la última interacción
    def update_last_interaction():
        st.session_state.last_interaction = time.time()

    # Mostrar mensajes anteriores en el chat
    for message in st.session_state.messages:
        avatar_image = Image.open('frontend/styles/euler.png') if message["role"] == "assistant" else Image.open('frontend/styles/user.png')
        with st.chat_message(message["role"], avatar=avatar_image):
            st.markdown(message["content"])

# Manejo de entrada de texto del usuario
if prompt := st.chat_input("Escribe aquí tu pregunta"):
    with col2:
        # Chequeo de inactividad y cierre de la sesión si es necesario
        if (time.time() - st.session_state.last_interaction > 60 * 15) or (not prompt):
            st.warning("No se ha detectado actividad en los últimos 10 minutos. La aplicación se detendrá.")
            st.stop()

        # Agregar el mensaje del usuario al historial de mensajes
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=Image.open('frontend/styles/user.png')):
            st.markdown(prompt)

        # Actualizar tiempo de última interacción
        update_last_interaction()

        # Procesar la pregunta del usuario usando LangChain
        with st.chat_message("assistant", avatar=Image.open('frontend/styles/euler.png')):
            try:
                with st.spinner('Espere...'):
                    stream = rag_chain.stream({"input": prompt, "chat_history": st.session_state.messages})

                    def stream_data():
                        for chunk in stream:
                            if "answer" in chunk:
                                yield chunk["answer"]
                                time.sleep(0.02)

                    response = st.write_stream(stream_data)
                    
                # Agregar la respuesta del asistente al historial de mensajes
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Ocurrió un error, vuelva a intentarlo. Error: {e}")



# app.py  import streamlit as st  # Your Streamlit app code here  if __name__ == '__main__':     st.set_option('server.enableCORS', True)