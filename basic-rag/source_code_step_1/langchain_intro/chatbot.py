"""
Este módulo contiene la configuración y las plantillas para el chatbot que responde 
preguntas usando reseñas de pacientes.

Implementa un sistema completo con LangChain que combina:
- Recuperación de información (RAG)
- Agentes con herramientas
- Memoria de conversación
- Prompting estructurado
"""
import dotenv
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from tools import get_current_wait_time
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Ruta a la base de datos vectorial de reseñas
REVIEWS_CHROMA_PATH = "chroma_data/"

# Carga de variables de entorno (útil para claves API, etc.)
dotenv.load_dotenv()

#------------------------------------------------------------------------
# PARTE 1: CONFIGURACIÓN DEL CHAIN DE RESEÑAS (RAG - Retrieval Augmented Generation)
#------------------------------------------------------------------------

# Definimos la plantilla para indicar al LLM cómo debe utilizar el contexto recuperado
# Esta es una parte clave del "prompt engineering" en LangChain
REVIEW_TEMPLATE_STR = """Tu trabajo es utilizar las reseñas de pacientes
para responder preguntas sobre su experiencia en un hospital.
Usa el siguiente contexto para responder las preguntas.
Sé lo más detallado posible, pero no inventes información
que no esté en el contexto. Si no conoces una respuesta, di
que no lo sabes.
{context}
"""

# Creamos un mensaje de sistema usando la plantilla
# SystemMessagePromptTemplates son usados para instrucciones generales al LLM
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=REVIEW_TEMPLATE_STR
    )
)

# Creamos un mensaje humano para la pregunta del usuario
# HumanMessagePromptTemplate se usa para formatear la entrada del usuario
review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)

# Combinamos los mensajes en una secuencia
messages = [review_system_prompt, review_human_prompt]

# Creamos una plantilla de chat completa
# ChatPromptTemplate es un contenedor que organiza múltiples mensajes
review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

# Definimos el modelo de chat a utilizar
# ChatOllama permite usar modelos locales a través de Ollama
chat_model = ChatOllama(model="gemma3:12b", temperature=0)

# Definimos un parser para convertir la salida del LLM en string
output_parser = StrOutputParser()

# Cargamos la base de datos vectorial previamente creada
# Chroma es un almacén vectorial que permite búsquedas semánticas
reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
)

# Configuramos el retriever para obtener los documentos más relevantes
# El parámetro k define cuántos documentos similares recuperar
reviews_retriever = reviews_vector_db.as_retriever(k=10)

# Creamos la cadena de procesamiento RAG completa usando el operador | (pipe)
# Esta es una característica poderosa de LangChain: la composición de componentes
review_chain = (
    # Este diccionario combina el contexto recuperado con la pregunta del usuario
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template  # Formatea los mensajes con el contexto y la pregunta
    | chat_model              # Envía los mensajes al modelo de lenguaje
    | StrOutputParser()       # Convierte la respuesta en texto plano
)

#------------------------------------------------------------------------
# PARTE 2: CONFIGURACIÓN DEL AGENTE CON HERRAMIENTAS
#------------------------------------------------------------------------

# Definimos las herramientas que el agente puede utilizar
# Las Tools en LangChain permiten que el LLM realice acciones concretas
tools = [
    Tool(
        name="Reviews",  # Nombre que el agente usará para referirse a esta herramienta
        func=review_chain.invoke,  # Función que se ejecutará cuando se use la herramienta
        description="""Útil cuando necesites responder preguntas
        sobre reseñas de pacientes o experiencias en el hospital.
        No es útil para responder preguntas sobre detalles específicos
        de visitas como pagador, facturación, tratamiento, diagnóstico,
        queja principal, hospital o información del médico.
        Pasa la pregunta completa como entrada a la herramienta. Por ejemplo,
        si la pregunta es "¿Qué piensan los pacientes sobre el sistema de triaje?",
        la entrada debe ser "¿Qué piensan los pacientes sobre el sistema de triaje?"
        """,  # Descripción que ayuda al agente a decidir cuándo usar esta herramienta
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Úsala cuando se pregunte sobre los tiempos de espera actuales
        en un hospital específico. Esta herramienta solo puede obtener el tiempo de espera
        actual en un hospital y no tiene información sobre tiempos de espera
        agregados o históricos. Esta herramienta devuelve tiempos de espera en
        minutos. No pases la palabra "hospital" como entrada,
        solo el nombre del hospital en sí. Por ejemplo, si la pregunta es
        "¿Cuál es el tiempo de espera en el hospital A?", la entrada debe ser "A".
        """,
    ),
]

# Obtenemos un prompt predefinido del Hub de LangChain
# LangChain Hub contiene prompts y cadenas predefinidas listas para usar
hospital_agent_prompt = hub.pull("hwchase17/react-chat")

# Definimos el modelo para el agente
# Usamos temperatura=0 para respuestas más deterministas
agent_chat_model = ChatOllama(model="gemma3:12b", temperature=0)

# Creamos un diccionario para almacenar historiales de conversación
# Esto permite manejar múltiples sesiones concurrentes
message_history = {}

def get_session_history(session_id):
    """
    Obtiene o crea un historial de mensajes para una sesión específica.
    
    Esta función implementa la gestión de memoria en LangChain, permitiendo
    que el agente recuerde conversaciones anteriores.
    
    Args:
        session_id (str): Identificador único de la sesión
        
    Returns:
        ChatMessageHistory: Objeto que contiene el historial de la conversación
    """
    if session_id not in message_history:
        message_history[session_id] = ChatMessageHistory()
    return message_history[session_id]

# Creamos el agente utilizando el framework ReAct
# ReAct (Reasoning and Acting) permite al LLM razonar paso a paso
hospital_agent = create_react_agent(
    llm=agent_chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

# Envolvemos el agente en un ejecutor que manejará la ejecución de herramientas
# AgentExecutor coordina la interacción entre el agente y sus herramientas
hospital_agent_executor = AgentExecutor(
    agent=hospital_agent,
    tools=tools,
    verbose=True,  # Muestra los pasos intermedios en la consola
    return_intermediate_steps=True,  # Permite acceder a los pasos de razonamiento
)

# Añadimos capacidad de memoria al agente
# RunnableWithMessageHistory integra el historial de conversación con el agente
hospital_agent_with_history = RunnableWithMessageHistory(
    hospital_agent_executor,
    get_session_history,
    input_messages_key="input",  # Clave donde se encuentra la entrada del usuario
    history_messages_key="chat_history",  # Clave donde se almacena el historial
)

#------------------------------------------------------------------------
# PARTE 3: INTERFAZ DE USUARIO PARA EL CHATBOT
#------------------------------------------------------------------------

# Bucle interactivo para conversar con el agente
if __name__ == "__main__":
    print("\n--- Chatbot de Hospital iniciado ---")
    print("Escribe 'salir' para terminar la conversación")
    
    while True:
        user_input = input("\nTú: ")
        
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("\n--- Finalizando chatbot ---")
            break
        
        try:
            # Ejecutar el agente con la entrada del usuario
            # Invocamos al agente con la entrada y la configuración de sesión
            result = hospital_agent_with_history.invoke(
                {"input": user_input}, 
                {"configurable": {"session_id": "default"}}
            )
            
            # Mostrar la respuesta
            print(f"\nAsistente: {result['output']}")
            
            # Opcionalmente, mostrar los pasos intermedios para debugging
            if "--debug" in user_input:
                print("\n--- Pasos intermedios ---")
                if "intermediate_steps" in result:
                    for step in result["intermediate_steps"]:
                        print(f"Acción: {step[0].tool}")
                        print(f"Entrada: {step[0].tool_input}")
                        print(f"Resultado: {step[1]}")
                        print("---")
                else:
                    print("No hay pasos intermedios disponibles")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
