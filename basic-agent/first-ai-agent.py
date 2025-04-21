"""Implementación de un agente de análisis de texto usando LangGraph.

Este agente clasifica texto, extrae entidades y lo resume.

Ejemplo de: https://medium.com/data-science-collective/the-complete-guide-to-building-your-first-ai-agent-its-easier-than-you-think-c87f376c84b2
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_ollama import ChatOllama

# --- Constantes --- #
MODEL_NAME = "gemma3:12b" # Nombre del modelo LLM a utilizar
TEMPERATURE = 0 # Temperatura para la generación del LLM

CATEGORIES = ["Noticias", "Blog", "Investigación", "Tecnología", "Otro"]
ENTITY_TYPES = ["Persona", "Organización", "Ubicación"]

CLASSIFICATION_TEMPLATE = PromptTemplate(
    input_variables=["text", "categories"],
    template="""
        Clasifica el siguiente texto en una de estas categorías: {categories}.
        No justifiques las respuestas.

        Texto: {text}

        Clasificación:
    """
)

ENTITY_EXTRACTION_TEMPLATE = PromptTemplate(
    input_variables=["text", "entity_types"],
    template="""
        Extrae todas las entidades ({entity_types}) del siguiente texto.
        Proporciona el resultado como una lista separada por comas.

        Texto: {text}

        Entidades:
    """
)

SUMMARIZATION_TEMPLATE = PromptTemplate.from_template(
    """
        Resume el siguiente texto en una oración corta.

        Texto: {text}

        Resumen:
    """
)

# --- Inicialización del LLM --- #
# Initialize the ChatOllama instance, specify the model if needed
llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)

# --- Definición del Estado --- #
class State(TypedDict):
    """Diccionario de estado."""
    text: str  # Almacena el texto de entrada original
    classification: str  # Representa el resultado de la clasificación (por ejemplo, etiqueta de categoría)
    entities: List[str]  # Almacena una lista de entidades extraídas (por ejemplo, entidades nombradas)
    summary: str  # Almacena una versión resumida del texto

# --- Nodos del Grafo --- #
def classification_node(state: State) -> dict:
    """Clasifica el texto usando el LLM.

    Args:
        state (State): El estado actual con el texto a clasificar.

    Returns:
        dict: Un diccionario con la clave "classification".
    """
    # Formatea la plantilla de prompt con el texto de entrada y categorías
    prompt_formatted = CLASSIFICATION_TEMPLATE.format(
        text=state["text"],
        categories=", ".join(CATEGORIES)
    )
    message = HumanMessage(content=prompt_formatted)
    # Invoca el modelo de lenguaje para clasificar el texto
    classification = llm.invoke([message]).content.strip()
    # Retorna el resultado de la clasificación
    return {"classification": classification}

def entity_extraction_node(state: State) -> dict:
    """Extrae entidades nombradas del texto usando el LLM.

    Args:
        state (State): El estado actual con el texto a analizar.

    Returns:
        dict: Un diccionario con la clave "entities".
    """
    # Formatea la plantilla de prompt con el texto y tipos de entidad
    prompt_formatted = ENTITY_EXTRACTION_TEMPLATE.format(
        text=state["text"],
        entity_types=", ".join(ENTITY_TYPES)
    )
    message = HumanMessage(content=prompt_formatted)
    # Envía al modelo, limpia y divide en lista
    entities_str = llm.invoke([message]).content.strip()
    entities = [entity.strip() for entity in entities_str.split(",") if entity.strip()] # List comprehension para limpiar
    # Retorna el diccionario con la lista de entidades
    return {"entities": entities}

def summarize_node(state: State) -> dict:
    """Resume el texto usando el LLM.

    Args:
        state (State): El estado actual con el texto a resumir.

    Returns:
        dict: Un diccionario con la clave "summary".
    """
    # Crea una cadena conectando la plantilla de prompt al modelo
    chain = SUMMARIZATION_TEMPLATE | llm
    # Ejecuta la cadena con el texto de entrada
    response = chain.invoke({"text": state["text"]})
    # Retorna un diccionario con el resumen
    return {"summary": response.content}

# --- Creación del Flujo de Trabajo (Workflow) --- #
def create_workflow():
    """Crea y configura el grafo de LangGraph.

    Returns:
        CompiledGraph: El grafo compilado listo para ser ejecutado.
    """
    workflow = StateGraph(State)
    # Agrega nodos al grafo
    workflow.add_node("classification_node", classification_node)
    workflow.add_node("entity_extraction_node", entity_extraction_node) 
    workflow.add_node("summarization_node", summarize_node) 
    # Agrega bordes al grafo
    workflow.set_entry_point("classification_node") 
    workflow.add_edge("classification_node", "entity_extraction_node")
    workflow.add_edge("entity_extraction_node", "summarization_node")
    workflow.add_edge("summarization_node", END)
    # Compila el grafo
    app = workflow.compile()
    return app

# --- Función Principal --- #
def main():
    """Función principal para ejecutar el agente de análisis de texto."""
    # Crea el flujo de trabajo
    app = create_workflow()

    # Define un texto de ejemplo
    sample_text = ("""
    El MCP (Protocolo de contexto de modelo) de Anthropic es una herramienta de código abierto
    que permite que sus aplicaciones interactúen sin esfuerzo con las API en varios sistemas.
    """)

    # Crea el estado inicial
    state_input = {"text": sample_text}

    print("--- Ejecutando Agente --- ")
    print(f"Texto de entrada:\n{sample_text}")

    # Ejecuta el flujo de trabajo
    result = app.invoke(state_input)

    # Imprime los resultados
    print("\n--- Resultados del Análisis ---")
    print(f"Clasificación: {result.get('classification', 'N/A')}")
    print(f"Entidades: {result.get('entities', [])}")
    print(f"Resumen: {result.get('summary', 'N/A')}")

# --- Punto de Entrada del Script --- #
if __name__ == "__main__":
    main()
