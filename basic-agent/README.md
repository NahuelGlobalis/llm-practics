# Primer Agente de Análisis de Texto con LangGraph

Este proyecto implementa un **agente de análisis de texto** usando LangGraph y LangChain. Permite:

- **Clasificar** texto en categorías predefinidas.
- **Extraer** entidades nombradas (persona, organización, ubicación).
- **Generar** un resumen en una sola oración.

## Requisitos

- Python 3.10 o superior
- [Ollama](https://ollama.com/) con el modelo `gemma3:12b` disponible
- Conexión local a Ollama para invocar el modelo LLM

## Instalación

```bash
# Crear y activar entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate

# Instalar dependencias
pip install langgraph langchain langchain-ollama
``` 

> **Nota:** Si prefieres usar un archivo `requirements.txt`, créalo con:
> ```txt
> langgraph
> langchain
> langchain-ollama
> ```

## Uso

```bash
# Ejecutar el agente con un texto de ejemplo
python first-ai-agent.py
``` 

Se imprimirá en consola:

1. Texto de entrada
2. Clasificación
3. Lista de entidades extraídas
4. Resumen generado

## Estructura del Proyecto

```
first-ai-agent.py   # Lógica principal del agente
README.md          # Documentación del proyecto
``` 

## Personalización

- Cambia la constante `MODEL_NAME` en `first-ai-agent.py` para usar otro modelo.
- Ajusta `CATEGORIES` y `ENTITY_TYPES` según tus necesidades.
- Modifica la temperatura (`TEMPERATURE`) para controlar la creatividad del LLM.
