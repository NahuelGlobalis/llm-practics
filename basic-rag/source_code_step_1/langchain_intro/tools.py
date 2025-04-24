"""
Este módulo contiene herramientas personalizadas para el agente de LangChain.
Las herramientas en LangChain son componentes que permiten a los agentes interactuar
con sistemas externos o realizar acciones específicas.
"""

import random
import time
from typing import Union  # Importamos Union para mejorar las type hints


def get_current_wait_time(hospital: str) -> Union[int, str]:
    """
    Genera tiempos de espera simulados para hospitales.
    
    Esta función representa una herramienta personalizada para LangChain.
    En LangChain, las herramientas (Tools) son funciones que un agente puede utilizar
    para interactuar con sistemas externos o realizar cálculos.
    
    Args:
        hospital (str): Identificador del hospital (A, B, C o D)
        
    Returns:
        Union[int, str]: Tiempo de espera en minutos o mensaje de error
        
    Ejemplo de uso en un agente LangChain:
        Tool(
            name="Waits",
            func=get_current_wait_time,
            description="Obtiene el tiempo de espera actual en un hospital"
        )
    """
    # Validamos que el hospital exista en nuestro sistema simulado
    if hospital not in ["A", "B", "C", "D"]:
        return f"Hospital {hospital} does not exist"

    # Simulamos retardo de API - esto representa una llamada a un servicio externo
    # En aplicaciones reales, aquí conectaríamos con APIs externas, bases de datos, etc.
    time.sleep(1)

    # Devolvemos un tiempo de espera aleatorio
    return random.randint(0, 10000)
