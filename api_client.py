# =============================================================================
# api_client.py — Envío del payload al endpoint del equipo
# =============================================================================

import json
from typing import Optional

import requests


# ─── Envío al endpoint ────────────────────────────────────────────────────────
def send_to_endpoint(
    endpoint_payload: dict,
    endpoint_url:     str,
    timeout:          int = 30,
    headers:          Optional[dict] = None,
) -> dict:
    """
    Envía el endpoint_payload (generado por build_endpoint_payload) al
    endpoint del equipo via POST con Content-Type application/json.

    Args:
        endpoint_payload: dict retornado por build_endpoint_payload()
        endpoint_url:     URL completa del endpoint, ej: "https://api.equipo.com/inspect"
        timeout:          segundos máximos de espera (default 30)
        headers:          headers adicionales, ej: {"Authorization": "Bearer TOKEN"}

    Returns:
        dict con las claves:
            "success"     bool   → True si HTTP 2xx
            "status_code" int    → código HTTP recibido
            "response"    dict   → body de la respuesta (parseado si es JSON)
            "error"       str    → mensaje de error si success es False

    Raises:
        No lanza excepciones; todos los errores se capturan y se retornan
        en el campo "error" del dict de respuesta.
    """
    _headers = {"Content-Type": "application/json"}
    if headers:
        _headers.update(headers)

    try:
        response = requests.post(
            url=endpoint_url,
            data=json.dumps(endpoint_payload, ensure_ascii=False),
            headers=_headers,
            timeout=timeout,
        )

        # Intentar parsear la respuesta como JSON
        try:
            response_body = response.json()
        except Exception:
            response_body = {"raw": response.text}

        success = response.status_code in range(200, 300)

        return {
            "success":     success,
            "status_code": response.status_code,
            "response":    response_body,
            "error":       None if success else f"HTTP {response.status_code}: {response.text[:300]}",
        }

    except requests.exceptions.ConnectionError:
        return {
            "success":     False,
            "status_code": None,
            "response":    {},
            "error":       f"No se pudo conectar con el endpoint: {endpoint_url}",
        }
    except requests.exceptions.Timeout:
        return {
            "success":     False,
            "status_code": None,
            "response":    {},
            "error":       f"Timeout después de {timeout}s. El endpoint no respondió.",
        }
    except Exception as e:
        return {
            "success":     False,
            "status_code": None,
            "response":    {},
            "error":       f"Error inesperado: {str(e)}",
        }