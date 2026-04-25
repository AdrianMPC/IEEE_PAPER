"""
Rag API Client
"""
import requests


def send_to_rag_api(payload: dict, rag_url: str, timeout: int = 60) -> dict:
    """
    Send a payload to the RAG API and return the response in a structured format.
    """
    try:
        response = requests.post(
            rag_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}

        return {
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "response": body,
            "error": None if response.ok else response.text[:500],
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": None,
            "response": {},
            "error": f"Timeout después de {timeout}s",
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status_code": None,
            "response": {},
            "error": f"No se pudo conectar con la API RAG: {rag_url}",
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": None,
            "response": {},
            "error": str(e),
        }
