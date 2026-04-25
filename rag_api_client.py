"""
Rag API Client
"""
import requests


def send_to_rag_api(payload: dict, rag_url: str, timeout: int = 90) -> dict:
    clean_url = str(rag_url).strip().strip('"').strip("'")

    try:
        response = requests.post(
            url=clean_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}

        return {
            "success": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": clean_url,
            "response": body,
            "error": None if 200 <= response.status_code < 300 else response.text,
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status_code": None,
            "url": clean_url,
            "response": {},
            "error": f"No se pudo conectar con la API RAG: {clean_url}",
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": None,
            "url": clean_url,
            "response": {},
            "error": f"Timeout después de {timeout}s",
        }

    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "url": clean_url,
            "response": {},
            "error": str(e),
        }
