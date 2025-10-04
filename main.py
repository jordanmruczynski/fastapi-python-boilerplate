import os, json
import asyncio
import tempfile
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
import httpx
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from realitydefender import RealityDefender

# --- nowo dodane importy dla /api/verify-link ---
from supadata import Supadata, SupadataError
from urllib.parse import urlparse, parse_qs

# -----------------------------------------------

app = FastAPI(
    title="Vercel + FastAPI",
    description="Vercel + FastAPI",
    version="1.0.0",
)


@app.get("/api/data")
def get_sample_data():
    return {
        "data": [
            {"id": 1, "name": "Sample Item 1", "value": 100},
            {"id": 2, "name": "Sample Item 2", "value": 200},
            {"id": 3, "name": "Sample Item 3", "value": 300}
        ],
        "total": 3,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.get("/api/items/{item_id}")
def get_item(item_id: int):
    return {
        "item": {
            "id": item_id,
            "name": "Sample Item " + str(item_id),
            "value": item_id * 100
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.post("/api/verify-image")
async def verify_image_with_reality_defender(
        file: UploadFile = File(..., description="Obraz do weryfikacji (image/*)")
):
    """
    Wysyła obraz do Reality Defender i zwraca surowy wynik (status, score, modele).
    Bez progu i bez wyliczania is_ai.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415,
                            detail=f"Nieobsługiwany typ pliku: {file.content_type or 'brak'} (wymagane image/*)")

    api_key = os.getenv("REALITY_DEFENDER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Brak konfiguracji: ustaw REALITY_DEFENDER_API_KEY")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[-1],
                                         dir="/tmp") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nie udało się zapisać pliku: {e}")

    try:
        rd = RealityDefender(api_key=api_key)

        upload_resp: Dict[str, Any] = await rd.upload(file_path=tmp_path)
        request_id = upload_resp.get("request_id")
        if not request_id:
            raise HTTPException(status_code=502, detail="Reality Defender nie zwrócił request_id")

        # SDK polluje aż do zakończenia
        result: Dict[str, Any] = await asyncio.wait_for(rd.get_result(request_id), timeout=90.0)

        return {
            "request_id": request_id,
            "status": result.get("status"),
            "score": result.get("score"),
            "models": [
                {
                    "name": m.get("name"),
                    "status": m.get("status"),
                    "score": m.get("score"),
                    "metadata": m.get("metadata"),
                } for m in result.get("models", [])
            ],
            # Opcjonalnie możesz też dodać cały surowy wynik:
            # "raw": result
        }

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Przekroczono czas oczekiwania na wynik Reality Defender")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Błąd Reality Defender: {e}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/verify-text")
async def verify_text_disinfo(text: str = Body(..., embed=True, description="Tekst do analizy")):
    """
    Analiza tekstu pod kątem dezinformacji/rosyjskiej propagandy przez xAI Grok.
    Przyjmuje WYŁĄCZNIE pole 'text' w JSON. Model jest stały: grok-4-fast-reasoning.
    Zwraca JSON wg zadanego schematu (decision, summary, ai_explanatation, sources).
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Brak konfiguracji: ustaw XAI_API_KEY")

    system_prompt = (
        "Jesteś dziennikarzem zajmującym się walką z dezinformacją i rosyjską propagandą. "
        "Przeanalizuj poniższy tekst i oceń, czy jest dezinformacją/rosyjską propagandą czy nie. "
        "Skup się na informacjach z internetu, przeanalizuj źródła popularnych stron dziennikarskich (np. Onet) "
        "oraz rządowe strony. Zwróć wynik wyłącznie jako poprawny JSON wg zadanego schematu."
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "disinfo_check",
            "schema": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "description": "tak/nie czy propaganda/dezinformacja",
                        "enum": ["tak", "nie"]
                    },
                    "summary": {
                        "type": "string",
                        "description": "krótkie podsumowanie, 1-2 zdania"
                    },
                    "ai_explanatation": {  # celowo utrzymana literówka wg Twojej specyfikacji
                        "type": "string",
                        "description": "około 5 zdań z wyjaśnieniami decyzji"
                    },
                    "sources": {
                        "type": "array",
                        "description": "max 10 najważniejszych źródeł/URL-i",
                        "items": {"type": "string"},
                        "maxItems": 10
                    }
                },
                "required": ["decision", "summary", "ai_explanatation", "sources"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-4-fast-reasoning",  # stały model
                    "messages": messages,
                    "response_format": response_format,
                    "search_parameters": {"mode": "on"},
                    "temperature": 0.2,
                },
            )

        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"xAI error: {resp.text}")

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise HTTPException(status_code=502, detail="Pusta odpowiedź modelu xAI")

        # oczekujemy poprawnego JSON-a
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Model zwrócił niepoprawny JSON")

        # twarde sprawdzenie schematu
        for key in ["decision", "summary", "ai_explanatation", "sources"]:
            if key not in parsed:
                raise HTTPException(status_code=502, detail=f"Brak wymaganego pola: {key}")
        if parsed.get("decision") not in ("tak", "nie"):
            raise HTTPException(status_code=502, detail="Pole 'decision' musi być 'tak' lub 'nie'")
        if not isinstance(parsed.get("sources"), list) or len(parsed["sources"]) > 10:
            raise HTTPException(status_code=502, detail="Pole 'sources' musi być listą (max 10)")

        return parsed

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Błąd podczas analizy: {e}")


# === NOWY ENDPOINT: weryfikacja linku (web/YouTube przez Supadata) ===
@app.post("/api/verify-link")
async def verify_link(url: str = Body(..., embed=True, description="URL strony www lub YouTube")):
    """
    1) Dla YouTube: pobiera transkrypcję przez Supadata.youtube.transcript(..., text=True)
    2) Dla zwykłej strony: pobiera treść przez Supadata.web.scrape(url)
    3) Przekazuje content do Grok (grok-4-fast-reasoning) z tym samym promptem co /api/verify-text
    4) Zwraca JSON: {decision, summary, ai_explanatation, sources}
    """
    # --- klucze ---
    supa_key = os.getenv("SUPADATA_API_KEY")
    if not supa_key:
        raise HTTPException(status_code=500, detail="Brak konfiguracji: ustaw SUPADATA_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")
    if not xai_key:
        raise HTTPException(status_code=500, detail="Brak konfiguracji: ustaw XAI_API_KEY")

    # --- prosta detekcja YouTube + wyciągnięcie video_id ---
    def _extract_youtube_video_id(u: str):
        try:
            parsed = urlparse(u)
            host = (parsed.netloc or "").lower()
            if host in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
                if parsed.path.startswith("/watch"):
                    q = parse_qs(parsed.query or "")
                    return (q.get("v", [None])[0]) or None
                if parsed.path.startswith("/shorts/"):
                    parts = parsed.path.split("/")
                    return parts[2] if len(parts) > 2 else None
            if host == "youtu.be":
                return parsed.path.lstrip("/").split("/")[0] or None
            return None
        except Exception:
            return None

    # --- pobranie contentu przez Supadata ---
    supadata = Supadata(api_key=supa_key)
    try:
        video_id = _extract_youtube_video_id(url)
        if video_id:
            yt_resp = supadata.youtube.transcript(video_id=video_id, text=True)
            content = yt_resp.content or ""
        else:
            web_resp = supadata.web.scrape(url)
            content = web_resp.content or ""

        if not content.strip():
            raise HTTPException(status_code=422, detail="Nie udało się wydobyć treści z podanego URL.")
    except SupadataError as e:
        code = getattr(e, "status", None) or getattr(e, "status_code", None)
        # Mapowanie komunikatów zgodnie ze specyfikacją:
        messages = {
            400: "Invalid Request: żądanie jest nieprawidłowe lub błędnie sformatowane",
            401: "Unauthorized: sprawdź klucz API",
            402: "Upgrade Required: funkcja niedostępna w Twoim planie",
            404: "Not Found: nie znaleziono zasobu",
            429: "Limit Exceeded: przekroczono dozwolony limit",
            206: "Transcript Unavailable: brak transkrypcji dla tego filmu",
            500: "Internal Error: błąd wewnętrzny",
        }
        msg = messages.get(code, f"Supadata error: {str(e)}")
        raise HTTPException(status_code=int(code or 502), detail=msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Błąd Supadata: {e}")

    # --- analiza Grok tak jak w /api/verify-text ---
    system_prompt = (
        "Jesteś dziennikarzem zajmującym się walką z dezinformacją i rosyjską propagandą. "
        "Przeanalizuj poniższy tekst i oceń, czy jest dezinformacją/rosyjską propagandą czy nie. "
        "Skup się na informacjach z internetu, przeanalizuj źródła popularnych stron dziennikarskich (np. Onet) "
        "oraz rządowe strony. Zwróć wynik wyłącznie jako poprawny JSON wg zadanego schematu."
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "disinfo_check",
            "schema": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["tak", "nie"]},
                    "summary": {"type": "string"},
                    "ai_explanatation": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}, "maxItems": 10}
                },
                "required": ["decision", "summary", "ai_explanatation", "sources"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {xai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-4-fast-reasoning",
                    "messages": messages,
                    "response_format": response_format,
                    "search_parameters": {"mode": "on"},
                    "temperature": 0.2,
                },
            )
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"xAI error: {resp.text}")

        data = resp.json()
        content_json = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content_json:
            raise HTTPException(status_code=502, detail="Pusta odpowiedź modelu xAI")

        parsed = json.loads(content_json)

        for k in ["decision", "summary", "ai_explanatation", "sources"]:
            if k not in parsed:
                raise HTTPException(status_code=502, detail=f"Brak wymaganego pola: {k}")
        if parsed.get("decision") not in ("tak", "nie"):
            raise HTTPException(status_code=502, detail="Pole 'decision' musi być 'tak' lub 'nie'")
        if not isinstance(parsed.get("sources"), list) or len(parsed["sources"]) > 10:
            raise HTTPException(status_code=502, detail="Pole 'sources' musi być listą (max 10)")

        return parsed

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Błąd podczas analizy: {e}")


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vercel + FastAPI</title>
        <link rel="icon" type="image/x-icon" href="/favicon.ico">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                background-color: #000000;
                color: #ffffff;
                line-height: 1.6;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            header {
                border-bottom: 1px solid #333333;
                padding: 0;
            }

            nav {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                padding: 1rem 2rem;
                gap: 2rem;
            }

            .logo {
                font-size: 1.25rem;
                font-weight: 600;
                color: #ffffff;
                text-decoration: none;
            }

            .nav-links {
                display: flex;
                gap: 1.5rem;
                margin-left: auto;
            }

            .nav-links a {
                text-decoration: none;
                color: #888888;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                transition: all 0.2s ease;
                font-size: 0.875rem;
                font-weight: 500;
            }

            .nav-links a:hover {
                color: #ffffff;
                background-color: #111111;
            }

            main {
                flex: 1;
                max-width: 1200px;
                margin: 0 auto;
                padding: 4rem 2rem;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
            }

            .hero {
                margin-bottom: 3rem;
            }

            .hero-code {
                margin-top: 2rem;
                width: 100%;
                max-width: 900px;
                display: grid,
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            }

            .hero-code pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: left;
                grid-column: 1 / -1;
            }

            h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                background: linear-gradient(to right, #ffffff, #888888);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .subtitle {
                font-size: 1.25rem;
                color: #888888;
                margin-bottom: 2rem;
                max-width: 600px;
            }

            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                width: 100%;
                max-width: 900px;
            }

            .card {
                background-color: #111111;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 1.5rem;
                transition: all 0.2s ease;
                text-align: left;
            }

            .card:hover {
                border-color: #555555;
                transform: translateY(-2px);
            }

            .card h3 {
                font-size: 1.125rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: #ffffff;
            }

            .card p {
                color: #888888;
                font-size: 0.875rem;
                margin-bottom: 1rem;
            }

            .card a {
                display: inline-flex;
                align-items: center;
                color: #ffffff;
                text-decoration: none;
                font-size: 0.875rem;
                font-weight: 500;
                padding: 0.5rem 1rem;
                background-color: #222222;
                border-radius: 6px;
                border: 1px solid #333333;
                transition: all 0.2s ease;
            }

            .card a:hover {
                background-color: #333333;
                border-color: #555555;
            }

            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background-color: #0070f3;
                color: #ffffff;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-bottom: 2rem;
            }

            .status-dot {
                width: 6px;
                height: 6px;
                background-color: #00ff88;
                border-radius: 50%;
            }

            pre {
                background-color: #0a0a0a;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 1rem;
                overflow-x: auto;
                margin: 0;
            }

            code {
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.85rem;
                line-height: 1.5;
                color: #ffffff;
            }

            @media (max-width: 768px) {
                nav {
                    padding: 1rem;
                    flex-direction: column;
                    gap: 1rem;
                }

                .nav-links {
                    margin-left: 0;
                }

                main {
                    padding: 2rem 1rem;
                }

                h1 {
                    font-size: 2rem;
                }

                .hero-code {
                    grid-template-columns: 1fr;
                }

                .cards {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <header>
            <nav>
                <a href="/" class="logo">Vercel + FastAPI</a>
                <div class="nav-links">
                    <a href="/docs">API Docs</a>
                    <a href="/api/data">API</a>
                </div>
            </nav>
        </header>
        <main>
            <div class="hero">
                <h1>Vercel + FastAPI</h1>
                <div class="hero-code">
                    <pre><code><span class="keyword">from</span> <span class="module">fastapi</span> <span class="keyword">import</span> <span class="class">FastAPI</span>

<span class="variable">app</span> = <span class="class">FastAPI</span>()

<span class="decorator">@app.get</span>(<span class="string">"/"</span>)
<span class="keyword">def</span> <span class="function">read_root</span>():
    <span class="keyword">return</span> {<span class="string">"Python"</span>: <span class="string">"on Vercel"</span>}</code></pre>
                </div>
            </div>

            <div class="cards">
                <div class="card">
                    <h3>Interactive API Docs</h3>
                    <p>Explore this API's endpoints with the interactive Swagger UI. Test requests and view response schemas in real-time.</p>
                    <a href="/docs">Open Swagger UI →</a>
                </div>

                <div class="card">
                    <h3>Sample Data</h3>
                    <p>Access sample JSON data through our REST API. Perfect for testing and development purposes.</p>
                    <a href="/api/data">Get Data →</a>
                </div>

            </div>
        </main>
    </body>
    </html>
    """
