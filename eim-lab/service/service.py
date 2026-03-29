from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn
import json
import logging
import asyncio

# Configurar el logging en nivel INFO para ver toda la salida por consola
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("OllamaProxy")

app = FastAPI(
    title="Ollama Proxy Service",
    description="Servicio que actúa exclusivamente como proxy hacia una instancia local de Ollama (puerto 11434)"
)

OLLAMA_URL = "http://localhost:11434"

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy_to_ollama(request: Request, path: str):
    url = f"{OLLAMA_URL}/{path}"
    
    logger.info("=====================================================")
    logger.info("--- NUEVA PETICIÓN RECIBIDA ---")
    logger.info(f"Método original: {request.method} | Path: {request.url.path}")
    
    headers = dict(request.headers)
    headers.pop("host", None) # No se debe pasar el host original al proxy
    
    body = await request.body()
    # Intentamos loggear el cuerpo si es JSON
    try:
        body_json = json.loads(body.decode("utf-8"))
        logger.info(f"Body (JSON enviado por el cliente): \n{json.dumps(body_json, indent=2)}")
    except:
        logger.info(f"Body (Raw): {body}")
        
    logger.info(f"URL de Destino Proxy -> Ollama: {url}")
    
    # Desactivar timeout porque Ollama puede tardar minutos en cargar un modelo de 32B en memoria
    client = httpx.AsyncClient(timeout=None)
    
    req = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
        params=request.query_params,
    )
    
    try:
        logger.info("-> Enviando petición a Ollama y esperando la primera respuesta (esto puede tardar si se está cargando el modelo en la GPU)...")
        response = await client.send(req, stream=True)
        logger.info(f"<- Respuesta inicial de Ollama recibida. Status Code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error al conectar con la instancia local de Ollama: {str(e)}")
        raise e
    
    async def stream_generator():
        logger.info("-> Empezando a hacer stream (reenviar a trozos) la respuesta hacia el cliente...")
        try:
            async for chunk in response.aiter_raw():
                # Si el chunk es JSON/texto, lo imprimimos (sólo los primeros caracteres para no saturar la pantalla)
                if chunk:
                    try:
                        chunk_text = chunk.decode('utf-8').strip()
                        if chunk_text:
                            # Imprime las partes de texto de la salida del modelo
                            logger.info(f"[OLLAMA_OUTPUT_CHUNK]: {chunk_text[:150]} ...")
                    except:
                        pass
                yield chunk
            logger.info("--- FIN DEL STREAM HACIA EL CLIENTE ---")
            logger.info("=====================================================")
        except asyncio.CancelledError:
            # El error que viste antes (CancelledError) ocurre cuando EL CLIENTE QUE HIZO LA PETICIÓN (curl, postman o la web) cierra la conexión antes de que acabe el stream.
            logger.warning("!!! EL CLIENTE (quien hizo la petición HTTP original) SE DESCONECTÓ O CANCELÓ LA PETICIÓN ANTES DE QUE OLLAMA TERMINASE !!!")
        except Exception as e:
            logger.error(f"!!! Error leyendo el stream de Ollama: {str(e)} !!!")
        finally:
            await client.aclose()
        
    # Limpiamos ciertas cabeceras que dan problema si devolvemos un StreamingResponse
    resp_headers = dict(response.headers)
    resp_headers.pop("content-length", None)
    resp_headers.pop("content-encoding", None)
    
    return StreamingResponse(
        stream_generator(), 
        status_code=response.status_code,
        headers=resp_headers
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
