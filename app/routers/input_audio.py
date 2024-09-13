from typing import Optional
from fastapi import FastAPI, File, UploadFile, APIRouter
from io import BytesIO
from services.service import converter_para_wav, prever_sentimento_audio



input_audio = APIRouter(
    prefix="/audio",
    tags=["audio"],
)
    
    
@input_audio.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...), description: str = None):
    # Lê o conteúdo do arquivo de áudio
    audio_data = BytesIO(await file.read())
    
    # Converte o áudio para .wav
    audio_wav = converter_para_wav(audio_data, file.content_type.split('/')[1]) 
    
    # Transcreve o áudio para texto
    try:
        texto_transcrito =  prever_sentimento_audio(audio_wav)  # Tornar essa função assíncrona
    except Exception as e:
        return {"error": "Erro ao transcrever o áudio", "details": str(e)}
    
    return {
        "filename": file.filename,
        "description": description,
        "transcricao": texto_transcrito
    }