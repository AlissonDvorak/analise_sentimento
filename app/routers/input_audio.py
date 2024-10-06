from typing import Optional
from fastapi import FastAPI, File, UploadFile, APIRouter
from io import BytesIO
from services.service import converter_para_wav, predict_sentiment, prever_sentimento_audio, transcribe




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
        texto_transcrito =  prever_sentimento_audio(audio_wav)  
    except Exception as e:
        return {"error": "Erro ao transcrever o áudio", "details": str(e)}
    
    return {
        "filename": file.filename,
        "description": description,
        "transcricao": texto_transcrito
    }
    


@input_audio.post("/upload-audio-whisper/")
async def transcribe_audio(file: UploadFile = File(...), description: str = None):
    # Lê o conteúdo do arquivo
    audio_bytes = await file.read()

    # Usa a função de transcrição
    transcription = transcribe(audio_bytes)
    
    # Verifica se a transcrição foi realizada com sucesso
    if 'text' not in transcription:
        return {"error": "Transcrição falhou", "details": transcription}

    try:
        # Passa a transcrição como uma lista
        resultado_sentimento = predict_sentiment([transcription['text']])  # Acessando a chave 'text'
    except Exception as e:
        return {"error": "Erro ao prever o sentimento", "details": str(e)}
    
    # Retorno final com a transcrição e sentimento
    return {
        "filename": file.filename,
        "description": description,
        "transcricao": resultado_sentimento,

    }