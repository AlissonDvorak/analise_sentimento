from fastapi import File, UploadFile, APIRouter
from io import BytesIO

from fastapi.responses import JSONResponse
from services.service import converter_para_wav, predict_sentiment, prever_sentimento_audio, process_audio, transcribe




input_audio = APIRouter(
    prefix="/audio",
    tags=["audio"],
)
    
    
@input_audio.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...), description: str = None):
    # Lê o conteúdo do arquivo de áudio
    """
    Endpoint para transcrever um arquivo de áudio e prever o sentimento.

    - Recebe um arquivo de áudio e uma descrição opcional.
    - Transcreve o áudio com o modelo recognizer.
    - Preve o sentimento da transcrição com o modelo BERT.
    - Retorna o nome do arquivo, a descrição e a transcrição com o sentimento.

    :param file: Arquivo de áudio a ser transcrito.
    :type file: UploadFile
    :param description: Descrição opcional do arquivo.
    :type description: str
    :return: Dicionário com o nome do arquivo, a descrição e a transcrição com o sentimento.
    :rtype: dict
    """
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
    """
    Endpoint para transcrever um arquivo de áudio e prever o sentimento.

    - Recebe um arquivo de áudio e uma descrição opcional.
    - Transcreve o áudio com o modelo Whisper.
    - Preve o sentimento da transcrição com o modelo BERT.
    - Retorna o nome do arquivo, a descrição e a transcrição com o sentimento.

    :param file: Arquivo de áudio a ser transcrito.
    :type file: UploadFile
    :param description: Descrição opcional do arquivo.
    :type description: str
    :return: Dicionário com o nome do arquivo, a descrição e a transcrição com o sentimento.
    :rtype: dict
    """
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
    
    
@input_audio.post("/upload-call/")
async def upload_audio(file: UploadFile = File(...)):
    # Ler o conteúdo do arquivo de áudio
    """
    Endpoint para upload de um áudio de uma ligação.
    Retorna a transcrição do áudio em forma de lista de strings identificando os locutores.
    Cada string representa uma fala, e contém o texto transcrito por cada locutor.
    """
    audio_data = BytesIO(await file.read())
    
    # Processar o áudio
    transcript = process_audio(audio_data, file.content_type.split('/')[1])
    
    # Retornar a transcrição organizada
    return JSONResponse(content={"transcript": transcript})