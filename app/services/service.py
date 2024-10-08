from typing import List
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from pyannote.audio import Pipeline
import concurrent.futures
import os
import warnings
warnings.filterwarnings("ignore")


# Definir o mapeamento de sentimentos para estrelas
def get_sentiment_label(stars: int) -> str:
    if stars in [1, 2]:
        return "negativo"
    elif stars == 3:
        return "neutro"
    elif stars in [4, 5]:
        return "positivo"
    else:
        return "Desconhecido"

device = 0 if torch.cuda.is_available() else -1




tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
pipeline_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="TOKEN_HERE")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", return_timestamps=True, device=device)


def preprocess_text(texts: List[str], max_length: int = 128):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return inputs

def predict_sentiment(texts: List[str]):
    inputs = preprocess_text(texts)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    results = []
    for text, prob in zip(texts, probabilities):
        sentiment_idx = torch.argmax(prob).item()  
        sentiment = get_sentiment_label(sentiment_idx + 1)  
        confidence = prob[sentiment_idx].item()
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4)
        })
    return results

def converter_para_wav(audio_data: BytesIO, input_format: str) -> BytesIO:
    if input_format == 'wav':
        return audio_data
    audio = AudioSegment.from_file(audio_data, format=input_format)
    wav_io = BytesIO()
    audio.export(wav_io, format='wav')
    wav_io.seek(0)
    return wav_io

def prever_sentimento_audio(audio_data: BytesIO):
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language='pt-BR')  
        text = predict_sentiment([text])
        return text
    except sr.UnknownValueError:
        return "Não consegui entender o áudio"
    except sr.RequestError:
        return "Erro ao se conectar ao serviço de reconhecimento"
    


def transcribe(audio_bytes):
    # Usa o pipeline para transcrever o áudio
    return transcriber(audio_bytes)




def process_audio(audio_data: BytesIO, input_format: str):
  
    # Salva o áudio em um arquivo temporário
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_data.read())
    
    # Função para realizar a diarização
    def diarize():
        diarization = pipeline_diarization(temp_audio_path)
        return diarization

    # Função para realizar a transcrição
    def transcribe():
        transcription = transcriber(temp_audio_path)
        return transcription

    # Executa diarização e transcrição em paralelo com limite de 2 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        diarization_future = executor.submit(diarize)
        transcription_future = executor.submit(transcribe)

        # Aguarda os resultados com intervalo de descanso
        diarization = diarization_future.result()
        transcription = transcription_future.result()

    # Cria um dicionário para armazenar as falas dos locutores em ordem
    speaker_lines = {speaker: [] for _, _, speaker in diarization.itertracks(yield_label=True)}

    # Itera sobre os segmentos da transcrição
    for segment in transcription['chunks']:
        segment_start, segment_end = segment['timestamp']
        segment_text = segment['text'].strip()

        # Verifica a qual locutor o segmento pertence
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turn_start = turn.start  # Renomeado para evitar confusão com 'start_time'
            turn_end = turn.end

            # Verifica se o segmento de transcrição está dentro do intervalo do locutor
            if segment_end > turn_start and segment_start < turn_end:
                # Adiciona a frase completa com o locutor
                speaker_lines[speaker].append(segment_text)

    # Monta a saída na ordem dos trechos do Whisper
    output = []
    for segment in transcription['chunks']:
        segment_start, segment_end = segment['timestamp']
        segment_text = segment['text'].strip()
        
        # Verifica a qual locutor o segmento pertence
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turn_start = turn.start  # Renomeado para evitar confusão com 'start_time'
            turn_end = turn.end
            
            # Verifica se o segmento de transcrição está dentro do intervalo do locutor
            if segment_end > turn_start and segment_start < turn_end:
                output.append(f"Locutor {speaker}: {segment_text}")
                break  # Sai do loop após encontrar o locutor correspondente
    # Remove o arquivo temporário
    os.remove(temp_audio_path)

    return output

# import os


# def process_audio(audio_data: BytesIO, input_format: str):
#     start_time = time.time()  # Início da contagem de tempo
#     # Salva o áudio em um arquivo temporário
#     temp_audio_path = "temp_audio.wav"
#     with open(temp_audio_path, "wb") as f:
#         f.write(audio_data.read())
    
#     # Diariza o áudio
#     diarization_start_time = time.time()  # Medindo o tempo da diarização
#     diarization = pipeline_diarization(temp_audio_path)
#     diarization_elapsed = time.time() - diarization_start_time
#     print(f"Tempo de diarização: {diarization_elapsed:.2f} segundos")

#     # Transcreve todo o áudio com timestamps
#     transcription_start_time = time.time()  # Medindo o tempo da transcrição
#     transcription = transcriber(temp_audio_path)
#     transcription_elapsed = time.time() - transcription_start_time
#     print(f"Tempo de transcrição: {transcription_elapsed:.2f} segundos")

#     # Cria um dicionário para armazenar as falas dos locutores em ordem
#     speaker_lines = {speaker: [] for _, _, speaker in diarization.itertracks(yield_label=True)}

#     # Itera sobre os segmentos da transcrição
#     for segment in transcription['chunks']:
#         segment_start, segment_end = segment['timestamp']
#         segment_text = segment['text'].strip()

#         # Verifica a qual locutor o segmento pertence
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             turn_start = turn.start  # Renomeado para evitar confusão
#             turn_end = turn.end

#             # Verifica se o segmento de transcrição está dentro do intervalo do locutor
#             if segment_end > turn_start and segment_start < turn_end:
#                 # Adiciona a frase completa com o locutor
#                 speaker_lines[speaker].append(segment_text)

#     # Monta a saída na ordem dos trechos do Whisper
#     output = []
#     for segment in transcription['chunks']:
#         segment_start, segment_end = segment['timestamp']
#         segment_text = segment['text'].strip()
        
#         # Verifica a qual locutor o segmento pertence
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             turn_start = turn.start  # Renomeado para evitar confusão
#             turn_end = turn.end
            
#             # Verifica se o segmento de transcrição está dentro do intervalo do locutor
#             if segment_end > turn_start and segment_start < turn_end:
#                 output.append(f"Locutor {speaker}: {segment_text}")
#                 break  # Sai do loop após encontrar o locutor correspondente

#     elapsed_time = time.time() - start_time  # Calcula o tempo total
#     print(f"Tempo total de processamento: {elapsed_time:.2f} segundos")

#     # Remove o arquivo temporário
#     os.remove(temp_audio_path)
    
#     return output



# # killer pc
# def process_audio(audio_data: BytesIO, input_format: str):
#     start_time = time.time()  # Início da contagem de tempo
    
#     # Salva o áudio em um arquivo temporário
#     temp_audio_path = "temp_audio.wav"
#     with open(temp_audio_path, "wb") as f:
#         f.write(audio_data.read())
    
#     # Função para realizar a diarização
#     def diarize():
#         print("Iniciando a diarização...")
#         diarization = pipeline_diarization(temp_audio_path)
#         print("Diarização concluída.")
#         return diarization

#     # Função para realizar a transcrição
#     def transcribe():
#         print("Iniciando a transcrição...")
#         transcription = transcriber(temp_audio_path)
#         print("Transcrição concluída.")
#         return transcription

#     # Executa diarização e transcrição em paralelo
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         diarization_future = executor.submit(diarize)
#         transcription_future = executor.submit(transcribe)

#         # Aguarda os resultados
#         diarization = diarization_future.result()
#         transcription = transcription_future.result()

#     # Verifica se a transcrição foi realizada na GPU
#     if torch.cuda.is_available():
#         print("A transcrição foi realizada na GPU.")
#     else:
#         print("A transcrição foi realizada na CPU.")

#     # Cria um dicionário para armazenar as falas dos locutores em ordem
#     speaker_lines = {speaker: [] for _, _, speaker in diarization.itertracks(yield_label=True)}

#     # Itera sobre os segmentos da transcrição
#     for segment in transcription['chunks']:
#         segment_start, segment_end = segment['timestamp']
#         segment_text = segment['text'].strip()

#         # Verifica a qual locutor o segmento pertence
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             turn_start = turn.start  # Renomeado para evitar confusão com 'start_time'
#             turn_end = turn.end

#             # Verifica se o segmento de transcrição está dentro do intervalo do locutor
#             if segment_end > turn_start and segment_start < turn_end:
#                 # Adiciona a frase completa com o locutor
#                 speaker_lines[speaker].append(segment_text)

#     # Monta a saída na ordem dos trechos do Whisper
#     output = []
#     for segment in transcription['chunks']:
#         segment_start, segment_end = segment['timestamp']
#         segment_text = segment['text'].strip()
        
#         # Verifica a qual locutor o segmento pertence
#         for turn, _, speaker in diarization.itertracks(yield_label=True):
#             turn_start = turn.start  # Renomeado para evitar confusão com 'start_time'
#             turn_end = turn.end
            
#             # Verifica se o segmento de transcrição está dentro do intervalo do locutor
#             if segment_end > turn_start and segment_start < turn_end:
#                 output.append(f"Locutor {speaker}: {segment_text}")
#                 break  # Sai do loop após encontrar o locutor correspondente
    
#     elapsed_time = time.time() - start_time  # Calcula o tempo total
#     print(f"Tempo de transcrição: {elapsed_time:.2f} segundos")

#     return output