from typing import List
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment

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

tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

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
    
    
