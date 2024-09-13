from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from typing import List

# Carregar o tokenizer e o modelo pré-treinado
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

def preprocess_text(texts: List[str], max_length: int = 128):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return inputs

def get_sentiment_label(stars: int) -> str:
    if stars in [1, 2]:
        return "negativo"
    elif stars == 3:
        return "neutro"
    elif stars in [4, 5]:
        return "positivo"
    else:
        return "Desconhecido"

def predict_sentiment(texts: List[str], confidence_threshold: float = 0.5):
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
        if confidence >= confidence_threshold:
            results.append({
                "text": text,
                "sentiment": sentiment,
                "confidence": round(confidence, 4)
            })
        else:
            results.append({
                "text": text,
                "sentiment": "incerto",
                "confidence": round(confidence, 4)
            })
    return results

