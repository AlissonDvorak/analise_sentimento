import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List

# Carregar o tokenizer e o modelo pré-treinado
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def preprocess_text(texts: List[str], max_length: int = 128):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        clean_up_tokenization_spaces=False
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
        print(sentiment_idx)
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


