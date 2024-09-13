from models.models import TextRequest
from services.service import predict_sentiment
from fastapi import APIRouter

input_texto = APIRouter(
    prefix="/text",
    tags=["text"],
)
    
    
@input_texto.post("/predict_sentiment")
def prever_sentimento(request: TextRequest):
    texts = request.texts

    # Verifique e ajuste os textos, se necessário
    texts = [text.replace('""', '"') for text in texts]  # Exemplo de substituição de aspas duplicadas, se necessário

    predictions = predict_sentiment(texts)
    return {"predictions": predictions}
    
    