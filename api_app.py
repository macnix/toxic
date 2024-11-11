from fastapi import FastAPI, HTTPException
from easynmt import EasyNMT
import fasttext
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Initialize FastAPI app
app = FastAPI()
# Declare models globally for app-level scope
translation_model = None
dmodel = None
bert_model = None
tokenizer = None

# Load models on startup
@app.on_event("startup")
async def load_models():
    global translation_model, dmodel, bert_model, tokenizer
    import nltk
    print(nltk.data.find('tokenizers/punkt'))
    # Load translation model
    translation_model = EasyNMT('m2m_100_418M')
    dmodel = fasttext.load_model('lid.176.bin')
    # Load TinyBERT model and tokenizer for toxicity classification
    bert_model = BertForSequenceClassification.from_pretrained('./toxic_job_description_model')
    tokenizer = BertTokenizer.from_pretrained('./toxic_job_description_tokenizer')
    print("Models loaded")

# Request and response models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float
 #   translated_text: str  # Include translated text in the response

# Translation function
def translate_to_english(text: str, source_lang: str) -> str:
    return translation_model.translate(text, source_lang=source_lang, target_lang="en")

# Prediction function
def predict_toxicity(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    score = torch.softmax(logits, dim=1)[0][1].item()
    label = "Toxic" if score >= 0.5 else "Non-toxic"
    return {"label": label, "score": score}

# Define a POST endpoint for predictions
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Step 1: Detect language
        text = str(request.text)
        
        label, confidence = dmodel.predict([text], k=1)
        iso_language_code = label[0][0].replace('__label__', '')
        #source_lang = iso_language_code if iso_language_code in ['uk', 'ru'] else 'uk'

        # Step 2: If language is not English, translate
        if iso_language_code != 'en':
            translated_text = translate_to_english(text, iso_language_code )
        else:
            translated_text = text  # No translation needed if already in English

        # Step 3: Predict toxicity on the English text
        result = predict_toxicity(translated_text)
        return PredictionResponse(label=result["label"], score=result["score"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
