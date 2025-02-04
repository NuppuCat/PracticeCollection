
import random
import string
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import pipeline

# Generate a random API token
API_TOKEN = "".join(random.choices(string.ascii_letters + string.digits, k=20))
print(f"🔹 Generated API Token: {API_TOKEN}")

# Initialize FastAPI
app = FastAPI()

# Load translation model (English → French as an example)
translator = pipeline("translation_en_to_fr")

# Authentication dependency
def verify_token(token: str):
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Token")

# Define request body schema
class TranslationRequest(BaseModel):
    text: str

# Define API endpoint for translation
@app.post("/translate/")
def translate_text(request: TranslationRequest, token: str = Depends(verify_token)):
    translated = translator(request.text)[0]["translation_text"]
    return {"original": request.text, "translated": translated}
