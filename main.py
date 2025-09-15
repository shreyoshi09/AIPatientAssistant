from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from textanalysis import HealthTextAnalyzer
from summarizer import OpenAISummarizer
from formrecognizerclient import FormRecognizerClient
import traceback
import json
import os
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(
    title="healthsumai-backend",
    description="API Docs",
    version="1.0.0"  
)

#Language Credentials
LANGUAGE_ENDPOINT =  os.getenv("LANGUAGE_ENDPOINT")
LANGUAGE_KEY = os.getenv("LANGUAGE_KEY")

# Azure Form Recognizer credentials
FORM_RECOGNIZER_ENDPOINT = os.getenv("FORM_RECOGNIZER_ENDPOINT")
FORM_RECOGNIZER_KEY = os.getenv("FORM_RECOGNIZER_KEY")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

form_recognizer = FormRecognizerClient(FORM_RECOGNIZER_ENDPOINT,FORM_RECOGNIZER_KEY )

@app.post("/process-note",  tags=["Summary"])
async def process_note(
    file: UploadFile = File(...),
    patient_id: str = Form(...)
):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="Provide a file")

        # 1️⃣ Extract text
      
        extracted_text = form_recognizer.extract_text_from_pdf(file)
            
        # 2️⃣ Create payload dynamically for HealthTextAnalyzer
        payload = {
            "analysisInput": {
                "documents": [
                    {"id": "1", "language": "en", "text": extracted_text}
                ]
            },
            "tasks": [
                {"taskId": "analyze1", "kind": "Healthcare", "parameters": {"fhirVersion": "4.0.1"}}
            ]
        }


        analyzer = HealthTextAnalyzer(LANGUAGE_ENDPOINT, LANGUAGE_KEY)
        # Get FHIR bundle directly
        fhir_bundle = analyzer.get_fhir_bundle(payload)

        compact = analyzer.preprocess_fhir_bundle(fhir_bundle)

        summarizer = OpenAISummarizer(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY,DEPLOYMENT_NAME)
        summary = summarizer.summarize_with_alerts(compact)

        return {
            "patient_id": patient_id,
            "summary": json.dumps(summary)       
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
