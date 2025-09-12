from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from textanalysis import HealthTextAnalyzer
from summarizer import OpenAISummarizer
from formrecognizerclient import FormRecognizerClient
import traceback
import json


# Initialize FastAPI
app = FastAPI()


# Azure Cognitive Services (Language/FHIR) 
LANGUAGE_ENDPOINT =  ''
LANGUAGE_KEY = ''

# Azure Form Recognizer credentials
FORM_RECOGNIZER_ENDPOINT = ''
FORM_RECOGNIZER_KEY = ''

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = '' 
AZURE_OPENAI_KEY = ''
DEPLOYMENT_NAME = ''


form_recognizer = FormRecognizerClient(FORM_RECOGNIZER_ENDPOINT,FORM_RECOGNIZER_KEY )

@app.post("/process-note")
async def process_note(
    file: UploadFile = File(None),
    note: str = Form(None),
    patient_id: str = Form(None)
):
    try:
        if not note and not file:
            raise HTTPException(status_code=400, detail="Provide text or PDF")

        # 1️⃣ Extract text
        if file:
            extracted_text = form_recognizer.extract_text_from_pdf(file)
        else:
            extracted_text = note
            

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

        print("=== Extracted Text ===")
        print(payload)

        print("=== Extracted Text ===")
        print(extracted_text)


        analyzer = HealthTextAnalyzer(LANGUAGE_ENDPOINT, LANGUAGE_KEY)
        # Get FHIR bundle directly
        fhir_bundle = analyzer.get_fhir_bundle(payload)
        print("=== Raw FHIR ===")
        print(json.dumps(fhir_bundle, indent=2))

        compact = analyzer.preprocess_fhir_bundle(fhir_bundle)
        
        print("=== Compact FHIR ===")
        print(json.dumps(compact, indent=2))

        summarizer = OpenAISummarizer(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY,DEPLOYMENT_NAME)
        summary = summarizer.summarize_with_alerts(compact)

        return {
            "patient_id": patient_id,
            "summary": json.dumps(summary)       
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
