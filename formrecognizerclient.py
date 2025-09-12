import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from fastapi import UploadFile, HTTPException

class FormRecognizerClient:
    def __init__(self, endpoint: str = None, key: str = None):
        self.endpoint = endpoint or os.getenv("FORM_RECOGNIZER_ENDPOINT")
        self.key = key or os.getenv("FORM_RECOGNIZER_KEY")

        if not self.endpoint or not self.key:
            raise ValueError("Form Recognizer endpoint and key must be provided")

        self.client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

    def extract_text_from_pdf(self, file: UploadFile) -> str:
        """Extract text from PDF using prebuilt-document model"""
        try:
            file.file.seek(0)  # reset pointer
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-document",
                document=file.file
            )
            result = poller.result()

            text = ""
            for page in result.pages:
                for line in page.lines:
                    text += line.content + " "
            return text.strip()

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Form Recognizer error: {str(e)}")
