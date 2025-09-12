from openai import AzureOpenAI

class OpenAISummarizer:
    def __init__(self, endpoint: str, key: str, deployment: str, api_version: str = "2024-05-01-preview"):
        """
        Initialize Azure OpenAI client.
        
        Args:
            endpoint (str): Azure OpenAI endpoint (e.g. https://myresource.openai.azure.com)
            key (str): API key
            deployment (str): Deployment name of your GPT model
            api_version (str): API version (default: 2024-05-01-preview)
        """
        self.client = AzureOpenAI(
            api_key=key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.deployment = deployment

    def summarize_with_alerts(self, fhir_bundle: dict) -> str:
        """
        Summarize a FHIR bundle into plain text with critical alerts.
        
        Args:
            fhir_bundle (dict): JSON FHIR bundle
        
        Returns:
            str: Summary text with alerts
        """
      
        prompt = self.build_openai_prompt(fhir_bundle)

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a helpful clinical AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )

        return response.choices[0].message.content
    
    def build_openai_prompt(self,compact_fhir: dict) -> str:
            """
            Build a prompt for OpenAI summarization, handling fallback mode.
            """
            if compact_fhir.get("fallback_mode"):
                # FHIR had no structured entries, rely on raw note_text
                note_text = compact_fhir.get("note_text", "")
                prompt = f"""
        You are a medical summarization assistant.

        Extract the following information from the clinical note below:

        1. Patient Name and Gender
        2. Problems / Conditions
        3. Medications
        4. Plan / Recommendations

        Output in JSON format like:
        {{
        "patient": {{"name": "...", "gender": "..."}},
        "conditions": ["...", "..."],
        "medications": ["...", "..."],
        "plan": ["...", "..."]
        }}

        Clinical Note:
        \"\"\"
        {note_text}
        \"\"\"
        """
            else:
                # FHIR has structured data, use it
                prompt = f"""
        You are a medical summarization assistant.

        Summarize the following structured clinical data into a compact JSON:

        Patient: {compact_fhir.get('patient', {})}
        Conditions: {compact_fhir.get('conditions', [])}
        Medications: {compact_fhir.get('medications', [])}
        Plan: {compact_fhir.get('plan', [])}

        Output in JSON format like:
        {{
        "patient": {{"name": "...", "gender": "..."}},
        "conditions": ["...", "..."],
        "medications": ["...", "..."],
        "plan": ["...", "..."]
        }}
        """
            return prompt

