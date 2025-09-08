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
        prompt = f"""
        You are a clinical assistant. Given the following FHIR bundle in JSON format:

        {fhir_bundle}

        1. Provide a short plain text clinical summary.  
        2. Highlight 1–2 critical alerts (e.g., allergies, drug interactions, abnormal values) if present.
        """

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
