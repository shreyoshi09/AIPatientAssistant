import os
import json
import requests
from dotenv import load_dotenv
from textanalysis import HealthTextAnalyzer
from summarizer import OpenAISummarizer

# -----------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------
load_dotenv()
# Azure Cognitive Services (Language/FHIR) 
LANGUAGE_ENDPOINT =  os.getenv("AZURE_LANGUAGE_ENDPOINT") # e.g. https://xxx.cognitiveservices.azure.com
LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://xxx.openai.azure.com
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT") # e.g. "gpt-4o-mini"


# -----------------------------------------------------------
# 2. HELPER: PREPROCESS FHIR BUNDLE
# -----------------------------------------------------------
def preprocess_fhir_bundle(bundle: dict) -> dict:
    """Compact FHIR bundle for summarization"""
    compact = {"MedicationStatement": [], "Observation": [], "Condition": [], "AllergyIntolerance": []}

    if bundle.get("resourceType") != "Bundle":
        raise ValueError("Expected FHIR Bundle")

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")

        if rtype == "MedicationStatement":
            compact["MedicationStatement"].append({
                "medication": resource.get("medicationCodeableConcept", {}).get("text"),
                "dosage": [d.get("text") for d in resource.get("dosage", []) if d.get("text")]
            })

        elif rtype == "Observation":
            obs = {
                "code": resource.get("code", {}).get("text"),
                "value": None,
                "unit": None
            }
            if "valueQuantity" in resource:
                obs["value"] = resource["valueQuantity"].get("value")
                obs["unit"] = resource["valueQuantity"].get("unit")
            elif "valueString" in resource:
                obs["value"] = resource["valueString"]
            compact["Observation"].append(obs)

        elif rtype == "Condition":
            compact["Condition"].append({
                "code": resource.get("code", {}).get("text"),
                "clinicalStatus": resource.get("clinicalStatus", {}).get("text")
            })

        elif rtype == "AllergyIntolerance":
            compact["AllergyIntolerance"].append({
                "substance": resource.get("code", {}).get("text"),
                "criticality": resource.get("criticality")
            })

    # Drop empty keys
    return {k: v for k, v in compact.items() if v}

# -----------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == "__main__":
    payload = {
        "analysisInput": {
            "documents": [
                {
                    "id": "1",
                    "language": "en",
                    "text": "The doctor prescribed 200mg Ibuprofen."
                }
            ]
        },
        "tasks": [
            {
                "taskId": "analyze1",
                "kind": "Healthcare",
                "parameters": {"fhirVersion": "4.0.1"}
            }
        ]
    }

    analyzer = HealthTextAnalyzer(LANGUAGE_ENDPOINT, LANGUAGE_KEY)

    # Get FHIR bundle directly
    fhir_bundle = analyzer.get_fhir_bundle(payload)
    compact = preprocess_fhir_bundle(fhir_bundle)

    print("=== COMPACT FHIR ===")
    print(json.dumps(compact, indent=2))

    summarizer = OpenAISummarizer(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY,DEPLOYMENT_NAME)
    summary = summarizer.summarize_with_alerts(compact)

    print("\n=== SUMMARY & ALERTS ===")
    print(summary)