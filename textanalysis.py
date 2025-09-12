import time
import requests

class HealthTextAnalyzer:
    def __init__(self, endpoint: str, key: str):
        """
        Initialize the analyzer with Azure Cognitive Services endpoint and key.
        """
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json"
        }

    def analyze(self, payload: dict, poll_interval: int = 5) -> dict:
        """
        Submit a healthcare text analysis job and return results.

        Args:
            payload (dict): JSON body for TA4H request
            poll_interval (int): Seconds between polling

        Returns:
            dict: Final job result
        """
        url = f"{self.endpoint}/language/analyze-text/jobs?api-version=2022-05-15-preview"

        # 1. Submit the job
        submit_resp = requests.post(url, headers=self.headers, json=payload)
        submit_resp.raise_for_status()
        job_location = submit_resp.headers["operation-location"]
        print(f"Job submitted. Polling at {job_location}")

        # 2. Poll until completion
        while True:
            poll_resp = requests.get(job_location, headers=self.headers)
            poll_resp.raise_for_status()
            result = poll_resp.json()

            status = result.get("status")
            if status in ["succeeded", "failed", "cancelled"]:
                break
            print(f"Job status: {status}... waiting {poll_interval}s")
            time.sleep(poll_interval)

        return result

    def get_fhir_bundle(self, payload: dict) -> dict:
        """
        Submit text and directly extract FHIR bundle if available.
        """
        result = self.analyze(payload)
        try:
            return result["tasks"]["items"][0]["results"]["documents"][0]["fhirBundle"]
        except KeyError:
            raise ValueError("FHIR bundle not found in response")

    def preprocess_fhir_bundle(self, bundle: dict) -> dict:
        """
        Convert FHIR bundle into a compact structure for OpenAI summarization.
        Falls back to raw clinical note text if structured resources are missing.
        """
        compact = {
            "patient": {},
            "conditions": [],
            "medications": [],
            "plan": [],
            "note_text": ""
        }

        if not bundle or "entry" not in bundle:
            return {"error": "Empty or invalid FHIR bundle"}

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")

            if rtype == "Patient":
                patient_name = resource.get("name", [{}])[0].get("text", "")
                compact["patient"]["name"] = patient_name
                compact["patient"]["gender"] = resource.get("gender", "unknown")
                # Fallback: if no note yet, stash text from patient name
                if not compact["note_text"] and patient_name:
                    compact["note_text"] = patient_name

            elif rtype == "Composition":
                subject_display = resource.get("subject", {}).get("display", "")
                if subject_display:
                    compact["note_text"] = subject_display  # stronger fallback

            elif rtype == "Condition":
                condition = resource.get("code", {}).get("text", "")
                if condition:
                    compact["conditions"].append(condition)

            elif rtype == "MedicationStatement":
                med = resource.get("medicationCodeableConcept", {}).get("text", "")
                if med:
                    compact["medications"].append(med)

            elif rtype in ("ServiceRequest", "ProcedureRequest"):
                proc = resource.get("code", {}).get("text", "")
                if proc:
                    compact["plan"].append(proc)

        # If no structured entities, rely on note_text
        if not compact["conditions"] and not compact["medications"] and compact["note_text"]:
            compact["fallback_mode"] = True
        else:
            compact["fallback_mode"] = False

        return compact