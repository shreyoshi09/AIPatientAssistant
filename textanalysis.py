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
