"""
Azure Text Analytics for Health – PDF → structured JSON extractor

Features
- Upload a PDF, extract text with PyMuPDF (fitz)
- Call Azure Text Analytics for Health to extract diagnoses, medications, symptoms, labs
- Fallback to rule-based regex extractor if Azure is unavailable or errors
- Exposes a FastAPI endpoint and a simple CLI

Dependencies (requirements.txt)
fastapi
uvicorn
python-multipart
pymupdf
azure-ai-textanalytics
azure-core

Env vars
- AZURE_LANGUAGE_ENDPOINT or LANGUAGE_ENDPOINT
- AZURE_LANGUAGE_KEY or LANGUAGE_KEY

Run (API)
  uvicorn azure_health_pdf_extractor:app --host 0.0.0.0 --port 8000 --reload

Test (curl)
  curl -F file=@sample.pdf http://localhost:8000/extract-health

Run (CLI)
  python azure_health_pdf_extractor.py path/to/file.pdf > out.json

⚠️ Notes
- This does NOT OCR scanned PDFs. If your PDFs are images, add OCR (e.g., Tesseract) before sending to Azure.
- Not medical advice; verify outputs clinically.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------- PDF TEXT EXTRACTION (PyMuPDF) ----------
try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    fitz = None

# ---------- AZURE SDK ----------
try:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
except Exception:  # pragma: no cover
    TextAnalyticsClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore
    HttpResponseError = Exception  # type: ignore

# ---------- FASTAPI (optional server) ----------
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException, Request
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    File = None  # type: ignore
    UploadFile = None  # type: ignore
    HTTPException = Exception  # type: ignore
    JSONResponse = None  # type: ignore


# ============================
# Utility & Data Structures
# ============================

@dataclass
class Medication:
    name: str
    normalized: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    form: Optional[str] = None
    confidence: Optional[float] = None
    assertion: Optional[Dict[str, Any]] = None
    source: str = "azure"

@dataclass
class Diagnosis:
    text: str
    normalized: Optional[str] = None
    confidence: Optional[float] = None
    assertion: Optional[Dict[str, Any]] = None
    source: str = "azure"

@dataclass
class Symptom:
    text: str
    normalized: Optional[str] = None
    confidence: Optional[float] = None
    assertion: Optional[Dict[str, Any]] = None
    source: str = "azure"

@dataclass
class Lab:
    name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    operator: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "azure"


# ============================
# PDF utilities
# ============================

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"Unable to open PDF: {e}")
    parts: List[str] = []
    for page in doc:  # keep natural reading order
        parts.append(page.get_text("text", sort=True))
    return "\n\f\n".join(parts)


def chunk_text(text: str, max_chars: int = 120_000) -> List[str]:
    """Split text into <= max_chars chunks. Azure Health allows up to 125k per doc.
    We keep a margin for safety.
    """
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to break at a newline for cleanliness
        nl = text.rfind("\n", start, end)
        if nl == -1 or nl <= start + 1000:  # avoid tiny last piece
            nl = end
        chunks.append(text[start:nl])
        start = nl
    return chunks


# ============================
# Azure client helper
# ============================

def get_azure_client() -> Optional[TextAnalyticsClient]:
    endpoint = (
        os.getenv("AZURE_LANGUAGE_ENDPOINT")
        or os.getenv("LANGUAGE_ENDPOINT")
        or os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")
    )
    key = (
        os.getenv("AZURE_LANGUAGE_KEY")
        or os.getenv("LANGUAGE_KEY")
        or os.getenv("AZURE_TEXT_ANALYTICS_KEY")
    )
    if not endpoint or not key or TextAnalyticsClient is None:
        return None
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# ============================
# Azure Healthcare extraction
# ============================

def normalize_assertion(assertion: Any) -> Optional[Dict[str, Any]]:
    if not assertion:
        return None
    out = {}
    # Attributes vary by SDK version; handle generically
    for attr in ("certainty", "conditionality", "association", "is_negated"):
        if hasattr(assertion, attr):
            out[attr] = getattr(assertion, attr)
    return out or None


def entity_key(e: Any) -> Tuple[Any, ...]:
    return (getattr(e, "text", None), getattr(e, "category", None), getattr(e, "offset", None))


def analyze_with_azure(text: str) -> Dict[str, Any]:
    client = get_azure_client()
    if client is None:
        raise RuntimeError("Azure client not configured")

    documents = chunk_text(text)

    diagnoses: Dict[Tuple[Any, ...], Diagnosis] = {}
    symptoms: Dict[Tuple[Any, ...], Symptom] = {}
    meds: Dict[Tuple[Any, ...], Medication] = {}
    labs: Dict[Tuple[Any, ...], Lab] = {}

    # Batch by Azure limit: 25 docs per async request
    BATCH = 25
    for i in range(0, len(documents), BATCH):
        batch = documents[i : i + BATCH]
        payload = [
            {"id": str(i + j), "language": "en", "text": t}
            for j, t in enumerate(batch)
            if t and t.strip()
        ]
        if not payload:
            continue
        try:
            poller = client.begin_analyze_healthcare_entities(payload)
            result_pages = list(poller.result())
        except HttpResponseError as e:
            raise RuntimeError(f"Azure call failed: {e}")

        for doc_result in result_pages:
            if getattr(doc_result, "is_error", False):
                # skip errored docs but continue others
                continue

            # First pass: capture standalone entities of interest
            for ent in doc_result.entities:
                cat = str(getattr(ent, "category", "")).upper()
                ek = entity_key(ent)
                norm = getattr(ent, "normalized_text", None)
                conf = getattr(ent, "confidence_score", None)
                assertion = normalize_assertion(getattr(ent, "assertion", None))

                if cat == "DIAGNOSIS":
                    diagnoses[ek] = Diagnosis(text=ent.text, normalized=norm, confidence=conf, assertion=assertion)
                elif cat == "SYMPTOM_OR_SIGN":
                    symptoms[ek] = Symptom(text=ent.text, normalized=norm, confidence=conf, assertion=assertion)
                elif cat == "MEDICATION_NAME":
                    meds.setdefault(
                        ek,
                        Medication(name=ent.text, normalized=norm, confidence=conf, assertion=assertion),
                    )
                elif cat == "EXAMINATION_NAME":
                    labs.setdefault(ek, Lab(name=ent.text, confidence=conf))

            # Second pass: use relations to enrich medications and labs
            for rel in getattr(doc_result, "entity_relations", []) or []:
                # Build a map of role -> entity
                roles = {getattr(r, "name", "").upper(): r.entity for r in rel.roles}
                # Medication-centric relations
                med_ent = None
                for k, v in roles.items():
                    if "MEDICATION" in k:  # role like 'MEDICATION'
                        med_ent = v
                        break
                if med_ent is not None:
                    mk = entity_key(med_ent)
                    m = meds.setdefault(
                        mk,
                        Medication(name=getattr(med_ent, "text", ""), normalized=getattr(med_ent, "normalized_text", None)),
                    )
                    # map common roles to fields
                    if "DOSAGE" in roles:
                        m.dosage = getattr(roles["DOSAGE"], "text", None)
                    if "FREQUENCY" in roles:
                        m.frequency = getattr(roles["FREQUENCY"], "text", None)
                    # Route may be 'ROUTE' or 'ROUTE_OF_ADMINISTRATION'
                    if "ROUTE" in roles:
                        m.route = getattr(roles["ROUTE"], "text", None)
                    if "ROUTEOFADMINISTRATION" in roles or "ROUTE_OF_ADMINISTRATION" in roles:
                        ent_obj = roles.get("ROUTEOFADMINISTRATION") or roles.get("ROUTE_OF_ADMINISTRATION")
                        if ent_obj:
                            m.route = getattr(ent_obj, "text", None)
                    if "FORM" in roles or "MEDICATION_FORM" in roles:
                        ent_obj = roles.get("FORM") or roles.get("MEDICATION_FORM")
                        if ent_obj:
                            m.form = getattr(ent_obj, "text", None)

                # Lab / measurement relations
                exam_ent = None
                for k, v in roles.items():
                    if "EXAMINATION" in k:  # role like 'EXAMINATION_NAME'
                        exam_ent = v
                        break
                if exam_ent is not None:
                    lk = entity_key(exam_ent)
                    lab = labs.setdefault(lk, Lab(name=getattr(exam_ent, "text", "")))
                    if "MEASUREMENT_VALUE" in roles:
                        lab.value = getattr(roles["MEASUREMENT_VALUE"], "text", None)
                    if "MEASUREMENT_UNIT" in roles:
                        lab.unit = getattr(roles["MEASUREMENT_UNIT"], "text", None)
                    if "RELATIONAL_OPERATOR" in roles:
                        lab.operator = getattr(roles["RELATIONAL_OPERATOR"], "text", None)

    return {
        "source": "azure",
        "diagnoses": [asdict(v) for v in diagnoses.values()],
        "medications": [asdict(v) for v in meds.values()],
        "symptoms": [asdict(v) for v in symptoms.values()],
        "labs": [asdict(v) for v in labs.values()],
    }


# ============================
# Rule-based fallback
# ============================

MED_DOSE = r"(?P<dose>\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|units|ml))"
MED_FREQ = r"(?P<freq>once daily|twice daily|daily|bid|tid|qid|q\d+h|q\d{1,2}h)"
MED_ROUTE = r"(?P<route>po|iv|im|sc|sl|topical|inhaled|pr)"

LAB_UNIT = (
    r"%|mg/dL|mmol/L|g/L|g/dL|U/L|IU/L|ng/mL|pg/mL|mEq/L|mmHg|bpm|°C|10\^9/L|x10\^9/L|k/\u00B5L"
)

# simple dictionaries for diagnoses / symptoms seed words (extend as needed)
DIAG_WORDS = [
    "diabetes", "hypertension", "pneumonia", "asthma", "copd", "migraine", "anemia", "covid-19",
]
SYMPTOM_TRIGGERS = [
    "complains of", "reports", "presents with", "symptoms include", "c/o", "denies",
]


def extract_rule_based(text: str) -> Dict[str, Any]:
    meds: Dict[str, Medication] = {}
    diagnoses: Dict[str, Diagnosis] = {}
    symptoms: List[Symptom] = []
    labs: List[Lab] = []

    # ---- Medications: look for lines with a drug-like token + dose/freq/route
    med_line_re = re.compile(
        rf"^(?P<name>[A-Z][A-Za-z0-9\- ]{{2,}})\s+(?:{MED_DOSE})?(?:\s+{MED_ROUTE})?(?:\s+{MED_FREQ})?",
        re.IGNORECASE | re.MULTILINE,
    )
    for m in med_line_re.finditer(text):
        name = m.group("name").strip()
        entry = meds.get(name) or Medication(name=name, source="rule_based")
        entry.dosage = entry.dosage or (m.groupdict().get("dose") or None)
        entry.frequency = entry.frequency or (m.groupdict().get("freq") or None)
        entry.route = entry.route or (m.groupdict().get("route") or None)
        meds[name] = entry

    # ---- Diagnoses: capture after headers
    for header in ("diagnosis", "diagnoses", "assessment", "impression"):
        sec = re.findall(rf"{header}\s*:\s*(.+)", text, flags=re.IGNORECASE)
        for s in sec:
            for piece in re.split(r"[,;\n]", s):
                w = piece.strip().strip(".- ")
                if not w:
                    continue
                key = w.lower()
                if key not in diagnoses:
                    diagnoses[key] = Diagnosis(text=w, source="rule_based")

    # Also scan for common diagnosis words standalone
    for w in DIAG_WORDS:
        for m in re.finditer(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE):
            key = w.lower()
            if key not in diagnoses:
                diagnoses[key] = Diagnosis(text=w, source="rule_based")

    # ---- Symptoms: capture phrases after triggers
    for trig in SYMPTOM_TRIGGERS:
        for m in re.finditer(rf"{trig}\s+([^\.;\n]+)", text, flags=re.IGNORECASE):
            phrase = m.group(1)
            for s in re.split(r",|and|/", phrase):
                cand = s.strip().strip(".- ")
                if cand:
                    symptoms.append(Symptom(text=cand, source="rule_based"))

    # ---- Labs: pattern like "HbA1c: 7.2%" or "Sodium 134 mmol/L"
    lab_re = re.compile(
        rf"(?P<name>[A-Za-z][A-Za-z0-9\-/ ]{{2,}})\s*[:=]\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>{LAB_UNIT})?",
        re.IGNORECASE,
    )
    for m in lab_re.finditer(text):
        labs.append(
            Lab(
                name=m.group("name").strip(),
                value=m.group("value"),
                unit=(m.group("unit") or None),
                source="rule_based",
            )
        )

    return {
        "source": "rule_based",
        "diagnoses": [asdict(v) for v in diagnoses.values()],
        "medications": [asdict(v) for v in meds.values()],
        "symptoms": [asdict(s) for s in symptoms],
        "labs": [asdict(l) for l in labs],
    }


# ============================
# Coordinator
# ============================

def extract_health_from_text(text: str) -> Dict[str, Any]:
    # Try Azure first
    try:
        return analyze_with_azure(text)
    except Exception:
        # Fallback to rule-based
        return extract_rule_based(text)


def extract_health_from_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    text = extract_text_from_pdf_bytes(pdf_bytes)
    return extract_health_from_text(text)


# ============================
# FastAPI server
# ============================

if FastAPI:
    app = FastAPI(title="PDF → Azure Health Extractor", version="1.0.0")

    @app.get("/health")
    def health() -> Dict[str, str]:  # pragma: no cover
        return {"status": "ok"}

    @app.post("/extract-health")
    async def extract_health(file: UploadFile = File(...)):
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file")
        try:
            pdf_bytes = await file.read()
            result = extract_health_from_pdf_bytes(pdf_bytes)
            return JSONResponse(content=result)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ============================
# CLI entrypoint
# ============================

def _cli(path: str) -> int:
    with open(path, "rb") as f:
        data = f.read()
    result = extract_health_from_pdf_bytes(data)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".pdf"):
        sys.exit(_cli(sys.argv[1]))
    else:
        print("Usage: python azure_health_pdf_extractor.py <file.pdf>")
        sys.exit(1)

    @app.post("/extract-health-raw")
    async def extract_health_raw(request: Request):
        """Alternative endpoint: accepts RAW PDF bytes (Content-Type: application/pdf).
        Useful for Windows PowerShell 5.x where multipart form uploads are awkward.
        """
        try:
            if request.headers.get("content-type", "").lower().startswith("application/pdf"):
                pdf_bytes = await request.body()
                if not pdf_bytes:
                    raise HTTPException(status_code=400, detail="Empty body")
                result = extract_health_from_pdf_bytes(pdf_bytes)
                return JSONResponse(content=result)
            raise HTTPException(status_code=415, detail="Send application/pdf body or use /extract-health for multipart upload")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
