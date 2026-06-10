from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


RoleType = Literal["radiologist", "clinician", "resident", "ai_qc", "system"]
ModalityType = Literal["CR", "DX", "CT", "MR", "US", "NM", "MG", "XA", "OT"]
PresenceType = Literal["present", "absent", "uncertain"]


class UserContext(BaseModel):
    user_id: str
    role: RoleType
    department: Optional[str] = None


class PatientContext(BaseModel):
    patient_id: str
    age: Optional[str] = None
    sex: Optional[Literal["M", "F", "O", "U"]] = "U"


class ExamContext(BaseModel):
    modality: ModalityType
    body_part: str
    procedure_code: Optional[str] = None
    laterality: Optional[Literal["L", "R", "B", "U"]] = None


class ComparisonContext(BaseModel):
    prior_study_uids: Optional[List[str]] = None
    prior_report_text: Optional[str] = None
    prior_key_findings: Optional[Dict[str, Any]] = None


class CaseContext(BaseModel):
    accession_number: str
    study_instance_uid: Optional[str] = None
    patient: PatientContext
    exam: ExamContext
    clinical_indication: str
    history: Optional[str] = None
    comparison: Optional[ComparisonContext] = None


class StructuredFinding(BaseModel):
    label: str
    presence: PresenceType
    severity: Optional[str] = None
    location: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    measurements: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None


class ObservationPayload(BaseModel):
    structured_findings: List[StructuredFinding] = Field(default_factory=list)
    key_measurements: Optional[Dict[str, Any]] = None
    critical_flags: Optional[List[str]] = None


class SourceText(BaseModel):
    findings_text: Optional[str] = None
    draft_impression_text: Optional[str] = None


class AttachedDoc(BaseModel):
    type: Literal["order", "clinical_note", "discharge_summary", "lab", "ecg", "pathology"]
    text: Optional[str] = None
    doc_id: Optional[str] = None


class Inputs(BaseModel):
    observation_payload: ObservationPayload
    source_text: Optional[SourceText] = None
    attached_docs: Optional[List[AttachedDoc]] = None


class StyleGuide(BaseModel):
    tone: Optional[Literal["concise", "standard", "detailed"]] = "concise"
    format: Optional[Literal["bullets", "paragraph", "numbered"]] = "numbered"
    language: Optional[str] = "en-US"
    institutional_phrasing: Optional[str] = None


class ACEContext(BaseModel):
    playbook_id: str
    playbook_version: str
    specialty_profile: Optional[str] = "radiology"
    style_guide: Optional[StyleGuide] = None


class RetrievalFilters(BaseModel):
    modality: Optional[str] = None
    body_part: Optional[str] = None
    vendor_algo: Optional[str] = None
    guideline_set: Optional[List[str]] = None


class RetrievalConfig(BaseModel):
    enabled: bool = True
    top_k: int = 2
    filters: Optional[RetrievalFilters] = None
    vector_namespace: Optional[str] = None


class CriticalResultPolicy(BaseModel):
    escalate_if: Optional[List[str]] = None
    escalation_text: Optional[str] = None


class Constraints(BaseModel):
    no_diagnosis_beyond_evidence: bool = True
    avoid_phi_in_output: bool = True
    must_include: Optional[List[str]] = None
    must_not_include: Optional[List[str]] = None
    critical_result_policy: Optional[CriticalResultPolicy] = None


class ModelConfig(BaseModel):
    provider: Optional[str] = "vertex"
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class GeneratorRequest(BaseModel):
    tenant_id: str
    site_id: Optional[str] = None
    request_id: str
    user: UserContext
    case: CaseContext
    inputs: Inputs
    ace_context: ACEContext
    retrieval: Optional[RetrievalConfig] = None
    constraints: Optional[Constraints] = None
    model: Optional[ModelConfig] = None


class ImpressionStructured(BaseModel):
    diagnoses: List[str] = Field(default_factory=list)
    differentials: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    follow_up: List[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    check: str
    passed: bool
    details: Optional[str] = None


class GeneratorResponse(BaseModel):
    request_id: str
    case_ref: Dict[str, Optional[str]]
    output: Dict[str, Any]
    provenance: Dict[str, Any]
    trace: Dict[str, Any]
    quality: Dict[str, Any]