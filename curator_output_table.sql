CREATE TABLE IF NOT EXISTS `project_id.ace_curator.dpo_candidates`
(
  request_id                  STRING    NOT NULL,
  created_at                  TIMESTAMP NOT NULL,
  event_id                    STRING,

  tenant_id                   STRING,
  site_id                     STRING,
  modality                    STRING,
  body_part                   STRING,
  decision                    STRING,
  disposition                 STRING,
  actual_route                STRING,
  expected_route              STRING,
  actual_scope_key            STRING,
  expected_scope_key          STRING,

  -- Flattened DPO fields for export/querying. The same values are also kept in dpo_record.
  prompt_text                 STRING,
  preferred_output            STRING,
  dispreferred_output         STRING,

  has_dpo_pair                BOOL,
  is_dpo_ready                BOOL,
  missing_fields              JSON,

  reward_score                FLOAT64,
  reward_label                STRING,

  latency_ms                  FLOAT64,

  -- Request-level analytics for dashboards and monitoring.
  has_generated_output        BOOL,
  has_human_reference         BOOL,
  is_dpo_training_ready       BOOL,
  analytics_json              JSON,

  -- Human feedback metrics derived from upstream disposition and clinical accuracy.
  -- NULL when disposition is not yet available (human review pending).
  is_human_overridden                        BOOL,
  is_human_accepted                          BOOL,
  human_feedback_edit_distance               INT64,
  human_feedback_normalized_edit_distance    FLOAT64,
  human_reward_score                         FLOAT64,
  curator_reward_score                       FLOAT64,
  human_feedback_json                        JSON,

  -- Clinical accuracy metrics comparing impression text against structured findings.
  -- All computed from the SAME structured_findings ground truth.
  -- NULL when structured_findings was not present in the Reflector row.
  clinical_accuracy_json                      JSON,    -- Full clinical accuracy object
  clinical_accuracy_score                     FLOAT64, -- Composite score (0-1)
  finding_recall_generated                    FLOAT64, -- Recall for generated impression
  finding_precision_generated                 FLOAT64, -- Precision for generated impression
  false_negative_rate_generated               FLOAT64, -- FNR for generated impression
  critical_finding_capture_rate_generated     FLOAT64, -- Critical capture rate
  laterality_accuracy_generated               FLOAT64, -- Laterality accuracy
  severity_accuracy_generated                 FLOAT64, -- Severity accuracy
  impression_agreement_score                  FLOAT64, -- Jaccard similarity (gen vs human)
  missed_findings_generated                    JSON,    -- List of all missed finding labels
  missed_critical_generated                   JSON,    -- List of missed critical findings
  finding_checks_json                          JSON,    -- Granular per-finding evaluation array
  hallucinated_findings_generated             JSON,    -- List of hallucinated findings

  -- Field performance analytics (Layer 3 — per-field error rates).
  -- NULL when structured_findings was not present in the Reflector row.
  field_performance_json                      JSON,    -- Full field performance object
  laterality_error_rate                       FLOAT64, -- Error rate for laterality field
  severity_error_rate                         FLOAT64, -- Error rate for severity field
  projection_error_rate                       FLOAT64, -- Error rate for projection field
  measurement_error_rate                      FLOAT64, -- Error rate for measurement field
  overall_field_error_rate                    FLOAT64, -- Combined error rate across all fields

  -- Operational metrics (Layer 4 — per-request passthrough).
  -- Token usage is 0 when mode=python or no LLM call was made.
  total_tokens                                INT64,   -- Total LLM tokens consumed
  prompt_tokens                               INT64,   -- LLM prompt/input tokens
  completion_tokens                           INT64,   -- LLM completion/output tokens
  estimated_cost_usd                          FLOAT64, -- Estimated USD cost from token pricing

  -- Generator model version from upstream (NULL when not provided by Reflector).
  generator_model_version                     STRING,
  -- Reflector model version from upstream (NULL when not provided).
  reflector_model_version                     STRING,

  -- Clinical accuracy mode used for this request ("python" or "llm").
  clinical_accuracy_mode                      STRING,

  -- Reflector flagged issues (True = issues found, triggering review).
  reflector_triggered                         BOOL,

  -- Upstream timing passthrough (NULL when not provided by Reflector).
  agent_latency_ms                            FLOAT64, -- Generator agent execution time
  eval_latency_ms                             FLOAT64, -- Evaluation pipeline time

  -- Upstream queue timestamps (NULL when not provided by Reflector).
  submitted_to_queue_at                       TIMESTAMP, -- When study entered review queue
  human_approved_at                           TIMESTAMP, -- When radiologist approved


  -- Clean DPO payload. Shape: {"prompt": "...", "chosen": "...", "rejected": "..."}
  -- Identical chosen/rejected pairs may be stored for audit but should be filtered
  -- with is_dpo_training_ready=true for DPO fine-tuning exports.
  dpo_record                  JSON,

  -- Full Curator response for audit/debug/reporting.
  curator_response            JSON
)
PARTITION BY DATE(created_at)
CLUSTER BY tenant_id, body_part, is_dpo_training_ready, disposition
OPTIONS (
  description = "Curator output table for DPO candidates and request-level analytics. Stores filterable readiness, human feedback metrics, clinical accuracy, DPO record, and full response JSON."
);
