"""Original evidence provenance, independent of causal extraction or scoring."""
import json
import hashlib


def source_audit(records, trace):
    chunks = []
    for record in records:
        value = record.model_dump(mode="json")
        identity = json.dumps(value, sort_keys=True, ensure_ascii=False)
        chunks.append({**value, "chunk_id": "source-" + hashlib.sha256(identity.encode()).hexdigest()})
    return {"graph": {"source_chunks": chunks}, "retrieval_trace": trace,
            "context": json.dumps({"admitted_evidence": chunks}, ensure_ascii=False),
            "weighting_status": "causal_not_requested"}
