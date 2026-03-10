# Deep RAG Skill Spec

This folder is the spec layer of the project.

- `query_types.md`: classification and expected behavior by query intent.
- `evidence_model.md`: normalized evidence schema.
- `sufficiency_rules.md`: deterministic coverage checks before generation.

Implementation policy:
- Runtime orchestration is in Python code, not in this markdown spec.
- LLM prompts and tool contracts must align with this spec.
