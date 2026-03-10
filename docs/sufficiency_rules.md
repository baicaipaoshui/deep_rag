# Sufficiency Rules

## trend_analysis
- Parse year range from `time_range`.
- Evidence set is sufficient only when all expected years are covered.

## cross_doc_summary
- Must cover all requested `expected_dimensions` when provided.
- If dimensions are empty, require at least 2 source files and 2 topic categories.

## numeric_query
- Must contain at least one `direct` evidence with non-null `numeric_value`.

## fact_lookup / definition_process
- Must contain at least one `direct` evidence.

## Fallback behavior
- If missing information remains and supplement rounds exceed limit, return
  `action=force_generate` with explicit `missing` list.
