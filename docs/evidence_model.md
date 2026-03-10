# Evidence Model

Every extracted evidence item should contain:

- `evidence_id`: unique id inside one query run
- `source_file`: file name
- `source_location`: section/page/sheet id
- `source_format`: `markdown|pdf|excel`
- `evidence_type`: `text|table|number|conclusion`
- `evidence_strength`: `direct|derived|opinion`
- `content`: normalized evidence text
- `time_period`: optional time label
- `numeric_value`: optional numeric value
- `unit`: optional numeric unit
- `topic_category`: optional topic tag
- `dedup_key`: optional dedup key
- `extraction_quality`: `good|degraded`

Notes:
- `direct` evidence is preferred for final answer.
- `numeric_query` must include at least one evidence with `numeric_value`.
