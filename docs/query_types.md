# Query Types

## 1) fact_lookup
- Purpose: find one concrete fact.
- Typical intent: "公司成立于哪一年？"
- Expected evidence: at least one `direct` evidence.

## 2) trend_analysis
- Purpose: compare changes across time.
- Typical intent: "最近三年的销售趋势如何？"
- Expected evidence: multiple time points covered.

## 3) cross_doc_summary
- Purpose: summarize themes from multiple files.
- Typical intent: "研发和销售部门的主要挑战是什么？"
- Expected evidence: multiple topic categories and sources.

## 4) numeric_query
- Purpose: retrieve or compare numbers.
- Typical intent: "2023年华东销售额是多少？"
- Expected evidence: numeric value plus direct source.

## 5) definition_process
- Purpose: ask for definitions/processes/steps.
- Typical intent: "新员工入职流程是什么？"
- Expected evidence: direct procedural description.
