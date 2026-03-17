# 数据库巡检规范

- source_table: documents
- source_id: 3
- updated_at: 2026-03-17T09:20:00Z

## Content

PostgreSQL 生产库每周执行 VACUUM ANALYZE，并监控慢查询 Top20。
