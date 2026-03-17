DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  title TEXT,
  content TEXT,
  updated_at TEXT
);
INSERT INTO documents (id, title, content, updated_at) VALUES
  (1, 'Aurora 发布计划', 'Aurora 项目 v1.0 发布窗口定为 2026-03-20，灰度策略为 5% 到 30% 再到 100%。', '2026-03-17T09:00:00Z'),
  (2, '销售同比分析', '2025 年 Q4 销售额 13.2M，较去年同期增长 18%，主要由华东区贡献。', '2026-03-17T09:10:00Z'),
  (3, '数据库巡检规范', 'PostgreSQL 生产库每周执行 VACUUM ANALYZE，并监控慢查询 Top20。', '2026-03-17T09:20:00Z');
