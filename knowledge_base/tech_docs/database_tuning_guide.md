# 数据库性能调优指南

## 一、索引优化

### 1.1 缺失索引识别

通过分析慢查询日志（slow_query_log），发现以下高频查询缺少合适的索引：

- `orders` 表按 `user_id + status + created_at` 组合查询，当前只有 `user_id` 单列索引，需要添加复合索引
- `products` 表按 `category_id + price` 范围查询用于商品筛选，缺少复合索引
- `order_items` 表按 `order_id` 关联查询，虽然有外键但未显式建索引

建议添加以下索引：
```
ALTER TABLE orders ADD INDEX idx_user_status_time (user_id, status, created_at);
ALTER TABLE products ADD INDEX idx_category_price (category_id, price);
ALTER TABLE order_items ADD INDEX idx_order_id (order_id);
```

预计可将相关查询的响应时间从平均200ms降至20ms以内。

### 1.2 冗余索引清理

`users` 表目前有8个索引，其中存在冗余：
- `idx_email` 和 `idx_email_status` 两个索引，前者被后者覆盖，可删除 `idx_email`
- `idx_phone` 和 `idx_phone_unique` 功能重复

清理冗余索引可减少写入时的索引维护开销，预计可提升INSERT性能10-15%。

### 1.3 索引使用率监控

建议定期检查索引使用情况：
- 使用 `sys.schema_unused_indexes` 视图识别从未被使用的索引
- 使用 `EXPLAIN` 验证关键查询是否命中预期索引
- 对于写多读少的表，控制索引数量在5个以内

## 二、查询优化

### 2.1 慢查询治理

当前慢查询日志中（阈值>500ms），Top 5慢查询：

1. 商品搜索（全文模糊匹配）：平均2.1秒 → 建议迁移到Elasticsearch
2. 月度销售报表统计：平均1.8秒 → 建议使用预计算+物化视图
3. 用户订单历史（大翻页）：平均1.2秒 → 建议使用游标分页替代OFFSET
4. 库存批量更新：平均800ms → 建议使用批量UPDATE替代逐条更新
5. 多表JOIN报表查询：平均750ms → 建议简化JOIN层级或使用宽表

### 2.2 大翻页优化

当前分页查询使用 `LIMIT offset, count` 模式，当offset超过10万时性能急剧下降。

建议改用游标分页：
- 前端传入上一页最后一条记录的ID
- 查询条件改为 `WHERE id > last_id LIMIT count`
- 可将大翻页查询从秒级降至毫秒级

### 2.3 批量操作优化

库存扣减场景中，当前逐条UPDATE的方式在100条商品时需要800ms。

建议使用 `INSERT ... ON DUPLICATE KEY UPDATE` 或 `CASE WHEN` 批量更新，将100条记录的更新合并为单次SQL执行，预计可将时间从800ms降至50ms。

## 三、表结构与分区

### 3.1 大表拆分

`order_logs` 表目前有8000万行，查询和维护成本高。

建议按时间分区：
- 使用RANGE分区，按月分区
- 保留最近12个月数据在热分区
- 超过12个月的数据归档到冷存储

### 3.2 字段类型优化

- `orders.status` 使用VARCHAR(20)存储，建议改为TINYINT+枚举映射，节省存储空间
- `products.description` 使用TEXT类型，但95%的记录长度不超过500字符，建议改为VARCHAR(500)
- 时间戳统一使用DATETIME而非VARCHAR存储，便于范围查询和索引

### 3.3 读写分离

- 主库处理写操作和强一致性读
- 从库处理报表查询和数据分析
- 通过中间件（如ShardingSphere）实现自动路由
- 注意从库延迟通常在100ms以内，对于需要读己之写的场景仍需读主库

## 四、连接与配置

### 4.1 MySQL配置优化

针对当前8核16GB的服务器配置，建议调整以下参数：

- `innodb_buffer_pool_size = 10G`（总内存的60-70%）
- `innodb_log_file_size = 1G`（减少checkpoint频率）
- `max_connections = 200`（当前设置为100，高峰期不够用）
- `innodb_flush_log_at_trx_commit = 2`（牺牲少量持久性换取性能，适合非金融场景）

### 4.2 监控指标

建议关注：
- QPS和TPS趋势
- 慢查询数量趋势
- 锁等待时间和死锁次数
- Buffer Pool命中率（应>99%）
- 主从复制延迟
