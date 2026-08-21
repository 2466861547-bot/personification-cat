[CLOSED] # Seq2SeqTrainer API 兼容性调试

## 会话信息
- **Session ID**: trainer-api-compat
- **时间**: 2026-08-20
- **问题**: Seq2SeqTrainer.__init__() 收到意外参数 `place_model_on_device` 和 `tokenizer`
- **影响**: Whisper 微调训练无法启动

## 错误日志
```
TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument 'place_model_on_device'
TypeError: Seq2SeqTrainer.__init__() got an unexpected keyword argument 'tokenizer'
```

## 假设验证结果

### H1: `place_model_on_device` 不是 Seq2SeqTrainer 的合法参数 ✅ 已确认
- transformers 5.x 已移除此参数
- **修复**: 从 trainer_kwargs 中移除

### H2: `tokenizer` 参数在新版 transformers 中已重命名为 `processing_class` ✅ 已确认
- **修复**: 主路径使用 `processing_class`，回退路径使用 `getattr` 安全访问

### H3: CPU 设备场景需要不同的处理方式 ✅ 已解决
- **修复**: 使用 `self.model.to("cpu")` 确保模型在 CPU 上，不再依赖 `place_model_on_device`

### H4: transformers 版本与代码不匹配 ✅ 已解决
- 代码现在兼容新旧两个版本的 API

## 修复内容

### 文件: `src/models/whisper_finetune.py`

1. **移除 `place_model_on_device` 参数**: 从 `trainer_kwargs` 中删除此不受支持的参数
2. **替代 CPU 处理**: 使用 `self.model.to("cpu")` 确保模型留在 CPU 上
3. **增强回退逻辑**: except 块同时捕获 `TypeError` 和 `AttributeError`，使用 `getattr` 安全访问 tokenizer
4. **简化 try/except**: 主路径直接使用 `processing_class=self.processor`

## 状态
- [x] 已记录假设
- [x] 已实施修复
- [x] 已验证语法正确性
- [x] 已更新文档
- [ ] 等待用户验证
