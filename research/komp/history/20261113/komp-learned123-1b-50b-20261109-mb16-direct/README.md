# Learned+123 1B / 50B batch16 direct resume

用户明确要求直接启动，不再做数值preflight。本版本每卡batch16 accum1 global512，32H100，从已保存step1068/1,119,879,168 tokens完整checkpoint恢复模型/Adam/RNG/输入状态，目标总50B。模型、数据、优化器、学习率进度不变。保留源码固定/数据哈希、父checkpoint和32rank状态哈希校验、严格加载、非有限梯度报错。数值预验收明确记为skipped_by_explicit_user_request，不伪造passed。
