# 最终模型

仅 `ppo_mlp_residual.zip` 作为最终提交方案的可复现推理 checkpoint 纳入版本控制。其余训练 checkpoint 可由对应 YAML 配置重新生成，继续由 `.gitignore` 排除，以避免仓库被消融实验模型占满。

一键运行脚本会优先使用该模型；若文件不存在，则按 `training/configs/ppo_mlp_residual.yaml` 完整训练。

- 文件大小：4,035,080 bytes
- SHA-256：`2F7593F4F250B775FA6A5B89D2DFC212ED00F0FC1003E602B8EA7A54E339DA4D`
- 隐私处理：仅将 SB3 元数据中的 `tensorboard_log` 绝对路径改为 `training/tb_logs`，策略参数未修改。
