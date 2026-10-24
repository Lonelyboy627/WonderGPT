# 本地验证记录

日期：2026-10-24。Python 3.13.12，PyTorch 2.11.0+cu128，Windows，NVIDIA GeForce RTX 4060 Laptop GPU（8 GB），驱动 576.40。

## 结果

| 验证 | 结果 |
|---|---|
| 10 项模型与数据管线测试 | 全部通过 |
| 因果性 | 改变未来字符不会改变前缀位置的输出 |
| CPU 断点续训 | 含 Dropout 的 2 步 + 2 步与连续 4 步权重逐位一致 |
| CPU CLI | 数据准备、10 步训练、续训到 12 步、加载权重和采样通过 |
| Tiny GPU 训练 | 822,528 参数，2,000 步，BF16，batch 16，context 128 |
| Tiny 初始验证交叉熵 | 4.629697 |
| Tiny 第 2,000 步验证交叉熵 | 2.036491 |
| Tiny GPU 续训 | 从第 2,000 步继续到第 2,020 步，验证交叉熵 2.037725；最佳断点仍在第 2,000 步 |
| Small GPU 流程检查 | 10,784,256 参数，2 步，context 256，梯度累积 4；训练与评估均正常 |
| 推理权重导出 | 不包含优化器和随机数状态，CPU 加载与采样通过 |

交叉熵是字符级、以自然对数计算的平均损失。评估使用固定随机种子的 10 个验证批次，每批 16 × 128 字符；不是全验证集的精确平均值。Small 的两步检查只确认该配置能运行，不代表完成模型质量训练。

完整 tiny 评估记录见 [tiny_metrics.jsonl](tiny_metrics.jsonl)。`elapsed_seconds` 从每次训练进程开始计时，续训时会重新计时。

## 重现

```powershell
python prepare.py
python -m unittest discover -s tests -v
python train.py --config configs/tiny.toml --device cuda
python train.py --config configs/tiny.toml --device cuda --resume out/tiny/latest.pt --max-iters 2020
python export.py --checkpoint out/tiny/best.pt --output out/wondergpt-tiny.pt
python sample.py --checkpoint out/wondergpt-tiny.pt --prompt "Once upon a time" --max-new-tokens 500 --temperature 0.8 --top-k 40 --seed 1337 --device cpu --output examples/tiny_sample.txt
```

GPU 内核、PyTorch 版本及设备变化可能改变数值结果和随机采样文本。实际示例没有人工润色，仍含有明显的拼写和语法错误。

## 发布权重

- 文件：`wondergpt-tiny.pt`
- 大小：3,310,493 bytes
- 训练步数：2,000（最佳验证断点）
- SHA-256：`aeebf244f2f8b59707aae44bbe103beddd5ba8905f2a64119f386dbd8e3b05f2`
- 类型：仅推理，配合 `sample.py`；不支持断点续训。
- 语料及权利依据：[SOURCES.md](../public_domain_fantasy_en/SOURCES.md)
