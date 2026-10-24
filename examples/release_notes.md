WonderGPT v0.1.0 提供从零实现的字符级 GPT Transformer，以及英文公版奇幻语料。

- 因果多头注意力、位置嵌入、Pre-LayerNorm、残差连接、GELU 和共享输入输出权重。
- 数据准备、CPU / CUDA 训练、混合精度、梯度累积、断点续训、文本采样及推理权重导出。
- 16 本原著的训练与验证 TXT、来源快照、哈希和版权依据。
- 10 项本地测试全部通过；完成 CPU 流程和 RTX 4060 GPU 验证。

附带 `wondergpt-tiny.pt`：822,528 参数，在本语料上从零训练 2,000 步，固定验证窗口的字符交叉熵从 4.6297 降至 2.0365。它是教学示例，仍会生成不连贯句子及拼写错误。

下载后运行：

```powershell
python sample.py --checkpoint wondergpt-tiny.pt --prompt "Once upon a time" --device cpu
```

该文件仅含推理需要的模型与词表，无法续训。训练自己的模型后请用 `out/tiny/latest.pt` 续训。

SHA-256：`aeebf244f2f8b59707aae44bbe103beddd5ba8905f2a64119f386dbd8e3b05f2`

代码为 MIT 许可；语料的来源与权利范围见仓库 `public_domain_fantasy_en/SOURCES.md`。
