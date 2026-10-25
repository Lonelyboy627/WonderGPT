# WonderGPT

从零实现的小型 **GPT Transformer**，用英文公版奇幻故事学习下一字符预测。项目参考 nanoGPT 的简洁训练方式，面向理解注意力机制、训练循环和自回归生成。

这是一个可训练的语言模型项目：不依赖外部模型 API，不下载预训练权重。小模型及少量训练步数只能生成初步的字符、词语和句式，不代表能够写出连贯长篇故事。

## 模型与功能

每层解码器由 `LayerNorm → 因果多头自注意力 → 残差连接 → LayerNorm → 前馈网络 → 残差连接` 组成。输入使用字符嵌入和可学习的位置嵌入；输出层与字符嵌入共享权重。

- PyTorch `scaled_dot_product_attention`，严格因果遮罩；支持加速注意力内核。
- GELU、Dropout、Pre-LayerNorm、多层解码器。
- AdamW、余弦学习率衰减、预热、梯度裁剪和梯度累积。
- 自动选择 CPU / CUDA；CUDA 支持 BF16 / FP16 混合精度。
- 独立验证集、JSONL 指标、原子保存 `latest.pt` / `best.pt`。
- 断点续训恢复权重、优化器、混合精度状态、步数及随机数状态。
- 根据提示生成文本，支持 temperature、top-k、随机种子和输出文件。

| 配置 | 层数 | 注意力头 | 隐藏维度 | 上下文字符数 | 参数量（100 字符词表） |
|---|---:|---:|---:|---:|---:|
| `smoke.toml` | 2 | 2 | 64 | 32 | 108,544 |
| `tiny.toml` | 4 | 4 | 128 | 128 | 822,528 |
| `small.toml` | 6 | 6 | 384 | 256 | 10,784,256 |
| `small_512.toml` | 6 | 6 | 384 | 512 | 10,882,560 |
| `small_1024.toml` | 6 | 6 | 384 | 1024 | 11,079,168 |

## Windows 快速开始

需要 Python 3.11+。在项目根目录运行，以下命令不需要激活虚拟环境。

```powershell
git clone https://github.com/Lonelyboy627/WonderGPT.git
cd WonderGPT
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe prepare.py
.\.venv\Scripts\python.exe train.py --config configs/smoke.toml
.\.venv\Scripts\python.exe sample.py --checkpoint out/smoke/best.pt --prompt "Once upon a time" --max-new-tokens 200 --device cpu
```

`smoke` 只训练 10 步，用于确认流程通畅，输出通常仍接近随机字符。开始正式的小模型实验：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/tiny.toml
.\.venv\Scripts\python.exe sample.py --checkpoint out/tiny/best.pt --prompt "The princess" --temperature 0.8 --top-k 40 --output out/tiny/story.txt
```

### 直接试用示例权重

[v0.1.0 Release](https://github.com/Lonelyboy627/WonderGPT/releases/tag/v0.1.0) 提供 82 万参数、训练 2,000 步的推理权重。下载后无需准备训练数据即可采样：

```powershell
New-Item -ItemType Directory -Force out | Out-Null
Invoke-WebRequest https://github.com/Lonelyboy627/WonderGPT/releases/download/v0.1.0/wondergpt-tiny.pt -OutFile out/wondergpt-tiny.pt
.\.venv\Scripts\python.exe sample.py --checkpoint out/wondergpt-tiny.pt --prompt "Once upon a time" --device cpu
```

这是用于教学的初步模型，仍会出现拼写错误和不连贯句子。实际采样见 [examples/tiny_sample.txt](examples/tiny_sample.txt)。发布权重只有模型和词表，不能续训；续训使用本地训练生成的 `latest.pt`。

Linux / macOS 可以使用 `python3 -m venv .venv`，把上述解释器路径换成 `.venv/bin/python`；macOS 安装依赖时使用默认 PyPI 索引。当前设备选项为 CPU 和 CUDA。

### NVIDIA GPU

先安装与驱动匹配的 CUDA 版 PyTorch。以下版本用于本项目的 RTX 4060 验证；不需要单独安装 CUDA Toolkit：

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe train.py --config configs/small.toml
```

`tiny.toml` 的 `device = "auto"` 也会使用可用的 GPU。显存不足时先降低 `batch_size`；可增加 `gradient_accumulation_steps` 保持每次更新的总字符数。`small` 的默认有效批量为 `16 × 256 × 4 = 16,384` 个字符。

### 更长上下文的 small 模型

`small_512.toml` 和 `small_1024.toml` 将模型增加为 6 层、6 个注意力头、384 维隐藏向量。二者分别使用 512 和 1024 个字符的上下文，是 tiny 的 4 倍和 8 倍。这里的长度按字符计数，不是单词数。

| 配置 | 每个微批次 | 梯度累积 | 每次更新处理的字符数 | 输出目录 |
|---|---:|---:|---:|---|
| `small_512.toml` | 8 × 512 | 4 | 16,384 | `out/small-512` |
| `small_1024.toml` | 4 × 1024 | 4 | 16,384 | `out/small-1024` |

新模型从零训练，计划 20,000 步；学习率用 200 步预热，并在 20,000 步内从 `0.0003` 余弦衰减到 `0.00003`。配置使用独立输出目录，保留 tiny 的断点。结构和位置嵌入尺寸发生了变化，不能使用 tiny 的 `--resume`。

```powershell
# 1024 字符上下文，从零训练
.\.venv\Scripts\python.exe train.py --config configs/small_1024.toml

# 中断后继续同一训练，目标总步数仍为 20,000
.\.venv\Scripts\python.exe train.py --config configs/small_1024.toml --resume out/small-1024/latest.pt --max-iters 20000

# 测试新模型，使用和 tiny 相同的提示及采样参数进行比较
.\.venv\Scripts\python.exe sample.py --checkpoint out/small-1024/best.pt --device cuda --prompt "The princess entered the forest and heard a voice behind her." --max-new-tokens 800 --temperature 0.7 --top-k 30 --num-samples 3 --seed 42 --output out/small-1024/quality_test.txt
```

若选择 512 字符版本，将配置路径换成 `configs/small_512.toml`，并将断点和采样输出目录换成 `out/small-512`。开始训练前调整批次参数；续训校验要求使用与该断点一致的批次和梯度累积设置。

新增上下文和容量是可比较的训练实验，不保证短时间内就能生成连贯故事。不同上下文配置的验证窗口不同，比较 loss 时也要同时记录配置并检查实际生成文本。

两套配置已在 8 GB RTX 4060 上通过训练、断点续训和采样检查，显存测量及检查范围见 [长上下文验证记录](examples/context_upgrade_validation.md)。

### 断点续训

使用原来的配置和输出目录。`--max-iters` 指目标总步数，例如把 2,000 步继续到 4,000 步：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/tiny.toml --resume out/tiny/latest.pt --max-iters 4000
```

模型结构、数据、词表和训练超参数必须一致。可以调整目标总步数、日志频率或设备；CPU 与 GPU 的数值结果不保证逐位一致。学习率衰减由配置里的 `lr_decay_iters` 决定，超过该步数后维持 `min_lr`。

首次训练不会覆盖已有运行目录。评估时及最后一步保存断点；中断后从最近保存的一步继续。不要同时向同一输出目录启动多个训练进程。

### 文本生成

```powershell
.\.venv\Scripts\python.exe sample.py --checkpoint out/tiny/best.pt --prompt "In the forest" --max-new-tokens 600 --temperature 0.8 --top-k 40 --num-samples 3 --seed 42 --output out/tiny/stories.txt
```

`--top-k 0` 关闭 top-k 筛选。较低温度使输出更集中；较高温度增加随机性。提示必须使用词表中已有的字符，这批英文语料不包含汉字。生成超过上下文长度时使用最近的上下文窗口；当前实现没有 KV cache。

## 训练数据

16 本经典英文原著，约 **104.6 万词、552.6 万字符**，词表包含 100 个字符，包括弯引号与少量重音字母。

| 作者 | 作品 |
|---|---|
| Lewis Carroll | *Alice's Adventures in Wonderland*、*Through the Looking-Glass* |
| L. Frank Baum | *The Wonderful Wizard of Oz*、*The Marvelous Land of Oz* |
| George MacDonald | *The Princess and the Goblin*、*The Princess and Curdie*、*Phantastes*、*At the Back of the North Wind*、*Lilith* |
| E. Nesbit | *Five Children and It*、*The Phoenix and the Carpet*、*The Story of the Amulet*、*The Enchanted Castle* |
| William Morris | *The Well at the World's End*、*The Wood Beyond the World* |
| Oscar Wilde | *The Happy Prince, and Other Tales*（五篇童话） |

训练集 5,068,218 个字符；验证集 457,865 个字符。整本留出 *Through the Looking-Glass* 和 *The Princess and Curdie*，比例约 8.3%。训练与验证仍有作者和系列重合，适合开发学习，不是严格独立的泛化基准。

语料已去掉平台声明、目录、插图标记及编辑序言；保留正文、章名和原有拼写。原始文件、书目快照、哈希及权利依据见 [来源记录](public_domain_fantasy_en/SOURCES.md) 和 [数据说明](public_domain_fantasy_en/README.md)。所选电子书均标注美国公版，作者最晚于 1924 年去世；其他地域的判断范围也记录在来源说明中。代码 MIT 许可不代替语料自身的权利依据。

分好的 TXT 和来源记录随仓库提供，训练不需要再次下载。准备脚本生成 little-endian uint16 的 `train.bin` / `val.bin` 和 JSON 词表；加载时验证文件哈希。不要把验证集再次拼入训练集。

```powershell
# 使用你自己的 UTF-8 文本与显式验证集
.\.venv\Scripts\python.exe prepare.py --train-file my_train.txt --val-file my_val.txt --out-dir data/custom
.\.venv\Scripts\python.exe train.py --config configs/tiny.toml --data-dir data/custom --out-dir out/custom

# 单个 TXT 的位置划分；前 90% 训练，后 10% 验证
.\.venv\Scripts\python.exe prepare.py --input-file my_stories.txt --out-dir data/custom

# 从归档的原始文件离线重建合并 TXT 和单书 TXT
.\.venv\Scripts\python.exe public_domain_fantasy_en/build_corpus.py
```

## 验证与项目结构

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖未来字符不可见、梯度与共享权重、固定小样本学习、采样和上下文裁剪、UTF-8 词表、数据文件完整性，以及 CPU 上断点续训与连续训练权重完全一致。GitHub Actions 在 Python 3.11 / 3.13 上执行这些测试，并跑完整的准备、训练、续训和采样命令。

```text
WonderGPT.py             兼容训练入口
prepare.py              数据准备入口
train.py / sample.py    训练与文本生成入口
export.py               导出不含优化器状态的推理权重
wondergpt/model.py      GPT、注意力、MLP 和解码器层
wondergpt/data.py       字符词表、二进制数据和批次采样
wondergpt/train.py      训练循环、评估和断点续训
wondergpt/sample.py     从断点生成文本
configs/                CPU 检查、小模型和单 GPU 配置
tests/                  模型与训练管线测试
public_domain_fantasy_en/ 语料、出处和重建脚本
```

`data/`、`out/`、虚拟环境、模型权重及本机编辑器配置不提交 Git。训练报告见 [examples/validation.md](examples/validation.md)。原来的 Bigram 练习脚本在本机 `legacy/bigram_original.py` 中备份。

## 参考与许可

- [nanoGPT](https://github.com/karpathy/nanoGPT)：简洁 GPT 实现与训练流程的参考。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 架构。
- [PyTorch 因果注意力文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
- [Project Gutenberg](https://www.gutenberg.org/)：原著文字来源，逐本链接见数据目录。

项目代码采用 [MIT License](LICENSE)。
