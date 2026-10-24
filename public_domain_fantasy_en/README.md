# 英文经典奇幻训练语料

16 本英文原著，适合 nanoGPT 类项目的文本预训练练习。包含爱丽丝两部、奥兹国前两部、George MacDonald 五部、E. Nesbit 四部、William Morris 两部，以及 Oscar Wilde 的五篇童话合集。逐本书目见 [SOURCES.md](SOURCES.md)。

## 文件

- `input.txt`：运行重建脚本后生成的 16 本合并语料，UTF-8 无 BOM、LF 换行；只含书名、作者和整理后的叙事文字。
- `books/`：运行重建脚本后生成的 16 个单书 TXT，便于挑选、调整比例。
- `train.txt` / `val.txt`：按整本书划分的参考训练集和验证集；验证集单独留出《Through the Looking-Glass》和《The Princess and Curdie》，约占总字符的 8.3%。使用这两个文件时，请勿再将 `input.txt` 加入训练。
- `manifest.json` / `stats.json`：来源、作者卒年、版权标记、清洗统计、字符统计、SHA-256。
- `sources/`：含原始许可的下载 TXT，以及逐本书目 HTML 快照，供追溯使用。
- `fetch_sources.py` / `build_corpus.py`：下载和重建脚本，仅依赖 Python 标准库。

`input.txt` 与 `train.txt` / `val.txt` 是两种读取方式，不应一起拼接；单书文件也已经包含在 `input.txt` 中。

## 读取方式

当前工作目录为 `D:\WonderGPT` 时，原先读取文本的位置可改为：

```python
with open('public_domain_fantasy_en/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
```

如果使用提供的整书划分，字符级训练可这样读取；后续不要再运行原脚本中按 90% 切分 `data` 的代码：

```python
from pathlib import Path
import torch

root = Path('public_domain_fantasy_en')
train_text = (root / 'train.txt').read_text(encoding='utf-8')
val_text = (root / 'val.txt').read_text(encoding='utf-8')
chars = sorted(set(train_text + val_text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda ids: ''.join(itos[i] for i in ids)
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)
```

文件是原著叙事语料，不含指令与回答配对。它们可用于学习文本生成；不能单靠这些故事训练出通用对话能力。验证集与训练集有作者和系列重合，适合开发练习，不作为独立泛化基准。

## 清理与核验

已去掉电子书平台页眉页脚、电子书制作说明、目录、封面出版信息、前置序言与献辞、插图标记、页码和尾部印刷说明。保留正文、章名、段落及原作中的诗歌、引文与注释；一般散文段落合并固定宽度换行，缩进内容和识别出的短诗行保留换行。标点、拼写、强调用下划线等沿用源文本，不是现代改写版。

逐本核验正文开头与结尾；带编号的章节数与标题在清洗前后完全一致，另外核验《The Marvelous Land of Oz》的 24 个章名和 Wilde 合集的 5 个故事名。检查严格 UTF-8 解码、乱码替换字符、残留平台说明、完整书籍重复和文件哈希。保留旧版文本可能存在的校录差异。

《The Well at the World's End》较长，约占总字符的五分之一，且语言较古雅。可通过 `books/` 调整其采样权重。此批为经典儿童奇幻、童话与早期成人奇幻的混合；不是当代史诗奇幻文风语料。

## 版权依据

每个所选版本的 Gutenberg 书目页都标记为美国公版；作者最晚于 1924 年去世。中国大陆财产权期限的筛选依据、地域范围和逐本链接见 [SOURCES.md](SOURCES.md)。公版状态有地域差异，不能把网站上的“免费阅读”自动等同于公版。此批只取英文原著文字，未纳入现代翻译或插图；作者姓名与作品名已保留。

## 重建

在本目录运行：

```powershell
python fetch_sources.py
python build_corpus.py
```

下载脚本复用已有原始文件；`source_manifest.json` 记录该批下载哈希。第二条命令离线重建所有清洗文本、清单及统计文件。仓库提供训练与验证 TXT；合并 TXT 和单书 TXT 可用此命令重建。Transformer 模型的训练、续训和生成方式见上一级 README.md。
