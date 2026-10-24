"""Build and validate UTF-8 training text from the saved Gutenberg sources."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARTS = {
    11: r'^CHAPTER I\.$', 12: r'^CHAPTER I\.$',
    55: r'^Chapter I$', 54: r'^Tip Manufactures a Pumpkinhead$',
    708: r'^CHAPTER 1$', 709: r'^CHAPTER 1$', 325: r'^CHAPTER I$',
    225: r'^CHAPTER I\. THE HAY-LOFT$', 1640: r'^CHAPTER I\. THE LIBRARY$',
    778: r'^CHAPTER 1$', 836: r'^CHAPTER 1\. THE EGG$',
    837: r'^CHAPTER I\.$', 34219: r'^CHAPTER I$',
    169: r'^BOOK ONE$', 3055: r'^CHAPTER I: OF GOLDEN WALTER AND HIS FATHER$',
    902: r'^The Happy Prince\.$',
}
EXPECTED_CHAPTERS = {11: 12, 12: 12, 55: 24, 708: 32, 709: 35, 325: 25,
                     225: 38, 1640: 47, 778: 11, 836: 12, 837: 14,
                     34219: 12, 169: 119, 3055: 36}
TAILS = {709: r'^End of Project Gutenberg', 778: r'^\*Project Gutenberg Etext',
         34219: r'^[ \t]*UNWIN BROTHERS, LIMITED',
         902: r'^[ \t]*Printed by BALLANTYNE'}
CHAPTER = r'^(?:CHAPTER|Chapter) (?:[IVXLCDM]+|\d+)(?:[.:] [^\n]*|\.?)$'


def write_text(path, text):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def clean(record):
    number = record['gutenberg_id']
    raw = (ROOT / record['raw_file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['raw_sha256']
    source = raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    begin = re.search(r'^\*\*\* START OF[^\n]*\*\*\*[ \t]*$', source, re.M)
    end = re.search(r'^\*\*\* END OF[^\n]*\*\*\*', source, re.M)
    assert begin and end and begin.end() < end.start()
    body = source[begin.end():end.start()]
    start = re.search(STARTS[number], body, re.M)
    assert start, f'{number}: missing reviewed narrative start'
    body = body[start.start():]
    if number in TAILS:
        tail = re.search(TAILS[number], body, re.M)
        assert tail, f'{number}: expected trailing material absent'
        body = body[:tail.start()]
    original_chapters = re.findall(CHAPTER, body, re.M)
    if number in EXPECTED_CHAPTERS:
        assert len(original_chapters) == EXPECTED_CHAPTERS[number], (number, len(original_chapters))
    named_sections = 0
    if number in {54, 902}:
        normalize = lambda s: re.sub(r'[^\w]+', ' ', s.lower()).strip()
        if number == 54:
            toc = source.split('LIST OF CHAPTERS', 1)[1].split('Tip Manufactures a Pumpkinhead', 1)[0]
            titles = [line.strip() for line in toc.splitlines() if line.strip()]
            titles[0] = 'Tip Manufactures a Pumpkinhead'
            assert len(titles) == 24
        else:
            titles = ['The Happy Prince', 'The Nightingale and the Rose', 'The Selfish Giant',
                      'The Devoted Friend', 'The Remarkable Rocket']
        lines = {normalize(line) for line in body.splitlines()}
        assert all(normalize(title) in lines for title in titles), (number, 'missing named section')
        named_sections = len(titles)
    # Multiline illustration captions are markup, not narrative paragraphs.
    body, illustrations = re.subn(r'\[(?:Illustration|Picture)(?:[^\]]*)\]', '', body, flags=re.I)
    body, page_numbers = re.subn(r'\[(?:Pg\.?\s*)?\d+\]', '', body)
    body = unicodedata.normalize('NFC', body).replace('\ufeff', '').replace('\u00ad', '')
    paragraphs = []
    for block in re.split(r'\n[ \t]*\n+', body):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        # Preserve indented verse, quotations and chapter-title line breaks.
        preserve = any(re.match(r'^[ \t]{2,}\S', line) for line in lines)
        preserve = preserve or bool(re.match(r'^(?:CHAPTER|Chapter|BOOK)\b', lines[0]))
        preserve = preserve or (len(lines) >= 3 and max(map(len, lines)) <= 55)
        if preserve:
            block = '\n'.join(lines)
        else:
            block = re.sub(r'[ \t]+', ' ', ' '.join(line.strip() for line in lines))
        paragraphs.append(block.strip('\n'))
    body = '\n\n'.join(paragraphs).strip()
    assert re.findall(CHAPTER, body, re.M) == original_chapters
    for forbidden in ['project gutenberg', 'pgdp.net', 'transcriber', '\ufffd', 'millennium fulcrum']:
        assert forbidden not in body.lower(), (number, forbidden)
    assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', body)
    assert len(body) > 50000, f'{number}: suspiciously short text'
    text = f'{record["title"]}\nby {record["author"]}\n\n{body}\n'
    return text, dict(chapters=len(original_chapters), named_sections=named_sections, illustration_markers_removed=illustrations,
                      page_markers_removed=page_numbers)


def main():
    records = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    folder = ROOT / 'books'
    folder.mkdir(exist_ok=True)
    texts, hashes = [], set()
    for record in records:
        text, details = clean(record)
        sha = digest(text)
        assert sha not in hashes, 'Duplicate book'
        hashes.add(sha)
        relative = f'books/{record["gutenberg_id"]:05d}_{record["slug"]}.txt'
        write_text(ROOT / relative, text)
        record.update(clean_file=relative, clean_sha256=sha, characters=len(text),
                      utf8_bytes=len(text.encode('utf-8')), words=len(re.findall(r"\b[\w]+(?:['’][\w]+)*\b", text)),
                      cleaning=details,
                      rights_basis='Catalogue explicitly labels this edition Public domain in the USA; '
                      'original English author died no later than 1924. Mainland-China economic-term '
                      'screen: author death + 50 years, end of calendar year; no later translations included.')
        texts.append(text)
        print(f'{record["gutenberg_id"]}: {len(text):,} characters; {details["chapters"]} numbered chapters')
    corpus = '\n\n'.join(texts)
    write_text(ROOT / 'input.txt', corpus)
    # Hold out two complete books. No passages from these books enter train.txt.
    # Series/authors still overlap: this is a learning split, not an independent benchmark.
    validation_ids = {12, 709}
    train = '\n\n'.join(t for r, t in zip(records, texts) if r['gutenberg_id'] not in validation_ids)
    val = '\n\n'.join(t for r, t in zip(records, texts) if r['gutenberg_id'] in validation_ids)
    write_text(ROOT / 'train.txt', train)
    write_text(ROOT / 'val.txt', val)
    write_text(ROOT / 'manifest.json', json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    stats = dict(books=len(records), characters=len(corpus), utf8_bytes=len(corpus.encode('utf-8')),
                 words=sum(r['words'] for r in records), unique_characters=len(set(corpus)),
                 sha256=digest(corpus), train_characters=len(train), val_characters=len(val),
                 validation_book_ids=sorted(validation_ids),
                 val_fraction=len(val) / (len(train) + len(val)),
                 encoding='UTF-8 without BOM', newlines='LF', book_separator='three newline characters',
                 quality_checks=['Strict UTF-8 decoding', 'Raw and cleaned SHA-256',
                                 'Explicit catalogue public-domain marker and author dates',
                                 'Reviewed narrative boundaries and endings',
                                 'Numbered chapter counts and headings preserved',
                                 'No duplicate complete books', 'No Gutenberg boilerplate or replacement characters'])
    write_text(ROOT / 'stats.json', json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    source_lines = ['# 来源与版权记录', '', '核验日期：2026-10-21。仅收录英文原著文字，目录中的电子书发布日期不是原著首次出版年。', '',
                    '|作品|作者及卒年|原著出版年|原始书目与版权标记|', '|---|---|---|---|']
    for r in records:
        source_lines.append(f'|{r["title"]}|{r["author"]}（{r["author_death_year"]}）|{r["first_publication_year"]}|[书目页]({r["catalogue_url"]})：{r["catalogue_copyright"]}|')
    source_lines += ['', '版权判断的范围', '',
                     '- 每本所选电子书的书目页均明确标注美国公版；书目 HTML 快照、含许可的原始 TXT 和 SHA-256 保存在 sources/ 与 manifest.json。',
                     '- 中国大陆筛选依据：[国家版权局《中华人民共和国著作权法》第二十三条](https://www.ncac.gov.cn/xxfb/flfg/flfg_532/202103/t20210309_50530.html)。自然人作品相关财产权通常至作者去世后第五十年的年底；本批作者最晚于 1924 年去世。由这些日期推得本批原著文字已超过该通常期限。第二十二条的人身权保护不受该期限限制，故保留作者与书名。',
                     '- 这里的“公版”不等于在所有法域均无任何权利限制。[Gutenberg 许可](https://www.gutenberg.org/policy/license)与[权限说明](https://www.gutenberg.org/policy/permission)仅直接确认美国状态；其他使用地应按当地规则判断。未纳入现代译文、插图或现代编辑序言。',
                     '- 训练 TXT 去除了 Gutenberg 名称、商标声明及许可正文；这些信息保留在独立来源文件中。',
                     '- 原著出版年为整理者提供的书目背景信息；下载版本及作者日期以所存书目页为准。', '']
    write_text(ROOT / 'SOURCES.md', '\n'.join(source_lines))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
