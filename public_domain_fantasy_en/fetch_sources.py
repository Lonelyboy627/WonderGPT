"""Download the fixed, audited reading list. Python 3; standard library only."""
import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOKS = [
    (11, 'alice_in_wonderland', 'Lewis Carroll', 1898, 1865),
    (12, 'through_the_looking_glass', 'Lewis Carroll', 1898, 1871),
    (55, 'wonderful_wizard_of_oz', 'L. Frank Baum', 1919, 1900),
    (54, 'marvelous_land_of_oz', 'L. Frank Baum', 1919, 1904),
    (708, 'princess_and_the_goblin', 'George MacDonald', 1905, 1872),
    (709, 'princess_and_curdie', 'George MacDonald', 1905, 1883),
    (325, 'phantastes', 'George MacDonald', 1905, 1858),
    (225, 'at_the_back_of_the_north_wind', 'George MacDonald', 1905, 1871),
    (1640, 'lilith', 'George MacDonald', 1905, 1895),
    (778, 'five_children_and_it', 'E. Nesbit', 1924, 1902),
    (836, 'phoenix_and_the_carpet', 'E. Nesbit', 1924, 1904),
    (837, 'story_of_the_amulet', 'E. Nesbit', 1924, 1906),
    (34219, 'enchanted_castle', 'E. Nesbit', 1924, 1907),
    (169, 'well_at_the_worlds_end', 'William Morris', 1896, 1896),
    (3055, 'wood_beyond_the_world', 'William Morris', 1896, 1894),
    (902, 'happy_prince_and_other_tales', 'Oscar Wilde', 1900, 1888),
]


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'PublicDomainFantasyCorpus/1.0 (personal research; 16 books)'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def plain(fragment):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]*>', ' ', fragment))).strip()


def main():
    folder = ROOT / 'sources'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for number, slug, author, death, publication in BOOKS:
        landing = f'https://www.gutenberg.org/ebooks/{number}'
        text_url = f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt'
        page_path = folder / f'pg{number}.html'
        text_path = folder / f'pg{number}.txt'
        if not page_path.exists():
            page_path.write_bytes(fetch(landing))
        page = page_path.read_text(encoding='utf-8')
        fields = {}
        for row in re.findall(r'<tr\b.*?</tr>', page, flags=re.S | re.I):
            cells = re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row, flags=re.S | re.I)
            if len(cells) >= 2:
                fields[plain(cells[0])] = plain(cells[1])
        if 'Public domain in the USA' not in fields.get('Copyright Status', fields.get('Copyright', '')):
            raise ValueError(f'{number}: no explicit public-domain status: {fields}')
        if fields.get('Language') != 'English':
            raise ValueError(f'{number}: unexpected language')
        if str(death) not in fields.get('Author', ''):
            raise ValueError(f'{number}: author dates do not match: {fields.get("Author")}')
        if not text_path.exists():
            text_path.write_bytes(fetch(text_url))
        raw = text_path.read_bytes()
        decoded = raw.decode('utf-8-sig')
        if '*** START OF' not in decoded or '*** END OF' not in decoded:
            raise ValueError(f'{number}: missing ebook boundary markers')
        record = dict(gutenberg_id=number, slug=slug, title=fields.get('Title'),
                      author=author, author_catalogue=fields.get('Author'), author_death_year=death,
                      first_publication_year=publication, language='en',
                      catalogue_copyright=fields.get('Copyright Status', fields.get('Copyright')),
                      catalogue_url=landing, download_url=text_url,
                      retrieved_date='2026-10-21', raw_sha256=hashlib.sha256(raw).hexdigest(),
                      raw_bytes=len(raw), raw_file=f'sources/pg{number}.txt',
                      catalogue_snapshot=f'sources/pg{number}.html',
                      catalogue_fields=fields)
        records.append(record)
        print(f'{number}: {record["title"]} | {len(raw):,} bytes | {record["author_catalogue"]}', flush=True)
    (ROOT / 'source_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
