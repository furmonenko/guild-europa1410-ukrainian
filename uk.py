"""Ukrainian localization for The Guild 1 Remake: Europa 1410.

    python uk.py tools       download repak/retoc into tools/
    python uk.py extract     pull the original localization out of the game into work/
    python uk.py check       validate translations/ against the current English source
    python uk.py status      translation coverage per namespace
    python uk.py build       build the mod into dist/
    python uk.py install     build + copy the mod into the game's Paks/~mods
    python uk.py uninstall   remove the mod from the game

Game directory: --game <path>, or the EUROPA1410_DIR environment variable.
"""
import argparse
import collections
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from ukloc.formats import PakReader, patch_locres, read_locres

ROOT = Path(__file__).resolve().parent
WORK, DIST, TOOLS, TRANSLATIONS = ROOT / 'work', ROOT / 'dist', ROOT / 'tools', ROOT / 'translations'
DEFAULT_GAME = r'D:\SteamLibrary\steamapps\common\The Guild - Europa 1410'

MOD_NAME = 'ZZ_Ukrainian_P'
ENGINE_VERSION = 'UE5_6'
# The Ukrainian text replaces this culture until the game gets a real `uk` culture.
TARGET_CULTURE = 'en'
REFERENCE_CULTURES = ['en', 'de']  # de is shown to translators as extra context
LOC_DIR = 'Europa1410/Content/Localization'
TARGETS = ['Game', 'Game_VO', 'Uncategorized Texts']

TOOLS_PINNED = {
    'repak': ('https://github.com/trumank/repak/releases/download/v0.2.3/repak_cli-x86_64-pc-windows-msvc.zip',
              '6720d602144d75df477a99d5bedb6ea780997546afc335901d4937cafeaa73fa'),
    'retoc': ('https://github.com/trumank/retoc/releases/download/v0.1.5/retoc_cli-x86_64-pc-windows-msvc.zip',
              'cc036b06ad3bdcf7003690b00d82719980c374e48a95bf0654f9959148d263aa'),
}


def ns_file(target, namespace):
    return TRANSLATIONS / target / f'{namespace or "_default"}.json'


def ns_from_file(path):
    return '' if path.stem == '_default' else path.stem


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def source_locres(target, culture):
    path = WORK / 'original' / target / culture / f'{target}.locres'
    if not path.exists():
        sys.exit(f'{path} not found - run `python uk.py extract` first')
    return path.read_bytes()


def source_strings(target, culture=TARGET_CULTURE):
    """{namespace: {key: text}}"""
    out = collections.defaultdict(dict)
    for ns, key, text in read_locres(source_locres(target, culture)):
        out[ns][key] = text
    return out


def translation_files():
    return sorted(p for p in TRANSLATIONS.glob('*/*.json'))


def tool(name):
    exe = TOOLS / name / f'{name}.exe'
    if not exe.exists():
        sys.exit(f'{exe} not found - run `python uk.py tools` first')
    return str(exe)


# ---------------------------------------------------------------- commands

def cmd_tools(args):
    for name, (url, sha) in TOOLS_PINNED.items():
        if (TOOLS / name / f'{name}.exe').exists():
            print(f'{name}: already present')
            continue
        data = urllib.request.urlopen(url).read()
        if hashlib.sha256(data).hexdigest() != sha:
            sys.exit(f'{name}: sha256 mismatch, refusing to use {url}')
        zipfile.ZipFile(io.BytesIO(data)).extractall(TOOLS / name)
        print(f'{name}: installed')


def cmd_extract(args):
    pak = PakReader(Path(args.game) / 'Europa1410/Content/Paks/Europa1410-Windows.pak')
    count = 0
    for path in pak.files:
        if path.startswith(LOC_DIR + '/'):
            dst = WORK / 'original' / path[len(LOC_DIR) + 1:]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(pak.read(path))
            count += 1
    print(f'extracted {count} files to {WORK / "original"}')

    # Per-namespace reference files for translators: work/source/<target>/<ns>.json
    for target in TARGETS:
        cultures = {c: source_strings(target, c) for c in REFERENCE_CULTURES
                    if (WORK / 'original' / target / c).exists()}
        for ns, keys in cultures[TARGET_CULTURE].items():
            rows = [{'key': k, **{c: cultures[c].get(ns, {}).get(k, '') for c in cultures}} for k in keys]
            save_json(WORK / 'source' / target / f'{ns or "_default"}.json', rows)
    print(f'reference texts written to {WORK / "source"}')


_TOKEN_RE = re.compile(r'\{[^{}]*\}|<[^<>]*>|\|plural\(\)|\|gender\(\)|\|ordinal\(\)|\r|\n|\[[^\]]+\]')


def markup_tokens(s):
    """Placeholders/tags that must survive translation. Plural/gender argument text is free."""
    s = re.sub(r'\|(plural|gender|ordinal)\([^)]*\)', r'|\1()', s)
    return collections.Counter(_TOKEN_RE.findall(s))


def cmd_check(args):
    errors = warnings = 0
    for path in translation_files():
        target, ns = path.parent.name, ns_from_file(path)
        source = source_strings(target).get(ns)
        rel = path.relative_to(ROOT)
        if source is None:
            print(f'{rel}: namespace not found in the game'); errors += 1
            continue
        for key, entry in load_json(path).items():
            where = f'{rel} :: {key}'
            if key not in source:
                print(f'{where}: key no longer exists in the game'); warnings += 1
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get('uk'), str):
                print(f'{where}: expected {{"en": ..., "uk": ...}}'); errors += 1
                continue
            if entry.get('en') != source[key]:
                print(f'{where}: English source changed, review translation\n'
                      f'   was: {entry.get("en")!r}\n   now: {source[key]!r}'); warnings += 1
            a, b = markup_tokens(source[key]), markup_tokens(entry['uk'])
            if a != b:
                print(f'{where}: markup mismatch, missing {dict(a - b)}, extra {dict(b - a)}'); errors += 1
            for m in re.finditer(r'\|plural\(([^)]*)\)', entry['uk']):
                if 'other=' not in m.group(1):
                    print(f'{where}: plural needs other='); errors += 1
    print(f'{errors} error(s), {warnings} warning(s)')
    return errors


def cmd_status(args):
    total_done = total_all = 0
    for target in TARGETS:
        for ns, keys in sorted(source_strings(target).items()):
            path = ns_file(target, ns)
            done = sum(1 for k in load_json(path) if k in keys) if path.exists() else 0
            total_done += done
            total_all += len(keys)
            if done or args.all:
                print(f'{target:20} {ns or "_default":24} {done:5}/{len(keys):<5} {100 * done // len(keys):3}%')
    print(f'{"total":45} {total_done:5}/{total_all:<5} {100 * total_done // total_all:3}%')


def cmd_build(args):
    if cmd_check(args):
        sys.exit('fix the errors above first')
    by_target = collections.defaultdict(dict)
    for path in translation_files():
        for key, entry in load_json(path).items():
            by_target[path.parent.name][(ns_from_file(path), key)] = entry['uk']

    stage = DIST / MOD_NAME  # repak names the .pak after the input directory
    shutil.rmtree(DIST, ignore_errors=True)
    for target, translations in by_target.items():
        existing = {(ns, k) for ns, k, _ in read_locres(source_locres(target, TARGET_CULTURE))}
        translations = {k: v for k, v in translations.items() if k in existing}  # skip stale keys
        dst = stage / LOC_DIR / target / TARGET_CULTURE / f'{target}.locres'
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(patch_locres(source_locres(target, TARGET_CULTURE), translations))
        print(f'{target}: {len(translations)} strings')

    pak = DIST / f'{MOD_NAME}.pak'
    subprocess.run([tool('repak'), 'pack', '--quiet', '--version', 'V11', str(stage)], check=True)
    # UE5 skips a .pak that has no IoStore pair next to it, so ship an empty .utoc/.ucas.
    empty = DIST / 'iostore'
    empty.mkdir()
    subprocess.run([tool('retoc'), 'to-zen', '--version', ENGINE_VERSION, str(pak),
                    str(empty / f'{MOD_NAME}.utoc')], check=True, stdout=subprocess.DEVNULL)
    for ext in ('utoc', 'ucas'):
        shutil.move(empty / f'{MOD_NAME}.{ext}', DIST / f'{MOD_NAME}.{ext}')
    shutil.rmtree(empty)
    shutil.rmtree(stage)
    print(f'built {DIST}')


def mods_dir(args):
    return Path(args.game) / 'Europa1410/Content/Paks/~mods'


def cmd_install(args):
    cmd_build(args)
    dst = mods_dir(args)
    dst.mkdir(exist_ok=True)
    for ext in ('pak', 'utoc', 'ucas'):
        shutil.copy(DIST / f'{MOD_NAME}.{ext}', dst)
    print(f'installed into {dst}')


def cmd_uninstall(args):
    for ext in ('pak', 'utoc', 'ucas'):
        p = mods_dir(args) / f'{MOD_NAME}.{ext}'
        if p.exists():
            p.unlink()
            print(f'removed {p}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--game', default=os.environ.get('EUROPA1410_DIR', DEFAULT_GAME))
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('tools', 'extract', 'check', 'build', 'install', 'uninstall'):
        sub.add_parser(name)
    sub.add_parser('status').add_argument('--all', action='store_true', help='include untranslated namespaces')
    args = parser.parse_args()
    result = globals()[f'cmd_{args.cmd}'](args)
    sys.exit(1 if args.cmd == 'check' and result else 0)


if __name__ == '__main__':
    main()
