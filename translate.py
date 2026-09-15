"""LLM translation driver: one-shot headless `claude -p` calls, no tools, cost-capped.

    python translate.py terms                     collect inline term codes ({XXX_C}...{##}) into work/terms/codes_en.json
    python translate.py run Game/Items [...]      translate a namespace (chunked, validated, merged into translations/)
    python translate.py review Game/Items [...]   second-model review -> work/review/<ns>.json (issues only)
    python translate.py apply Game/Items          apply fixes from work/review/<ns>.json (edit the file first if needed)
    python translate.py log                       cost log summary (per day / per namespace)

Common options: --model sonnet|opus|<id>  --effort low|medium|high  --max-cost USD  --workers N  --chunk N  --dry-run
Every call's usage and total_cost_usd go to work/log/calls.jsonl; --max-cost stops the run when the sum is reached,
the next run resumes with whatever is still untranslated.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from uk import ROOT, WORK, TRANSLATIONS, load_json, save_json, source_strings, markup_tokens, ns_file, TARGETS

DOCS = ROOT / 'docs'
LOG = WORK / 'log' / 'calls.jsonl'
TERMS = ROOT / 'termcodes.json'
CLAUDE = shutil.which('claude') or shutil.which('claude.cmd')

MODELS = {'sonnet': 'sonnet', 'opus': 'opus', 'haiku': 'haiku'}
SKIP_RE = re.compile(r'\(LocMe\)|^\s*TBD\s*$')
LETTERS_RE = re.compile(r'[A-Za-z]{2,}')
CYR_RE = re.compile(r'[А-Яа-яЇїІіЄєҐґ]')

_lock = threading.Lock()


# ---------------------------------------------------------------- helpers

def split_ns(arg):
    """'Game/Items' -> ('Game', 'Items'); 'Uncategorized Texts/_default' -> ('Uncategorized Texts', '')"""
    target, _, ns = arg.partition('/')
    if target not in TARGETS:
        sys.exit(f'unknown target {target!r}, expected one of {TARGETS}')
    return target, ('' if ns == '_default' else ns)


def source_rows(target, ns):
    path = WORK / 'source' / target / f'{ns or "_default"}.json'
    if not path.exists():
        sys.exit(f'{path} not found - run `python uk.py extract` first')
    return load_json(path)


UI_NAMESPACES = {'Settings', 'Controls', 'General'}


def is_ui(path):
    """Menu/settings namespaces and widget texts: short strings there mean something else in-game."""
    return path.parent.name == 'Uncategorized Texts' or path.stem in UI_NAMESPACES


def translation_memory():
    """en text -> uk from everything already translated (any namespace)."""
    tm = {}
    for path in sorted(TRANSLATIONS.glob('*/*.json')):
        ui = is_ui(path)
        for entry in load_json(path).values():
            if isinstance(entry, dict) and entry.get('uk') and entry.get('en'):
                en = entry['en'].strip()
                if ui and len(en.split()) < 3:
                    continue    # "Master" in Settings is master volume, not a guild master
                tm.setdefault(en, entry['uk'])
    return tm


def short_terms():
    """Already translated short strings (names of items, buildings, ...) for reuse inside longer texts."""
    out = {}
    for path in sorted(TRANSLATIONS.glob('*/*.json')):
        if is_ui(path):
            continue
        for key, entry in load_json(path).items():
            en = (entry.get('en') or '').strip()
            if 3 <= len(en) <= 40 and not any(c in en for c in '{}<>\n') and entry.get('uk'):
                out.setdefault(en, entry['uk'])
    return out


def relevant_terms(rows, terms):
    """Subset of translated short strings whose English form occurs inside this chunk's texts."""
    text = '\n'.join(r['en'] for r in rows).lower()
    return {en: uk for en, uk in terms.items() if len(en) >= 4 and en.lower() in text}


def read_text(path):
    return Path(path).read_text(encoding='utf-8')


def system_prompt(kind):
    parts = [read_text(DOCS / 'translator-brief.md'), '\n\n# Глосарій\n', read_text(ROOT / 'glossary.md')]
    if TERMS.exists():
        parts += ['\n\n# Коди інлайн-термінів (код → канонічний український термін; відмінюйте за контекстом)\n',
                  '\n'.join(f'{code}: {v["uk"]}' for code, v in load_json(TERMS).items())]
    parts.append(read_text(DOCS / f'prompt-{kind}.md'))
    text = ''.join(parts)
    path = WORK / 'tmp' / f'system-{kind}-{hashlib.sha1(text.encode()).hexdigest()[:8]}.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding='utf-8')
    return path


def call_claude(system_file, prompt, payload, schema, model, effort, tag):
    """One headless call. Returns (structured_output or None, cost_usd)."""
    if not CLAUDE:
        sys.exit('`claude` CLI not found in PATH')
    cmd = [CLAUDE, '-p', '--model', MODELS.get(model, model), '--tools', '', '--effort', effort,
           '--output-format', 'json', '--max-turns', '1', '--system-prompt-file', str(system_file),
           '--json-schema', json.dumps(schema), prompt]
    # 5-minute cache: the shared system prompt is re-read within minutes anyway, and the per-call payload
    # (never re-read) is written at 1.25x instead of the 1-hour tier's 2x.
    env = dict(os.environ, PYTHONIOENCODING='utf-8', CLAUDE_CODE_PROMPT_CACHE_TTL='5m')
    proc = subprocess.run(cmd, input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                          encoding='utf-8', errors='replace', env=env, cwd=ROOT)
    rec = {'time': dt.datetime.now().isoformat(timespec='seconds'), 'tag': tag, 'model': model, 'effort': effort}
    try:
        out = json.loads(proc.stdout)
        rec.update(cost=out.get('total_cost_usd', 0), usage=out.get('usage'), error=out.get('is_error'))
        result = out.get('structured_output')
    except json.JSONDecodeError:
        rec.update(cost=0, error=(proc.stderr or proc.stdout)[-500:])
        result = None
    with _lock:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    if result is None:
        print(f'  ! {tag}: no structured output ({rec.get("error")})', file=sys.stderr)
    return result, rec.get('cost') or 0


class Budget:
    def __init__(self, max_cost):
        self.max_cost, self.spent = max_cost, 0.0

    def add(self, cost):
        with _lock:
            self.spent += cost

    def exhausted(self):
        return self.max_cost is not None and self.spent >= self.max_cost


# ---------------------------------------------------------------- validation

def validate(en, uk):
    """Return None if ok, else a short problem description for the retry prompt."""
    if not isinstance(uk, str) or not uk.strip():
        return 'empty'
    if uk.strip() == en.strip():
        return 'not translated (identical to English)'
    a, b = markup_tokens(en), markup_tokens(uk)
    if a != b:
        return f'markup mismatch: missing {dict(a - b)}, extra {dict(b - a)}'
    for m in re.finditer(r'\|plural\(([^)]*)\)', uk):
        if 'other=' not in m.group(1):
            return 'plural needs other='
    if LETTERS_RE.search(en) and not CYR_RE.search(uk) and not re.fullmatch(r'[\W\d_]*(\{[^}]*\}[\W\d_]*)*', en):
        return 'no Cyrillic letters in translation'
    if en.startswith(('\n', '\r\n')) != uk.startswith(('\n', '\r\n')):
        return 'leading newline changed'
    return None


# ---------------------------------------------------------------- commands

def cmd_terms(args):
    codes = collections.defaultdict(lambda: {'en': collections.Counter(), 'examples': []})
    for target in TARGETS:
        for ns_rows in (WORK / 'source' / target).glob('*.json'):
            for r in load_json(ns_rows):
                for m in re.finditer(r'\{([A-Z][A-Z0-9_]*_C)\}(.*?)\{##\}', r['en']):
                    c = codes[m.group(1)]
                    c['en'][m.group(2)] += 1
                    if len(c['examples']) < 2:
                        c['examples'].append(r['en'][:160])
    out = {code: {'en': [t for t, _ in v['en'].most_common(6)], 'examples': v['examples']}
           for code, v in sorted(codes.items())}
    dst = WORK / 'terms' / 'codes_en.json'
    save_json(dst, out)
    print(f'{len(out)} term codes -> {dst}')
    if not args.translate:
        return
    existing = load_json(TERMS) if TERMS.exists() else {}
    new = {c: v for c, v in out.items() if c not in existing}
    if not new:
        print('termcodes.json is up to date'); return
    schema = {'type': 'object', 'properties': {'terms': {'type': 'object', 'additionalProperties': {'type': 'string'}}},
              'required': ['terms'], 'additionalProperties': False}
    prompt = ('For every inline term code on stdin choose ONE canonical Ukrainian term (nominative, lowercase unless a '
              'proper noun) that will be used consistently for that code across the whole game; keep the glossary. '
              'Related codes must get related terms (skill vs skill training). Return {"terms": {"<code>": "<term>"}}.')
    result, cost = call_claude(system_prompt('translate'), prompt, new, schema, args.model, args.effort, 'termcodes')
    terms = (result or {}).get('terms', {})
    for code, v in new.items():
        if terms.get(code):
            existing[code] = {'uk': terms[code], 'en': v['en']}
    save_json(TERMS, dict(sorted(existing.items())))
    print(f'{len(terms)} terms translated (${cost:.2f}) -> {TERMS}; review it by hand before translating texts')


def cmd_run(args):
    target, ns = split_ns(args.namespace)
    rows = source_rows(target, ns)
    path = ns_file(target, ns)
    done = load_json(path) if path.exists() else {}
    tm = translation_memory()
    terms = short_terms()

    todo, reused = [], 0
    for r in rows:
        if r['key'] in done and done[r['key']].get('en') == r['en']:
            continue
        en = r['en']
        if SKIP_RE.search(en) or not LETTERS_RE.search(re.sub(r'\{[^}]*\}', '', en)):
            done[r['key']] = {'en': en, 'uk': en}          # technical / placeholder rows stay as they are
        elif en.strip() in tm:
            done[r['key']] = {'en': en, 'uk': tm[en.strip()]}; reused += 1
        else:
            todo.append(r)
    # translate each distinct English text once
    by_text = collections.OrderedDict()
    for r in sorted(todo, key=lambda r: r['key']):
        by_text.setdefault(r['en'].strip(), []).append(r)
    units = [{'id': str(i), 'key': rs[0]['key'], 'en': rs[0]['en'], 'de': rs[0].get('de', ''), '_rows': rs}
             for i, rs in enumerate(by_text.values(), 1)]
    print(f'{args.namespace}: {len(rows)} rows, {len(done)} already done/reused ({reused} from memory), '
          f'{len(todo)} to translate = {len(units)} distinct texts')
    if args.dry_run or not units:
        save_json(path, dict(sorted(done.items())))
        return

    chunks = [units[i:i + args.chunk] for i in range(0, len(units), args.chunk)]
    sysfile = system_prompt('translate')
    schema = {'type': 'object', 'properties': {'t': {'type': 'object', 'additionalProperties': {'type': 'string'}}},
              'required': ['t'], 'additionalProperties': False}
    budget = Budget(args.max_cost)
    failed = []

    def work(chunk, attempt):
        payload = {'namespace': args.namespace,
                   'known_terms': relevant_terms(chunk, terms),
                   'rows': [{k: u[k] for k in ('id', 'key', 'en', 'de') if u.get(k) != ''} | ({'problem': u['_problem']} if u.get('_problem') else {})
                            for u in chunk]}
        prompt = ('Translate the "en" field of every row in the JSON on stdin into Ukrainian. '
                  'Return {"t": {"<id>": "<Ukrainian text>"}} with exactly one entry per row id, nothing else.')
        tag = f'{args.namespace} chunk {chunk[0]["id"]}-{chunk[-1]["id"]} try{attempt}'
        result, cost = call_claude(sysfile, prompt, payload, schema, args.model, args.effort, tag)
        budget.add(cost)
        ok = bad = 0
        got = (result or {}).get('t', {})
        with _lock:
            for u in chunk:
                uk = got.get(u['id'])
                problem = validate(u['en'], uk) if uk is not None else 'missing'
                if problem:
                    u['_problem'] = problem; failed.append(u); bad += 1
                else:
                    for r in u['_rows']:
                        done[r['key']] = {'en': r['en'], 'uk': uk}
                    ok += 1
            save_json(path, dict(sorted(done.items())))
        print(f'  {tag}: {ok} ok, {bad} to retry, ${cost:.3f} (run total ${budget.spent:.2f})')

    def run_chunks(chunk_list, attempt):
        pending, queue = set(), list(chunk_list)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            if queue and not budget.exhausted():
                work(queue.pop(0), attempt)  # first call alone: warms the system-prompt cache for the parallel ones
            while queue or pending:
                while queue and len(pending) < args.workers and not budget.exhausted():
                    pending.add(pool.submit(work, queue.pop(0), attempt))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in finished:
                    f.result()
        return queue  # chunks not started because of the budget

    left = run_chunks(chunks, 1)
    if failed and not left and not budget.exhausted():
        retry, failed = failed, []
        print(f'retrying {len(retry)} rows in small chunks')
        left = run_chunks([retry[i:i + 20] for i in range(0, len(retry), 20)], 2)
    if left:
        print(f'budget ${args.max_cost} reached: {sum(len(c) for c in left)} texts not started, rerun to continue')
    if failed:
        print(f'{len(failed)} texts still failing validation:')
        for u in failed:
            print(f'  {u["key"]}: {u["_problem"]}')
    print(f'spent ${budget.spent:.2f}; now run `python uk.py check`')


def cmd_review(args):
    target, ns = split_ns(args.namespace)
    path = ns_file(target, ns)
    entries = load_json(path)
    units = [{'id': str(i), 'key': k, 'en': e['en'], 'uk': e['uk']}
             for i, (k, e) in enumerate(sorted(entries.items()), 1)
             if e['uk'] != e['en'] and not SKIP_RE.search(e['en'])]
    print(f'{args.namespace}: reviewing {len(units)} translated rows')
    if args.dry_run:
        return
    chunks = [units[i:i + args.chunk] for i in range(0, len(units), args.chunk)]
    sysfile = system_prompt('review')
    schema = {'type': 'object', 'properties': {'issues': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'id': {'type': 'string'}, 'category': {'type': 'string'},
                                         'problem': {'type': 'string'}, 'fix': {'type': 'string'}},
        'required': ['id', 'category', 'problem', 'fix'], 'additionalProperties': False}}},
        'required': ['issues'], 'additionalProperties': False}
    budget = Budget(args.max_cost)
    issues = []

    def work(chunk):
        prompt = ('Review the Ukrainian translations ("uk") of the rows on stdin against "en". '
                  'Return {"issues": [...]} listing ONLY rows that need a change, with the corrected full text in "fix". '
                  'Do not list rows that are fine.')
        tag = f'review {args.namespace} {chunk[0]["id"]}-{chunk[-1]["id"]}'
        result, cost = call_claude(sysfile, prompt, {'namespace': args.namespace, 'rows': chunk}, schema,
                                   args.model, args.effort, tag)
        budget.add(cost)
        by_id = {u['id']: u for u in chunk}
        found = [{**i, 'key': by_id[i['id']]['key'], 'en': by_id[i['id']]['en'], 'uk': by_id[i['id']]['uk']}
                 for i in (result or {}).get('issues', []) if i.get('id') in by_id]
        with _lock:
            issues.extend(found)
        print(f'  {tag}: {len(found)} issues, ${cost:.3f} (run total ${budget.spent:.2f})')

    if chunks and not budget.exhausted():
        work(chunks.pop(0))  # warm the cache before going parallel
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for c in chunks:
            if budget.exhausted():
                print('budget reached, stopping'); break
            futures.append(pool.submit(work, c))
            if len(futures) >= args.workers:
                wait(futures[:1]); futures.pop(0)
        for f in futures:
            f.result()
    dst = WORK / 'review' / target / f'{ns or "_default"}.json'
    save_json(dst, sorted(issues, key=lambda i: i['key']))
    counts = collections.Counter(i['category'] for i in issues)
    print(f'{len(issues)} issues -> {dst}  {dict(counts)}\nspent ${budget.spent:.2f}')


def cmd_apply(args):
    target, ns = split_ns(args.namespace)
    src = WORK / 'review' / target / f'{ns or "_default"}.json'
    path = ns_file(target, ns)
    entries, applied, rejected = load_json(path), 0, 0
    for issue in load_json(src):
        e = entries.get(issue['key'])
        if not e or e['uk'] != issue['uk']:
            continue  # already changed by hand
        problem = validate(e['en'], issue['fix'])
        if problem:
            print(f'  skip {issue["key"]}: fix fails validation ({problem})'); rejected += 1
            continue
        e['uk'] = issue['fix']; applied += 1
    save_json(path, entries)
    print(f'applied {applied}, rejected {rejected}; now run `python uk.py check`')


def cmd_log(args):
    if not LOG.exists():
        print('no calls logged yet'); return
    per_day, per_ns, total = collections.Counter(), collections.Counter(), 0.0
    for line in LOG.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        cost = r.get('cost') or 0
        per_day[r['time'][:10]] += cost
        per_ns[re.sub(r' (chunk )?\d+-\d+( try\d)?$', '', r['tag']).replace('review ', 'review:')] += cost
        total += cost
    print('per day:'); [print(f'  {d}  ${c:.2f}') for d, c in sorted(per_day.items())]
    print('per namespace:'); [print(f'  {n:40s} ${c:.2f}') for n, c in sorted(per_ns.items())]
    print(f'total ${total:.2f}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('terms')
    t.add_argument('--translate', action='store_true', help='also fill termcodes.json for codes not yet there')
    t.add_argument('--model', default='opus'); t.add_argument('--effort', default='medium')
    sub.add_parser('log')
    for name, model, effort, chunk in (('run', 'sonnet', 'low', 120), ('review', 'opus', 'medium', 150)):
        s = sub.add_parser(name)
        s.add_argument('namespace')
        s.add_argument('--model', default=model)
        s.add_argument('--effort', default=effort)
        s.add_argument('--chunk', type=int, default=chunk)
        s.add_argument('--workers', type=int, default=3)
        s.add_argument('--max-cost', type=float, default=None, help='stop when the run has spent this many USD')
        s.add_argument('--dry-run', action='store_true')
    sub.add_parser('apply').add_argument('namespace')
    args = p.parse_args()
    globals()[f'cmd_{args.cmd}'](args)


if __name__ == '__main__':
    main()
