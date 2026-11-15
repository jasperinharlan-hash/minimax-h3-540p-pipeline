"""Download the frozen public H3 weights, reusing verified existing files.

Run --list first. This script neither starts ComfyUI nor submits render jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=Path('/kaggle/temp/h3_shared_models'))
    parser.add_argument('--search-root', action='append', type=Path, default=[])
    parser.add_argument('--profile', choices=['reference', 'all'], default='all')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('public_models.json').read_text(encoding='utf-8'))
    items = [x for x in manifest['files'] if args.profile == 'all' or x['profile'] in ('shared', 'reference')]
    total = sum(x['bytes'] for x in items)
    print(f'{len(items)} public files: {total / 1e9:.3f} GB / {total / 2**30:.3f} GiB')
    for item in items:
        print(f"{item['path']}  {item['bytes'] / 1e9:.3f} GB")
    if args.list:
        return
    from huggingface_hub import hf_hub_download
    roots = [p for p in args.search_root if p.exists()]
    names = {Path(x['path']).name for x in items}
    candidates = {}
    for root in roots:
        for p in root.rglob('*.safetensors'):
            if p.name in names:
                candidates.setdefault(p.name, []).append(p)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        target = args.models_dir / item['path']
        if target.exists():
            if verify(target, item):
                print('[verified existing]', target)
                continue
            raise RuntimeError(f'Existing target does not match the frozen public weight; preserved: {target}')
        reused = False
        for candidate in candidates.get(target.name, []):
            if verify(candidate, item):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(candidate.resolve())
                print('[reused verified file]', target)
                reused = True
                break
        if reused:
            continue
        print('[download]', item['path'], flush=True)
        # The frozen official repository is public; no personal token is needed.
        got = Path(hf_hub_download(repo_id=manifest['repo'], revision=manifest['revision'],
                                  filename=item['path'], local_dir=args.models_dir, token=False))
        if not verify(got, item):
            raise RuntimeError(f'Download SHA256 mismatch; file retained for diagnosis: {got}')
        print('[verified download]', got)
    print('Public weights ready. No render was submitted.')


if __name__ == '__main__':
    main()
