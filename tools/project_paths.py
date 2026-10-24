"""Resolve original author filenames directly inside the supplied Curated layout.

No files are renamed, copied or assembled by this module.
"""
import csv
import hashlib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

class ProjectLayout:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not (self.root / 'classification').is_dir() and (self.root / 'Curated' / 'classification').is_dir():
            self.root = self.root / 'Curated'
        with (ROOT / 'metadata' / 'FILE_MAP.csv').open(encoding='utf-8', newline='') as f:
            self.rows = list(csv.DictReader(f))
        self.files = {r['original_project_path']: r['package_path'] for r in self.rows}
        candidates = {}
        for old, new in self.files.items():
            op, np = PurePosixPath(old).parts, PurePosixPath(new).parts
            for count in range(1, len(op)):
                suffix = op[count:]
                if tuple(np[-len(suffix):]) != tuple(suffix):
                    continue
                candidates.setdefault('/'.join(op[:count]), set()).add('/'.join(np[:-len(suffix)]))
        self.directories = {k: next(iter(v)) for k, v in candidates.items() if len(v) == 1}
        self.directories.update({'': 'classification/input', 'TG': 'regression/input',
                                 'TG/blind2': 'regression/blind_2'})

    def path(self, original_relative):
        relative = str(original_relative).replace('\\', '/').strip('/')
        p = PurePosixPath(relative)
        if '..' in p.parts or ':' in relative or str(original_relative).startswith(('/', '\\')):
            raise ValueError('Expected a relative project path: ' + str(original_relative))
        if relative in self.files:
            return self.root / self.files[relative]
        if relative in self.directories:
            return self.root / self.directories[relative]
        for prefix in sorted(self.directories, key=len, reverse=True):
            if prefix and relative.startswith(prefix + '/'):
                return self.root / self.directories[prefix] / relative[len(prefix)+1:]
        # New original-named output files not present in the archived snapshot.
        if relative.startswith(('classification/', 'regression/', 'CrossTask_Family_Multisplit_RF/', 'Manuscript_Figures/')):
            return self.root / relative
        tg = relative.startswith('TG/')
        name = relative[3:] if tg else relative
        endpoint = 'regression' if tg else 'classification'
        if name.startswith('master_'):
            return self.root / endpoint / 'encoded' / name
        return self.root / endpoint / 'result' / name

    def validate(self, hashes=False):
        missing, changed = [], []
        for row in self.rows:
            p = self.root / row['package_path']
            if not p.is_file():
                missing.append(row['package_path'])
                continue
            if hashes:
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != row['sha256']:
                    changed.append(row['package_path'])
        return {'canonical_files': len(self.rows), 'missing': missing, 'hash_mismatches': changed,
                'hashes_checked': hashes, 'data_root': str(self.root),
                'passed': not missing and not changed}

def bind_paths(root):
    return ProjectLayout(root).path
