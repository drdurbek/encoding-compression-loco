#!/usr/bin/env python3
"""Run original analyses against the supplied Curated directory layout."""
import argparse
import json
from pathlib import Path
import sys
from project_paths import ProjectLayout

ROOT=Path(__file__).resolve().parents[1]
DEPENDENCIES={**{f'classification_{n:02d}':['classification_06'] for n in [8,9,10,20,21,22]},
              'classification_17':['classification_16']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true')
    p.add_argument('--analysis',nargs='+',default=[])
    p.add_argument('--data-root',type=Path)
    p.add_argument('--dry-run',action='store_true',help='Compile only; do not train, predict, render or write analysis outputs.')
    p.add_argument('--check-data',action='store_true',help='Check canonical package files; does not execute analyses.')
    p.add_argument('--check-hashes',action='store_true',help='Also check archived data SHA-256 values; use before reruns that overwrite outputs.')
    a=p.parse_args()
    rows=json.loads((ROOT/'metadata/source_inventory.json').read_text())['scripts'];lookup={r['id']:r for r in rows}
    if a.list:
        for r in rows:
            dep=', '.join(DEPENDENCIES.get(r['id'],[])) or 'none'
            print(f"{r['id']:20} {r['file']} | prior stages in same command: {dep}")
        if not a.analysis and not a.check_data and not a.check_hashes:return
    if a.data_root is None:p.error('--data-root must point to your Curated folder')
    layout=ProjectLayout(a.data_root)
    if not layout.root.is_dir():p.error('Curated data directory does not exist: '+str(layout.root))
    if a.check_data or a.check_hashes:
        report=layout.validate(hashes=a.check_hashes);print(json.dumps(report,indent=2))
        if not report['passed']:raise SystemExit(1)
        if not a.analysis:return
    if not a.analysis:p.error('supply --analysis IDs, --list or --check-data')
    unknown=[v for v in a.analysis if v not in lookup]
    if unknown:p.error('unknown or excluded analysis IDs: '+', '.join(unknown))
    # Never run an implicit model fit merely because a plotting cell needs state.
    if not a.dry_run:
        seen=set()
        for name in a.analysis:
            missing=set(DEPENDENCIES.get(name,[]))-seen
            if missing:p.error(name+' requires earlier IDs in the same command: '+', '.join(sorted(missing)))
            seen.add(name)
    namespace={'__name__':'__main__','__builtins__':__builtins__,'data_path':layout.path}
    argv=sys.argv[:]
    try:
        for name in a.analysis:
            r=lookup[name];path=ROOT/r['file'];source=path.read_text(encoding='utf-8');code=compile(source,str(path),'exec')
            print(name+': compiled; filenames preserved; Curated paths configured')
            if a.dry_run:continue
            namespace['__file__']=str(path);sys.argv=[str(path)];exec(code,namespace,namespace)
    finally:sys.argv=argv

if __name__=='__main__':main()
