#!/usr/bin/env python3
"""Run in the actual paper environment; record versions without inventing a lock."""
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    packages=['numpy','pandas','scipy','scikit-learn','statsmodels','xgboost','matplotlib','seaborn','umap-learn','joblib','shap']
    versions={}
    for name in packages:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    record={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'os':platform.system(),'os_release':platform.release(),'machine':platform.machine(),'logical_cpus':os.cpu_count(),'packages':versions,'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    result=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('pip freeze failed; JSON was saved but no lock is claimed')
    args.out.with_suffix('.pip-freeze.txt').write_text(result.stdout)
    print(args.out)

if __name__ == '__main__':main()
