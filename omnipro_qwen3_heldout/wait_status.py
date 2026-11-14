import sys,time,json
from pathlib import Path
ROOT=Path('/iopsstor/scratch/cscs/dbartaula/querystream_omnipro_20261114')
sys.path.insert(0,str(ROOT))
from control import status
for i in range(int(sys.argv[1]) if len(sys.argv)>1 else 4):
    s=status(); print(json.dumps(s),flush=True)
    if s['final']: break
    time.sleep(int(sys.argv[2]) if len(sys.argv)>2 else 30)
