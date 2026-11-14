import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
with (root/'orchestrator.log').open('a') as log:
    proc=subprocess.Popen([sys.executable,str(root/'orchestrate.py')],stdin=subprocess.DEVNULL,
                          stdout=log,stderr=log,start_new_session=True,cwd=root)
(root/'orchestrator.pid').write_text(str(proc.pid))
print('Orchestrator PID',proc.pid)
