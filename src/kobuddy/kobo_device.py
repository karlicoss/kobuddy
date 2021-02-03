import json
from pathlib import Path
import subprocess
from typing import Optional


def get_kobo_mountpoint(label: str='KOBOeReader') -> Optional[Path]:
    try:  # Linux
        xxx = subprocess.check_output(['lsblk', '-f', '--json']).decode('utf8')
        jj = json.loads(xxx)
        kobos = [d for d in jj['blockdevices'] if d.get('label', None) == label]
        kobos = [k['mountpoint'] for k in kobos]
    except FileNotFoundError:  # macOS (does not have lsblk)
        output = subprocess.check_output(('df', '-Hl')).decode('utf8')
        output = [o.split() for o in output.split('\n')]
        kobos = [o[-1] for o in output if f'/Volumes/{label}' in o]

    if len(kobos) > 1:
        raise RuntimeError(f'Multiple Kobo devices detected: {kobos}')
    elif len(kobos) == 0:
        return None
    else:
        [kobo] = kobos
        return Path(kobo)
