import json
from pathlib import Path
from subprocess import check_output
from typing import Optional


def get_kobo_mountpoint(label: str='KOBOeReader') -> Optional[Path]:
    xxx = check_output(['lsblk', '-f', '--json']).decode('utf8')
    jj = json.loads(xxx)
    kobos = [d for d in jj['blockdevices'] if d.get('label', None) == label]
    if len(kobos) > 1:
        raise RuntimeError(f'Multiple Kobo devices detected: {kobos}')
    elif len(kobos) == 0:
        return None
    else:
        [kobo] = kobos
        return Path(kobo['mountpoint'])

