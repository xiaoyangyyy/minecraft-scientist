"""Install a verified, pinned upstream engine archive into the persistent runtime."""
import hashlib
from pathlib import Path
import zipfile

ARCHIVE = Path('/mnt/d/minecraft-scientist/runs/installers/minestudio-engine.zip')
DESTINATION = Path('/opt/minecraft-scientist-engine-verified')
EXPECTED = '293fac6ac72245b3365dce0e8bfbb6396fb94df29b23b6538f3bd7e2eec13ec6'


def main():
    digest = hashlib.sha256()
    with ARCHIVE.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED:
        raise RuntimeError('Engine archive SHA256 mismatch')
    with zipfile.ZipFile(ARCHIVE) as archive:
        root = DESTINATION.resolve()
        for entry in archive.infolist():
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError('Unsafe archive path')
        archive.extractall(root)
    jar = DESTINATION / 'engine/build/libs/mcprec-6.13.jar'
    if not jar.is_file():
        raise RuntimeError('Expected engine JAR missing')
    print(f'Verified engine installed: {jar}')


if __name__ == '__main__':
    main()
