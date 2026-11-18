"""Verify archive checksums, manifest coverage and every packaged file."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def digest_file(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(path):
    expected = path.with_suffix(path.suffix + '.sha256').read_text().split()[0]
    actual = digest_file(path)
    assert actual == expected, f'Archive checksum mismatch: {path}'
    with ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate archive members'
        for name in names:
            p = PurePosixPath(name)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
            assert ':' not in name, f'Nonportable archive path: {name}'
        manifest = json.loads(z.read('MANIFEST.json'))
        listed = [row['path'] for row in manifest]
        assert len(listed) == len(set(listed)), 'Duplicate manifest records'
        extra = set(names) - set(listed)
        assert extra <= {'MANIFEST.json', 'READ_FIRST_SOURCE_PACKAGE.md'}, extra
        assert not set(listed) - set(names), 'Missing manifest files'
        for item in manifest:
            data = z.read(item['path'])
            assert len(data) == item['size'], item['path']
            assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
        required = {'REPRODUCIBILITY.md', 'MATCONVNET_SETUP.md',
                    'Run_thesis_experiments.m', 'Review_thesis_experiments.m',
                    'Verify_thesis_review.m', 'Test_thesis_experiments.m',
                    'Setup_thesis_matconvnet.m', 'experiments/thesis_config.m',
                    'experiments/setup/cudnn8_compat.hpp', 'Install_thesis_bm3d.ps1'}
        assert required <= set(names), required - set(names)
        assert not any('third_party/bm3d/' in n or n.endswith('BM3D.zip') for n in names)
        assert not any(n.endswith(('.dll', '.docx')) for n in names)
    return {'archive': str(path), 'sha256': actual, 'members': len(names),
            'manifest_files': len(manifest), 'all_hashes_verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([verify(p) for p in args.archives], indent=2))
