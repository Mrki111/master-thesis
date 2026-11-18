"""Create a private committee handover archive and verify every file hash."""
from __future__ import annotations
import hashlib
import json
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'release'
OUT.mkdir(exist_ok=True)
verification = ROOT/'results/thesis/mentor_review_20261118/VERIFIED.json'
assert json.loads(verification.read_text())['checks_passed'], 'Verify the completed review first.'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=OUT/'IRCNN_reprodukcija_20261118.zip')
args = parser.parse_args()
target = args.output.resolve()
target.parent.mkdir(parents=True,exist_ok=True)
assert not target.exists(), 'Preserve existing handovers; choose a new archive name.'
paths = set()

def include(base, accept=lambda p: True):
    for p in (ROOT / base).rglob('*'):
        if p.is_file() and accept(p):
            paths.add(p)

for pattern in ('*.m', '*.md', '.gitignore', 'Install_thesis_bm3d.ps1'):
    paths.update(p for p in ROOT.glob(pattern) if p.is_file())
include('experiments', lambda p: 'third_party' not in p.parts and '.runtime' not in p.parts
        and p.suffix.lower() in {'.m', '.hpp', '.cu', '.cmd', '.xml'})
include('utilities')
include('models')
include('kernels')
for dataset in ('BSD68', 'CBSD68', 'Set12', 'Val20', 'Set3G', 'Set5', 'Set14'):
    include('testsets/' + dataset)
include('matconvnet', lambda p: p.suffix.lower() not in {'.dll', '.obj', '.pdb', '.lib', '.exp'}
        and not any(s in {'build', '.git', '__pycache__'} for s in p.parts))
for name in ('full_20261011T180756', 'fourier_alignment_20261021', 'mentor_review_20261118'):
    include('results/thesis/' + name)
for smoke in (ROOT/'results/thesis').glob('smoke_20261118*'):
    include(smoke.relative_to(ROOT).as_posix())
for name in ('thesis_cuda_check.log', 'thesis_matconvnet_gpu_model_check.log',
             'thesis_matconvnet_gpu_build.log', 'mentor_probe.log', 'mentor_review.log',
             'mentor_environment_versions.log', 'mentor_numeric_tests.log',
             'mentor_smoke.log', 'mentor_review_verification.log', 'package_relocation_check.log'):
    p = ROOT / 'logs' / name
    if p.exists():
        paths.add(p)
include('tools', lambda p: p.suffix in {'.py', '.ps1'})
paths.add(ROOT/'thesis/ODGOVOR_NA_PRIMJEDBE_2026-11-18.md')
# Scientific handover excludes the Word thesis and all personal authoring history.
manifest = []
with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=5, allowZip64=True) as z:
    for p in sorted(paths):
        rel = p.relative_to(ROOT).as_posix()
        assert 'third_party/bm3d/' not in rel and not rel.endswith('BM3D.zip')
        assert not rel.endswith('.dll')
        data = p.read_bytes()
        z.writestr(rel, data)
        manifest.append({'path': rel, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    z.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
with ZipFile(target) as z:
    for item in manifest:
        assert hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256'], item['path']
digest = hashlib.sha256(target.read_bytes()).hexdigest()
(OUT / (target.name + '.sha256')).write_text(digest + '  ' + target.name + '\n', encoding='ascii')
print(json.dumps({'archive': str(target), 'files': len(manifest), 'bytes': target.stat().st_size,
                  'sha256': digest, 'all_members_verified': True}, indent=2))
