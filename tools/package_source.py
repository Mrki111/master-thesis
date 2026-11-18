"""Make a compact code/configuration handover from the verified full archive."""
import hashlib
import json
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
release=ROOT/'release'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--full',type=Path,default=release/'IRCNN_reprodukcija_20261118.zip')
parser.add_argument('--output',type=Path,default=release/'IRCNN_kod_i_konfiguracije_20261118.zip')
args=parser.parse_args()
full=args.full.resolve()
target=args.output.resolve()
target.parent.mkdir(parents=True,exist_ok=True)
assert not target.exists()
manifest=[]
with ZipFile(full) as source, ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=6) as out:
    for item in json.loads(source.read('MANIFEST.json')):
        name=item['path']; p=Path(name)
        if name.startswith(('models/','testsets/','kernels/')):continue
        if p.suffix.lower() in {'.mexw64','.mexa64','.mexmaci64','.png','.jpg','.jpeg','.bmp','.tif','.tiff','.mat'}:
            # States retain exact configurations and numerical rows, not pixels.
            if p.name not in {'state.mat','state_before_config_annotation.mat'}:continue
        if name.startswith('results/') and '/source/' not in name and p.suffix.lower() not in {'.csv','.md','.json','.mat','.txt'}:continue
        data=source.read(name); assert hashlib.sha256(data).hexdigest()==item['sha256']
        out.writestr(name,data);manifest.append(item)
    readme=(
        '# Compact thesis code and configuration package\n\n'
        'This package contains the MATLAB code, setup adapters, MatConvNet source, '
        'configurations, numerical records and verification logs. It omits image data, '
        'pretrained weights, saved observations and compiled binaries.\n\n'
        f'The complete companion archive {full.name} contains '
        'those research inputs and the reviewed Windows MEX binaries. Obtain both '
        'packages from the thesis author when reproducing the saved experiments. '
        'Read REPRODUCIBILITY.md for exact commands and dependency versions. '
        'Without the supplied binaries, compile using Setup_thesis_matconvnet(true) '
        'for CPU or Setup_thesis_matconvnet(true,[],\'gpu\') for GPU. '
        'MATLAB, NVIDIA libraries and the restricted BM3D package are obtained separately.\n\n'
        'This is a local handover prepared for repository upload or email; it has '
        'not been uploaded or sent. Preserve third-party attribution and licence notices.\n')
    out.writestr('READ_FIRST_SOURCE_PACKAGE.md',readme)
    out.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
with ZipFile(target) as z:
    for item in manifest:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
digest=hashlib.sha256(target.read_bytes()).hexdigest()
target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n',encoding='ascii')
print(json.dumps({'archive':str(target),'files':len(manifest)+1,'bytes':target.stat().st_size,
                  'sha256':digest,'members_verified':True},indent=2))
