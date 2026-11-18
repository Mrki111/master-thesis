"""Write the committee-facing review record from verified CSV evidence."""
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/thesis/mentor_review_20261118'
verified=json.loads((FOLDER/'VERIFIED.json').read_text())
assert verified['checks_passed']
def read(name):
    with (FOLDER/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def n(value,d=2):return f'{float(value):.{d}f}'.replace('.',',')
bm=read('bm3d_summary.csv'); grid=read('grid_changes.csv'); rgb=read('rgb_reproduced.csv')
changed=[r for r in grid if r['OldParameters']!=r['NewParameters']]
lines=['# Provjera primjedbi mentora od 18. novembra 2026.', '',
       'Dopunski eksperimenti koriste ista sačuvana opažanja kao osnovni eksperiment. '
       'Prvobitni rezultati i Furjeova korekcija ostaju sačuvani. Izmjene rada prate dopunu odvojeno od osnovnih tabela.', '',
       '## 1 Poređenje sa BM3D', '',
       'Izvršena je zvanična naslijeđena BM3D implementacija v2.01, normalni profil, '
       'na svih 68 BSD68 slika za svaki nivo Gausovog šuma. Referenca nije proslijeđena restauraciji. '
       'Poznata σ, opažanja, kvantizacija i metrike usklađeni su sa IRCNN poređenjem.', '',
       '| σ | BM3D PSNR dB | BM3D SSIM | IRCNN − BM3D dB | IRCNN PSNR pobjede |',
       '|---:|---:|---:|---:|---:|']
for r in bm:lines.append('| '+' | '.join([r['Case'].split('_')[-2],n(r['MeanPSNR']),n(r['MeanSSIM'],3),n(r['IRCNNMinusMethodPSNR']),r['IRCNNWins']+'/68'])+' |')
lines+=['', 'Prednost IRCNN-a od 0,56–0,58 dB prema BM3D znatno je manja od razlike prema '
        'lokalnim filtrima. Dopuna obuhvata sive Gausove uslove; ne određuje rang prema kolor BM3D ili svim klasičnim metodama.', '',
        '## 2 Širina validacione pretrage', '',
        'Bilateralna pretraga proširena je sa 6 na 16 kombinacija: raniji kandidati zadržani su, '
        'uz prostornu standardnu devijaciju 3, intenzitetske množioce 6 i 12 i granični slučaj ∞. '
        'Wienerovi dekonvolucioni množioci prošireni su do 1024, Richardson–Lucy do 320 iteracija, '
        'a slijepa dekonvolucija i ispod ranije donje granice, na 1/2/5 iteracija.', '',
        f'Izbor se promijenio u {len(changed)} od 20 provjerenih parova uslova i metode. '
        'Sve postavke izabrane su prema validacionom PSNR-u na istim odvojenim skupovima; '
        'testne slike nijesu učestvovale u izboru. Promijenjeni izlazi ponovo su ocijenjeni na sačuvanim opažanjima.', '',
        '| Uslov | Metoda | Ranije | Poslije | Validacioni dobitak dB |',
        '|---|---|---|---|---:|']
for r in grid:
    lines.append('| '+' | '.join([r['Case'],r['Method'],r['OldParameters'],r['NewParameters'],n(float(r['ValidationPSNR'])-float(r['OldValidationPSNR']),4)])+' |')
lines+=['', '∞ označava prostorno Gausovo filtriranje pri istom konačnom susjedstvu. '
        'Preostali granični izbori i dalje ograničavaju zaključak; proširena mreža ne dokazuje globalni optimum.', '',
        '## 3 RGB obrada i metrike', '',
        'Svih 68 CBSD68 slika ima tri stvarna, međusobno različita kanala. Sačuvane čiste reference '
        'jednake su izvornim kolor ulazima. Ponovo je generisano svih 476 RGB opažanja i reprodukovano '
        'svih 1.428 ocjena Wienerovog, medijanskog i bilateralnog filtra. Provjerena je obrada svakog '
        'kanala, PSNR iz zajedničkog RGB MSE-a i SSIM kao prosjek kanala.', '',
        f"Najveća apsolutna razlika PSNR-a je {max(abs(float(r['PSNRError'])) for r in rgb):.3g}, "
        f"a SSIM-a {max(abs(float(r['SSIMError'])) for r in rgb):.3g}. "
        'Nije nađena neželjena konverzija u sive tonove niti prepisivanje rezultata. '
        'Učitavanje RGB denoizirajućih ulaza sada odbija jednokanalne datoteke umjesto tihog ponavljanja kanala.', '',
        '## 4 Okruženje i kod za reprodukciju', '',
        'Sačuvani state.mat potvrđuje MATLAB 8.5.0.197613 (R2015a), MatConvNet 1.0-beta25 '
        'i NVIDIA GeForce RTX 4070 Laptop GPU. Log kompilacije i nezavisna CUDA provjera '
        'potvrđuju nvcc 11.0.194 i cuDNN 8.0.1. Kombinacija koristi lokalnu adaptaciju '
        'cuDNN interfejsa i compute-7.5 PTX. MATLAB-ov ToolkitVersion 6.5 odnosi se '
        'na njegov ugrađeni alatni skup, odvojeno od CUDA 11.0 korišćenog za spoljašnji MEX modul.', '',
        'Uputstvo: [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). Lokalni paket za predaju Komisiji '
        'sadrži kod, konfiguracije, modele, potrebne skupove, opažanja, zapise i SHA-256 manifest. '
        'BM3D se preuzima sa [stranice autora](https://webpages.tuni.fi/foi/GCF-BM3D/), '
        'u skladu sa njegovom licencom; MATLAB i NVIDIA zavisnosti pribavljaju se zasebno. '
        'Pripremljeni paket nije objavljen na GitHub-u i mejl nije poslat.', '',
        '## Provjere', '',
        f"Verifikovano je {verified['rows']} ocjena metoda, uključujući 204 nova BM3D rezultata. "
        f"Ponovo je izmijenjeno {verified['changed_baseline_rows']} osnovnih ocjena, a "
        f"{verified['inherited_rows']} zadržano je iz prethodno korigovanog eksperimenta. "
        'Provjere obuhvataju potpunost, uparene slike i sjemena, obnovu opažanja, '
        'validacioni izbor, RGB regresije i očuvanje naslijeđenih rezultata. '
        'Izvorni dokazi nalaze se u results/thesis/mentor_review_20261118.', '']
(ROOT/'thesis/ODGOVOR_NA_PRIMJEDBE_2026-11-18.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote mentor review record.')
