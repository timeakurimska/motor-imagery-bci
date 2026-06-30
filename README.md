# Klasifikácia motorickej imaginácie s využitím BCI zariadenia

Zdrojový kód a dataset k diplomovej práci *Klasifikácia motorickej imaginácie
s využitím BCI zariadenia* (Ústav informatiky, Prírodovedecká fakulta UPJŠ
v Košiciach, 2026).

Projekt skúma, do akej miery je možné z EEG signálu spoľahlivo rozlíšiť pokoj
od motorickej imaginácie (a ľavú vs. pravú ruku) na cenovo dostupnom
spotrebiteľskom zariadení **Neurosity Crown** s ôsmimi suchými elektródami,
pri prenose modelu medzi rôznymi meraniami (cross-session). Navrhnutý postup
spracovania je založený na kovariančných maticiach, nesupervizovanej doménovej
adaptácii (recentering) a Riemannovskej geometrii.

## Štruktúra repozitára

```
motor-imagery-bci/
├── data/                           # sem sa ukladajú nové merania zo zberu dát
│   └── session_glob/               # 7 meraní použitých v experimentoch
│       └── brainwaves_*_T6s.json   # surový EEG signál vo formáte JSON
├── imgs/
│   ├── rest.png                    # vizuálny podnet – pokoj
│   ├── left.png                    # vizuálny podnet – MI ľavej ruky
│   └── right.png                   # vizuálny podnet – MI pravej ruky
├── .env.example                    # vzor pre prístupové údaje k zariadeniu
├── .gitignore                      # súbory a priečinky ignorované Gitom
├── 01_rest_vs_mi.ipynb             # binárna úloha: pokoj vs. MI
├── 02_rest_left_right.ipynb        # trojtriedna úloha: pokoj / ľavá / pravá
├── 03_ablacia.ipynb                # ablačná analýza reťazca
├── README.md                       # dokumentácia projektu
├── crown_handler.py                # pripojenie a stream zo zariadenia Neurosity Crown
├── mi_pipeline.py                  # spoločné parametre a LOSO logika
├── record_session.py               # aplikácia na zber dát (tkinter + Neurosity SDK)
└── requirements.txt                # zoznam Python knižníc
```


## Požiadavky

- Python 3.12
- knižnice zo súboru `requirements.txt` (hlavné: MNE 1.11, scikit-learn 1.8,
  pyRiemann 0.11, Pillow 12.0, Neurosity SDK)
- `tkinter` — súčasť štandardného Pythonu na Windows a macOS; na Linuxe ho
  prípadne doinštaluj systémovo (`sudo apt install python3-tk`)

## Inštalácia

```bash
# 1. (odporúčané) vytvor virtuálne prostredie
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. nainštaluj knižnice
pip install -r requirements.txt
```

## Prístupové údaje k zariadeniu

Aplikácia na zber dát sa pripája k Neurosity Crown cez Neurosity SDK, ktoré
vyžaduje prihlasovacie údaje. Tie sa **nikdy neukladajú do repozitára**.
Skopíruj vzorový súbor a doplň vlastné hodnoty:

```bash
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env
```

Potom v súbore `.env` vyplň svoje údaje (device ID, e-mail a heslo k účtu
Neurosity). Súbor `.env` je v `.gitignore`, takže sa neodošle na GitHub.

## Použitie

**Zber dát:**

```bash
python record_session.py
```

Spustí sa aplikácia, ktorá zobrazuje vizuálne podnety (pokoj / ľavá / pravá
ruka) a ukladá surový EEG signál do `data/` vo formáte JSON.

**Analýza a klasifikácia:**

Notebooky spusti cez Jupyter:

```bash
jupyter notebook
```

- `01_rest_vs_mi.ipynb` — binárna klasifikácia pokoja oproti MI
- `02_rest_left_right.ipynb` — trojtriedna klasifikácia
- `03_ablacia.ipynb` — ablačná analýza jednotlivých krokov reťazca

Všetky tri používajú spoločný modul `mi_pipeline.py` (rovnaké parametre
a LOSO validácia).

## Dataset

Adresár `data/session_glob/` obsahuje vlastný EEG dataset zaznamenaný počas siedmich nezávislých meraní v rozpätí dvoch mesiacov (spolu 650 trialov, tri triedy: pokoj, MI ľavej a MI pravej ruky) od jedného dobrovoľného subjektu. Dáta sú anonymné a slúžia výhradne na výskumné účely; nie sú určené na klinickú diagnostiku.

