# 🧠 Modulo Percettivo Basato su Co-Occorrenze Temporali

Questa repository contiene la prima milestone del progetto **Arca** ricalibrata per
rispettare i vincoli imposti sul nuovo schema percettivo. Il focus è la costruzione di
adapters visivi e acustici che imparano esclusivamente da co-occorrenze ripetute nel
tempo, senza utilizzare loss contrastive né compiti predittivi cross-modali.

## Principi non negoziabili

- **Nessun contrastive loss.** L’apprendimento si basa su correlazioni temporali
  accumulate e non su InfoNCE/SimCLR/affini.
- **Nessuna predizione cross-modale esplicita.** Le associazioni emergono dalla
  co-presenza di stimoli visivi e sonori in clip temporali ripetuti.
- **Temporalità multi-scala.** Ogni associazione è pesata da un kernel temporale che
  combina più scale di decadimento, non solo dalla sincronia istantanea.
- **Parsing object-centric opzionale.** Gli oggetti vengono segmentati in maniera
  deterministica, senza supervisioni esterne o obiettivi predittivi.
- **Determinismo e riproducibilità.** Seed fissati, manifest JSON con tutti i fattori
  latenti e pipeline idempotente.

## Dataset sintetico controllato

Il dataset è generato on-the-fly da `src/data/datasets.py` e produce clip da 1.5–3 s con
1–3 oggetti e 1–3 eventi sonori. Ogni clip viene descritta nel manifest JSON con i
fattori latenti (forma, colore, scala, posizione, tempi).

- **Componenti visivi**: forme semplici (quadrato, cerchio, triangolo) renderizzate su
  sfondo nero con jitter controllato su colore, scala e posizione.
- **Componenti audio**: onde sinusoidali/saw/square generate sinteticamente, parametrizzate
  per pitch, timbro, durata e ampiezza.
- **Mapping deterministico**:
  - colore ↔ timbre (es. rosso → saw, verde → square, blu → sine)
  - forma ↔ intervallo (0/4/7 semitoni sull’oscillatore base)
  - scala ↔ loudness (ampiezza proporzionale alla dimensione visiva)
  - numero di oggetti ↔ polifonia (eventi audio sovrapposti)
- **Split composizionale**: alcune combinazioni forma/colore vengono tenute fuori dal
  training e appaiono solo in validation/test per il test zero-shot.
- **Manifest**: `data/processed/cooccurrence_manifest.json` elenca ogni clip con i
  metadati, mentre `cooccurrence_stats.json` raccoglie valori globali e kernel temporali.

## Pipeline

1. **Generazione dati** (deterministica):

   ```bash
   python -m src.data.datasets --config configs/default.yaml
   ```

   Produce tre file `.npz` (`train/val/test`) con tensori oggetto/evento già allineati,
   il manifest JSON e lo stats JSON.

2. **Training adapters** basato su co-occorrenza temporale:

   ```bash
   python -m src.trainers.ssl --config configs/default.yaml
   ```

   - Gli embeddings visivi e audio vengono calcolati per ogni oggetto/evento.
   - La loro similarità viene avvicinata al kernel temporale multi-scala precomputato.
   - Penalità di varianza garantiscono rappresentazioni non collassate.
   - Il training produce un checkpoint unico contenente entrambi gli adapters e la storia
     delle loss.

## Architettura

```
project/
├── configs/default.yaml           # configurazione globale (seed, dataset, trainer)
├── data/
│   ├── raw/                       # placeholder (nessun download necessario)
│   └── processed/                 # dataset sintetico + manifest/stats
├── src/
│   ├── data/datasets.py           # generatore sintetico co-occorrenza
│   ├── adapters/vision.py         # encoder CNN object-centric
│   ├── adapters/audio.py          # encoder 1D log-mel
│   └── trainers/ssl.py            # trainer co-occorrenza senza loss contrastive
└── artifacts/
    ├── ckpts/                     # checkpoint finali
    └── logs/                      # log + curva loss
```

## Dettagli del trainer

- **Input**: batch di clip con tensori `vision [B, O, 3, 96, 96]` e `audio [B, O, 64, T]`,
  maschere per oggetti/eventi e kernel temporale `[B, O, O]`.
- **Obiettivo**: minimizzare l’errore quadratico tra la similarità cosine (mappata in [0,1])
  e il kernel temporale normalizzato, garantendo che le correlazioni rispettino le scale
  temporali fornite.
- **Regolarizzazione**: varianza minima per ciascun embedding e bilanciamento delle medie
  di similarità per evitare collasso o allineamenti spurii.
- **Output**: `artifacts/ckpts/associative_adapters.pt` con pesi degli adapters e stato
  dell’optimizer.

## Configurazione

`configs/default.yaml` controlla:

- Numero di clip per split e combinazioni holdout.
- Ampiezza del jitter su colore/pitch/loudness/posizione/scala.
- Parametri del trainer: numero di epoche, learning rate, weight decay,
  soglia di varianza e scale del kernel temporale.

Modificando il file è possibile riprodurre esperimenti o creare nuovi manifest
composizionali mantenendo il determinismo (seed globale).

## Stato attuale

- ✅ Dataset sintetico minimalista con mapping deterministico e manifest dei fattori.
- ✅ Trainer che rispetta il vincolo di apprendimento per co-occorrenza senza loss
  contrastive né compiti predittivi cross-modali.
- ✅ Pipeline completamente riproducibile (stessi seed → stessi file, stessi risultati).

Questo modulo rappresenta la base sensoriale per step successivi del progetto Arca,
in cui verrà introdotto un backbone multimodale più ampio.


## Esportare l'intero progetto

Per creare rapidamente un archivio ZIP contenente l'intera repository (inclusi README e
configurazioni) è disponibile uno script dedicato:

```bash
python scripts/export_project.py
```

Il comando genera un file `zip` all'interno della cartella `exports/` con un timestamp nel
nome. Per salvare l'archivio in un percorso personalizzato è possibile usare l'opzione
`--output`:

```bash
python scripts/export_project.py --output /percorso/destinazione/arca.zip
```

Le cartelle temporanee come `.git`, `exports/` e `__pycache__/` sono escluse
automaticamente dall'archivio.
