🧠 Milestone 1 — Modulo Percettivo Auto-Supervisionato (Adapters)
📘 Obiettivo generale

Costruire un sottosistema percettivo universale che, senza alcuna supervisione esterna o hard-coding semantico,
impari a riconoscere e rappresentare in modo coerente pattern basso-livello (bordi, colori, texture, ritmi, timbri, ecc.)
producendo un embedding numerico d-dimensionale comune a tutte le modalità.

Questo modulo sarà la base sensoriale del progetto più ampio (“backbone di co-occorrenza / cervello”).
Deve poter essere addestrato from scratch, replicato integralmente, e fornire rappresentazioni
stabili, invariate e informative da collegare successivamente al backbone concettuale.

⚙️ Requisiti

Nessuna etichetta. Solo invarianze e regolarizzazioni auto-supervisionate.

Nessun hard-coding semantico. Vietato introdurre obiettivi “intelligenti” (es. Sobel, classificatori proxy).

Replicabilità assoluta. Config unica, seed fissi, checksum dataset, idempotenza totale.

Modularità. Ogni componente isolato e riutilizzabile: dati, adapters, trainer, metriche.

🗂️ Struttura della repository
project/
├── configs/
│   └── default.yaml                # tutti i parametri (dataset, model, trainer, seed…)
├── data/
│   ├── raw/                        # file scaricati (read-only)
│   └── processed/                  # feature preprocessate + stats
├── src/
│   ├── data/
│   │   ├── datasets.py             # download, checksum, split, DataLoader
│   │   └── augment.py              # augmentazioni per modalità
│   ├── adapters/
│   │   ├── vision.py               # CNN percettiva
│   │   └── audio.py                # CNN su log-mel
│   ├── trainers/
│   │   └── ssl.py                  # SimCLR-lite / VICReg-lite / DINO-lite
│   ├── eval/
│   │   └── metrics.py              # NN-consistency, uniformity, OOS test
│   └── utils/
│       ├── config.py               # parser YAML, override da CLI
│       ├── seed.py                 # seed globali e determinismo
│       ├── logging.py              # logger e tracking run
│       └── dist.py                 # setup multi-GPU opzionale
├── artifacts/
│   ├── runs/                       # tensorboard / wandb facoltativo
│   ├── ckpts/                      # checkpoint .pt
│   └── logs/                       # metriche + config salvati
├── README_MILESTONE1.md            # (questo file)
└── requirements.txt

🧩 Descrizione moduli
datasets.py

Scarica dataset pubblici (CIFAR-10, STL-10 unlabeled, DTD, ESC-50, UrbanSound8K, YESNO).
Per l'audio la configurazione di default usa UrbanSound8K con uno slicing ``train[:2000]``
per mantenere un compromesso tra varietà di classi e dimensione su disco.

Verifica checksum e decomprime.

Preprocess:

Visione: resize 96×96, normalizza mean/std train.

Audio: resample 16 kHz → log-mel (96 bande, 25 ms win, 10 ms hop) → standardizza per banda.

Split deterministico: 95 % train / 5 % val.

Genera file .npz/.pt in data/processed/.

Restituisce DataLoader train/val + OOS (dataset fuori dominio).

augment.py

Augmentazioni fisicamente plausibili ma non “semantiche”.

Visione: crop, flip, blur, jitter di colore, solarize soft.

Audio: time-shift, masking soft, jitter ampiezza, rumore bianco debole.
Parametri da config.yaml.

adapters/vision.py

Piccola CNN percettiva:

Conv(3×3, stride=1) → GELU → LayerNorm
Conv(3×3, stride=2) → GELU → LayerNorm
Conv(3×3, stride=2) → GELU → Global AvgPool
Linear(d_in→d) → LayerNorm → L2-norm


Output: vettore ℝ^d, con d=512 (default).

adapters/audio.py

Pipeline audio:

Input: log-mel [F×T]
Conv(3×3)×3 blocchi → pooling tempo
Linear(d_in→d) → LayerNorm → L2-norm

trainers/ssl.py

Implementa loop self-supervisionato scelto da config:

SimCLR-lite (contrastiva, InfoNCE τ).

VICReg-lite (invarianza + varianza + covarianza).

DINO-lite (teacher EMA, opzionale).
Supporta queue negativi, cosine LR, checkpoint.

metrics.py

NN-consistency: % in cui due viste augmentate sono NN reciproche.

Uniformity: varianza delle cosine → 1 = distribuzione uniforme.

OOS-shift: distanza media train ↔ OOS.

Retrieval robustness: rank reciproco medio intra-modal.

utils/

Seed, logging, distibuzione multiprocess e parsing config.

🧮 Configurazione (configs/default.yaml)
seed: 42
modalities: { vision: true, audio: true }
data:
  root: ./data
  datasets:
    vision: { name: CIFAR10, checksum: null, options: { train: true } }
    audio:  { name: UrbanSound8K, checksum: null, options: { split: "train[:2000]" } }
  batch_size: { vision: 256, audio: 128 }
  num_workers: 8
model:
  d: 512
  trainer: vicreg    # o simclr / dino
  epochs: 100
  lr: 1e-3
  weight_decay: 0.05
  temperature: 0.1
  augment:
    vision: { crop: 0.8, flip: true, color_jitter: 0.2, blur: 0.3 }
    audio:  { time_shift: 0.2, mask_time: 0.1, noise: 0.01 }
eval:
  every_n_epochs: 10
  metrics: [nn_consistency, uniformity, oos_shift]
logging:
  output_dir: ./artifacts
  use_wandb: false

🚀 Flusso operativo

Preparazione

python -m src.data.datasets --config configs/default.yaml


Scarica, preprocessa, salva statistiche normalizzazione.

Allenamento adapters

python -m src.trainers.ssl --config configs/default.yaml


→ produce checkpoint adapters_vision.pt, adapters_audio.pt, log metriche, curve uniformity.

Valutazione non-supervisionata

python -m src.eval.metrics --config configs/default.yaml --ckpt artifacts/ckpts/last.pt


Stampa e salva:

NN-consistency: ...
Uniformity: ...
OOS-shift: ...


Riproduzione

Run con seed identico → stessi risultati (±rumore floating).

Tutti i parametri e stats normalizzazione salvati in artifacts/.

📊 Risultati attesi e criteri di “prontezza”
1. Stabilità e non-collasso
Metrica	Descrizione	Soglia
Uniformity U	Varianza delle cosine intra-batch	0.8 ≤ U ≤ 1.2 (evita collasso o cluster unici)
Varianza embedding	Var(X) per dimensione	> 0.05
2. Invarianza percettiva
Test	Atteso
NN-consistency (augm → augm)	≥ 0.75
Retrieval OOS (stesso stimolo in dominio diverso)	≥ 0.6 reciprocal rank
3. Separabilità percettiva

k-means (k=50) su embedding → silhouette > 0.25 (senza label).

PCA 2D mostra cluster coerenti con pattern visivi/audio (texture, timbro).

4. Robustezza dominio

Media cosine(train,OOS)/cosine(train,train) ≥ 0.8 (≤ 20 % degrado).

5. Convergenza

Loss auto-sup stabile (plateau ±5 % ultime 10 epoche).

Nessun gradiente nan, varianza costante embedding.

✅ Il sistema è “pronto per il backbone” quando:

tutte le metriche sopra rientrano nelle soglie per 3 run consecutive (seed diversi);

l’output degli adapters (vision/audio) è normalizzato e stabile nel tempo;

il modello ricostruibile con python -m src.data.datasets && python -m src.trainers.ssl
produce embedding equivalenti (cosine > 0.98) rispetto al checkpoint di riferimento.

Solo a questo punto si procede alla Milestone 2 (introduzione del backbone di co-occorrenza e dell’allineamento multimodale).

🧩 Risultato finale della Milestone 1

✅ Adapters auto-supervisionati che forniscono vettori ℝ^d coerenti, invariante ad augmentazioni e robusti a shift OOS.

✅ Pre-processing pipeline deterministica e ri-eseguibile.

✅ Metriche quantitative di stabilità, invarianza e separabilità superate.

✅ Checkpoint e script ri-lanciabili da zero (replicabilità).

📅 Step successivi (Milestone 2 – preview)

Introduzione del backbone condiviso (Perceiver-like).

Training su co-occorrenze multimodali (immagine↔audio).

Misure di allineamento cross-modal e costruzione concettuale.

🧭 Note finali

Tutti i componenti devono poter essere lanciati anche separatamente (python -m src.adapters.vision --test → dry run).

Evitare ogni scorciatoia semantica: nessuna label, nessuna loss “furba”.

Ogni parametro modificato va tracciato in config_version.json.

Ogni milestone chiude solo quando i criteri quantitativi vengono soddisfatti.
