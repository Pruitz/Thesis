# Spiegazione Architetture di Segmentazione: Backbone (Encoder) e Testa (Decoder)

Questo documento fornisce una guida chiara, rigorosa e comparativa sulla decomposizione **Backbone / Testa** per i tre modelli di segmentazione semantica valutati nella tesi (FCN, DeepLabV3, LR-ASPP) e per il foundation model SAM 3 utilizzato nella fase di annotazione.

Tutti i dettagli corrispondono esattamente al testo redatto nel **Capitolo 4**, alle impostazioni sperimentali del **Capitolo 5** e ai risultati computazionali su NVIDIA Jetson Orin Nano del **Capitolo 7**.

---

## 1. Principio Fondamentale: Il Paradigma Encoder-Decoder

Nelle reti neurali convoluzionali per la segmentazione semantica, il problema consiste nell'assegnare a ogni singolo pixel $(i,j)$ dell'immagine di ingresso una classe semantica $c \in \{1, \dots, C\}$. L'architettura è formalmente suddivisa in due stadi complementari (Sezione 4.1 della tesi):

```
       IMMAGINE INPUT (H x W x 3)
                   │
                   ▼
     ┌───────────────────────────┐
     │      ENCODER (Backbone)   │  Riduzione risoluzione spaziale (stride S)
     │  Estrae feature semantiche│  Aumento della ricchezza semantica (canali C')
     └─────────────┬─────────────┘
                   │ Feature Maps compresse: F ∈ R^{(H/S) x (W/S) x C'}
                   ▼
     ┌───────────────────────────┐
     │   DECODER (Segmentation   │  Ripristina la risoluzione spaziale
     │           Head)           │  Fonde contesto multi-scala e mappa a C classi
     └─────────────┬─────────────┘
                   │ Logit o Probabilità: Z ∈ R^{H x W x C}
                   ▼
       MASCHERA PIXEL-LEVEL (H x W)
```

1. **Encoder (Backbone / Feature Extractor):**
   * **Obiettivo:** Trasformare i pixel grezzi in una rappresentazione astratta ad alto livello (bordi, pattern, forme di piante e terreno).
   * **Meccanismo:** Applica strati convoluzionali progressivi intervallati da pooling o stride, riducendo le dimensioni spaziali di un fattore $S \in \{8, 16, 32\}$ (output stride) e aumentando i canali fino a $C' \in [512, 2048]$.
   * **Transfer Learning:** I backbone (come ResNet-50 o MobileNetV3) derivano tipicamente da modelli pre-addestrati su ImageNet (1000 classi di classificazione), consentendo alla rete di partire da pesi visivi già ottimizzati.

2. **Decoder (Segmentation Head / Testa di Segmentazione):**
   * **Obiettivo:** Ricevere le feature compresse dell'encoder, recuperare il dettaglio spaziale e mappare i canali astratti nel numero esatto di classi del problema specifico ($C$ classi).
   * **Meccanismo:** Applica convoluzioni di fusione o moduli multi-scala (skip connections, ASPP o LR-ASPP), seguiti da classificatori $1 \times 1$ e upsampling bilineare o convoluzioni trasposte per tornare alla risoluzione originale $H \times W$.

---

## 2. Tabella Sinottica Comparativa dei Modelli nella Tesi

| Parametro / Componente | FCN-ResNet50 | DeepLabV3-ResNet50 | LR-ASPP-MobileNetV3 | SAM 3 (Annotatore) |
| :--- | :--- | :--- | :--- | :--- |
| **Ruolo nella Tesi** | Baseline storica di riferimento | Modello di massima accuratezza | **Modello finale scelto** (Real-time edge) | Generazione semi-automatica dataset |
| **Backbone (Encoder)** | ResNet-50 standard | ResNet-50 con convoluzioni atrous | MobileNetV3-Large | Perception Encoder (ViT unificato) |
| **Tipo di Convoluzioni nel Backbone** | Convoluzioni $2D$ standard + residual bottleneck | Convoluzioni $2D$ dilatate (dilation 2 e 4 in layer3/4) | Convoluzioni Depthwise Separable + Inverted Residuals | Self-Attention Transformer a blocchi |
| **Attenzioni nel Backbone** | Nessuna | Nessuna | Squeeze-and-Excitation (SE) nei blocchi | Multi-Head Self-Attention |
| **Output Stride del Backbone ($S$)** | **Stride 32** ($H/32 \times W/32$) | **Stride 8** ($H/8 \times W/8$) | **Multi-scala: Stride 16 e Stride 8** | Stride 16 (Patch embedding) |
| **Canali estratti dal Backbone** | 2048 da `layer4` (1024 da `layer3` per aux) | 2048 da `layer4` dilatato | 960 da stride 16; 40 da stride 8 | Embedding vettoriale a 256/768 dim |
| **Testa (Segmentation Head)** | **`FCNHead`** | **`ASPP`** (Atrous Spatial Pyramid Pooling) | **`LRASPPHead`** (Lite Reduced ASPP) | **Prompt-Driven Mask Decoder** (DETR-based) |
| **Operazioni Chiave della Testa** | Conv $3\times3$ (512 ch), BN, ReLU, Dropout 0.1, Conv $1\times1$ | 5 rami paralleli: $1\times1$, 3 atrous (rate 12, 24, 36), Global Pooling; concat (1280 ch) $\to$ $1\times1$ (256 ch) | Global Context Gating (GAP + Sigmoid), moltiplicazione, Conv $1\times1$ separati e **somma in logit space** | Cross-attention tra prompt (punti/box/testo) e feature visuali |
| **Upsampling Finale** | Bilineare $32\times$ | Bilineare $8\times$ | Bilineare $8\times$ | Convoluzioni trasposte $2\times$ + interpolazione |
| **Numero Parametri** | 32.95 M | 39.63 M | **3.22 M** ($-90\%$ rispetto a FCN) | Centinaia di M (Foundation model) |
| **Latenza Jetson Orin Nano (FP16)**| 38.2 ms (26 FPS) | 71.4 ms (14 FPS) | **10.41 ms (96 FPS)** | Non idoneo a real-time embedded |
| **mIoU Validazione (Tesi Cap. 8)** | 84.12% | 89.20% | **88.75%** | Utilizzato offline come oracolo |

---

## 3. Dettaglio Architetturale per Singolo Modello

### 3.1 FCN-ResNet50 (Sezione 4.2 della Tesi)

```
Input (H x W x 3)
      │
[ResNet-50 Backbone]
      ├── layer3 (stride 16, 1024 ch) ──> [FCNHead Aux: 3x3 (256ch) + 1x1] ──> Loss Aux (peso 0.4)
      └── layer4 (stride 32, 2048 ch)
            │
            ▼
      [FCNHead Principale]
            ├── Conv 3x3 (2048 -> 512 canali)
            ├── BatchNorm + ReLU
            ├── Dropout(0.1)
            └── Conv 1x1 (512 -> C classi)
            │
            ▼ Stride 32 Score Map
      [Upsampling Bilineare 32x] ─────────> Predizione finale H x W
```

* **Backbone:** ResNet-50 convenzionale. A ogni stadio (`layer1` a stride 4, `layer2` a stride 8, `layer3` a stride 16, `layer4` a stride 32), la risoluzione dimezza e i canali raddoppiano. L'estrazione finale avviene a stride 32 con 2048 canali.
* **Testa (`FCNHead`):**
  1. Convoluzione intermedia $3 \times 3$ con padding 1 che proietta i 2048 canali a **512 canali**.
  2. Batch Normalization e funzione di attivazione ReLU.
  3. Dropout con probabilità $p = 0.1$ per regolarizzazione.
  4. Convoluzione pointwise $1 \times 1$ finale che mappa da 512 canali alle $C$ classi di output.
* **Testa Ausiliaria:** Durante il training, una seconda `FCNHead` identica è collegata a `layer3` (stride 16, 1024 canali) con una proiezione a 256 canali per calcolare la perdita ausiliaria pesata a $0.4$, prevenendo la scomparsa del gradiente negli strati profondi.
* **Limite concettuale:** Il fattore di compressione $32\times$ elimina i dettagli fini dei contorni delle colture, costringendo l'interpolazione a ricostruire $32$ pixel da un singolo punto di griglia.

---

### 3.2 DeepLabV3-ResNet50 (Sezione 4.3 della Tesi)

```
Input (H x W x 3)
      │
[Dilated ResNet-50 Backbone] (stride limitato a 8, dilation r=2 in layer3, r=4 in layer4)
      └── layer4 output (stride 8, 2048 canali)
            │
            ▼
      [Modulo ASPP (Atrous Spatial Pyramid Pooling)]
            ├── Ramo 1: Conv 1x1 (2048 -> 256 canali)
            ├── Ramo 2: Atrous Conv 3x3, rate=12 (2048 -> 256 canali)
            ├── Ramo 3: Atrous Conv 3x3, rate=24 (2048 -> 256 canali)
            ├── Ramo 4: Atrous Conv 3x3, rate=36 (2048 -> 256 canali)
            └── Ramo 5: Global Average Pooling + Conv 1x1 (2048 -> 256 canali) + Upsample
            │
            ▼ Concatenazione di tutti i 5 rami (5 * 256 = 1280 canali)
      [Proiezione e Classificazione]
            ├── Conv 1x1 (1280 -> 256 canali) + BN + ReLU
            ├── Conv 3x3 (256 -> 256 canali) + BN + ReLU
            └── Conv 1x1 (256 -> C classi)
            │
            ▼ Stride 8 Logits
      [Upsampling Bilineare 8x] ──────────> Predizione finale H x W
```

* **Backbone:** ResNet-50 modificata con **convoluzioni atrous (dilatate)**. Invece di dimezzare la risoluzione in `layer3` e `layer4` con stride 2, lo stride viene mantenuto a 1 e i filtri vengono dilatati con fattori $r=2$ e $r=4$. In questo modo l'uscita rimane ad alta risoluzione (**stride 8** invece di stride 32), preservando la fedeltà spaziale senza ridurre il campo ricettivo (*receptive field*).
* **Testa (`ASPP`):**
  1. Campiona il contesto a scale multiple contemporaneamente tramite 5 rami paralleli:
     * Un ramo $1 \times 1$ per le feature locali.
     * Tre rami convoluzionali $3 \times 3$ con fattori di dilatazione $r \in \{12, 24, 36\}$ (per stride 8) che catturano contesti spaziali di raggio crescente.
     * Un ramo di Global Average Pooling (GAP) che cattura il contesto globale dell'intera scena.
  2. Tutti i 5 rami producono 256 canali e vengono concatenati (totale $1280$ canali).
  3. Una convoluzione $1 \times 1$ riduce i canali a $256$, seguita da una convoluzione $3 \times 3$ di raffinamento e dal classificatore $1 \times 1$ finale.
* **Punto di forza e debolezza:** Eccellente precisione sui bordi (mIoU $89.20\%$), ma i 5 rami paralleli su feature map a stride 8 (risoluzione 4 volte superiore a FCN) richiedono un consumo di memoria e banda proibitivo su hardware embedded edge (latenza di 71.4 ms, insufficiente per il target real-time di $\ge 20$ Hz).

---

### 3.3 LR-ASPP MobileNetV3-Large (Sezione 4.4 della Tesi)

```
Input (H x W x 3)
      │
[MobileNetV3-Large Backbone]
(Inverted Residuals, Depthwise Separable Conv, Squeeze-and-Excitation, hard-swish)
      ├── Estrazione Stride 8  (F_high: H/8 x W/8 x 40 canali)
      └── Estrazione Stride 16 (F_low:  H/16 x W/16 x 960 canali)
            │
            ▼
      [LRASPPHead - Ramo Bassa Risoluzione F_low]
            ├── Percorso CBR: Conv 1x1 (960 -> 128) + BN + ReLU ───┐
            │                                                      │ (Moltiplicazione elemento per elemento)
            └── Global Context Gate: GAP + Conv 1x1 (960 -> 128) + Sigmoid ┘
            │
            ▼ Modulazione (128 canali)
      [Upsampling Bilineare 2x] (porta a stride 8)
            │
            ▼
      [Classificatore Low-Res]: Conv 1x1 (128 -> C classi) ──> Logits Z_low (stride 8)
                                                                    │
      [Classificatore High-Res]: Conv 1x1 (40 -> C classi)  ──> Logits Z_high (stride 8)
                                                                    │
                                            ┌───────────────────────┘
                                            ▼
                                [Somma in Logit Space]
                                Z_fused = Z_low + Z_high
                                            │
                                            ▼
                               [Upsampling Bilineare 8x] ──────> Predizione finale H x W
```

* **Backbone:** MobileNetV3-Large progettato specificamente per hardware a basso consumo:
  * **Convoluzioni Depthwise Separabili:** scompongono la convoluzione standard in una componente spaziale per canale ($3\times 3$) e una pointwise ($1\times 1$), riducendo le operazioni dell'88--89% (Eq. 4.12 della tesi).
  * **Blocchi Inverted Residual:** struttura *narrow $\to$ wide $\to$ narrow*, con espansione interna ($e=3$ o $e=6$), convoluzione depthwise e proiezione a collo di bottiglia lineare (senza ReLU) per evitare la distruzione di informazione geometrica (Sezione 4.4.2).
  * **Squeeze-and-Excitation (SE):** calibra l'importanza relativa dei canali comprimendo le dimensioni spaziali con un pooling globale e ripesando i canali tramite due strati fully-connected con attivazione hard-sigmoid (Eq. 4.14).
  * **Non-linearità Efficienti:** sostituisce sigmoid e swish trascendenti con approssimazioni lineari a tratti basate su $\text{ReLU6}$ (`h-sigmoid` e `h-swish`, Eq. 4.17 e 4.18), eliminando i calcoli esponenziali in virgola mobile.
* **Testa (`LRASPPHead`):**
  1. **Decomposizione Multi-Risoluzione:** preleva simultaneamente due flussi:
     * $F_{\text{low}}$ a stride 16 (960 canali): ricco di astrazione semantica per distinguere le classi.
     * $F_{\text{high}}$ a stride 8 (40 canali): ricco di dettagli spaziali per localizzare con precisione i confini interfilari.
  2. **Global Context Gating:** sostituisce i 5 rami pesanti dell'ASPP con un singolo pooling globale seguito da una convoluzione $1\times 1$ e attivazione sigmoide standard $\sigma$, che agisce da modulatore di gating sui canali estratti dal percorso convoluzionale ($128$ canali).
  3. **Fusione in Logit Space:** a differenza delle altre reti che concatenano le feature intermedie e usano classificatori congiunti, LR-ASPP proietta **indipendentemente** i due rami nello spazio delle classi tramite due convoluzioni $1\times 1$, e ne somma direttamente i logit:
     $$Z_{\text{fused}} = Z_{\text{low}} + Z_{\text{high}}$$
  4. L'uscita viene poi riportata a piena risoluzione tramite un unico upsampling bilineare $8\times$.
* **Vantaggio determinante:** Con appena **3.22 milioni di parametri** e un tempo di inferenza di **10.41 ms** su NVIDIA Jetson Orin Nano (96 FPS in FP16), offre un'accuratezza quasi identica a DeepLabV3 (mIoU dell'88.75% contro 89.20%), garantendo il pieno rispetto del vincolo real-time del robot agricolo Demetra.

---

## 4. SAM 3 nel Flusso della Tesi (Foundation Model di Annotazione)

* **Perché è presente nella tesi (Sezione 4.5 e 5.2):** Non viene eseguito a bordo del robot per la guida, bensì impiegato nella pipeline di preparazione dati per abbattere il tempo di annotazione manuale dei pixel del corridoio agricolo da 25 minuti a pochi secondi per immagine.
* **Backbone:** Perception Encoder (PE), un Transformer visuale (ViT) condiviso che elabora l'immagine e i descrittori linguistici open-vocabulary.
* **Testa:** Mask Decoder iterativo basato su query e cross-attention di tipo DETR, capace di ricevere prompt sia geometrici (coordinate di punti o bounding box sul terreno) sia testuali (es. *"central farm path"* o *"lettuce crop"*), producendo maschere di segmentazione binarie o multi-classe ad altissima risoluzione.

---

## 5. Riferimenti Incrociati ai Capitoli della Tesi

| Capitolo | Sezione | Argomento Trattato |
| :--- | :--- | :--- |
| **Capitolo 2** | Sez. 2.2, Fig. 2.2 | Inserimento della segmentazione semantica all'interno del flusso complessivo di guida autonoma. |
| **Capitolo 4** | Sez. 4.1 | Teoria generale Encoder-Decoder e formulazione matematica della segmentazione. |
| **Capitolo 4** | Sez. 4.2 | FCN: skip connections (FCN-32s, 16s, 8s) e testa convoluzionale. |
| **Capitolo 4** | Sez. 4.3 | DeepLabV3: atrous convolutions, output stride 8 e modulo ASPP. |
| **Capitolo 4** | Sez. 4.4 | MobileNetV3 e LR-ASPP: depthwise separable, inverted residuals, SE, hard-swish, global context gate e fusione logit. |
| **Capitolo 4** | Sez. 4.5 | SAM 3: foundation model, promptable segmentation, Perception Encoder e DETR decoder. |
| **Capitolo 5** | Sez. 5.1--5.3 | Dataset multi-coltura, annotazione con SAM 3, Weighted Cross-Entropy Loss e loss ausiliaria. |
| **Capitolo 7** | Sez. 7.3 | Esportazione in ONNX FP16, allocazione memoria e benchmark di latenza/FPS su Jetson Orin Nano. |
| **Capitolo 8** | Sez. 8.1 | Risultati sperimentali comparativi: mIoU per classe, trade-off accuratezza-velocità e giustificazione della scelta di LR-ASPP. |
