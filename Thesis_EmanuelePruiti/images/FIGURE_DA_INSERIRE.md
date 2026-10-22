# Guida e Riepilogo delle Figure da Inserire nella Tesi

Questo documento elenca e organizza tutte le figure mancanti nella tesi (i box `\placeholderfigure`), specificando per ciascuna cosa deve mostrare, la fonte (paper, script o foto reale) e la priorità di inserimento.

---

## 1. Figure da Paper Scientifici / Letteratura (Capitolo 4) -- [COMPLETATO: 100%]

Tutte le 4 figure architetturali di letteratura per il Capitolo 4 sono state estratte dai paper ufficiali ad alta risoluzione (300 DPI) e integrate nel testo:
1. `fig:fcn_architecture` (Long et al., 2015) -- Pagina 21 (`images/architectures/fcn_architecture.png`)
2. `fig:deeplabv3_aspp` (Chen et al., 2017) -- Pagina 23 (`images/architectures/deeplabv3_aspp.png`)
3. `fig:mobilenetv3_lraspp` (Howard et al., 2019) -- Pagina 28:
   * (a) `fig:mobilenetv3_block`: Blocco Inverted Residual con SE (`images/architectures/mobilenetv3_block.png`)
   * (b) `fig:lraspp_head`: Testa di segmentazione Lite R-ASPP (`images/architectures/lraspp_head.png`)
4. `fig:sam3_architecture` (Kirillov et al., 2023) -- Pagina 30 (`images/architectures/sam_architecture.png`)

Nessun placeholder rimanente nel Capitolo 4.

---

## 2. Grafici e Schemi Algoritmici (Generabili via Software / Python)

*Queste figure possono essere generate automaticamente tramite script Python (Matplotlib), TikZ o diagrammi vettoriali senza necessità di fotografie.*

### `fig:overall_system_pipeline` (Capitolo 2, riga 58)
* **Descrizione:** Diagramma a blocchi generale dell'intero sistema autonomo.
* **Flusso:** 
  `Camera RGB (Acquisizione)` $\longrightarrow$ `Segmentazione Semantica (LR-ASPP ONNX FP16)` $\longrightarrow$ `Maschera Corridoio Terreno` $\longrightarrow$ `Estrazione Geometrica a Fasce (IRLS Bilaterale)` $\longrightarrow$ `Calcolo Errori (ey, e_theta)` $\longrightarrow$ `Controllo Sterzo 4WS (Robot Demetra)`.
* **Formato consigliato:** Diagramma pulito SVG/PNG o TikZ.

### `fig:field_corridor_geometry` (Capitolo 2, riga 40)
* **Descrizione:** Modello geometrico del robot all'interno del corridoio interfilare.
* **Elementi da evidenziare:** La larghezza del corridoio $W$, la linea centrale ideale $x_C(y)$, l'asse longitudinale del robot, l'offset laterale $e_y$ e l'errore di orientamento (angolo di heading) $e_\theta$.
* **Formato consigliato:** Schema vettoriale (Matplotlib o TikZ).

### `fig:training_loss_curves` (Capitolo 5, riga 212)
* **Descrizione:** Curve di apprendimento durante le 200 epoche di training per i tre modelli (FCN, DeepLabV3, LR-ASPP).
* **Elementi:** Due sotto-grafici:
  1. Andamento della Weighted Cross-Entropy Loss su train e validation.
  2. Andamento della Mean IoU (mIoU) di validazione, evidenziando il picco all'epoca 194 per LR-ASPP.
* **Fonte dati:** Ricavabile direttamente dal file `training_log.csv` generato durante i run.

### `fig:ros2_architecture` (Capitolo 7, riga 63)
* **Descrizione:** Grafo dell'architettura software in ROS 2.
* **Nodi e Topic:**
  * Nodo telecamera $\to$ topic `/camera/image_raw`.
  * Nodo inferenza `image_segmenter_node` $\to$ topic `/corridor_mask`.
  * Nodo geometrico `geometric_extractor_node` $\to$ topic `/navigation_reference` (`ey`, `etheta`).
  * Nodo controllore cinematico 4WS $\to$ topic `/cmd_vel` verso i driver dei motori.
* **Formato:** Schema a blocchi o esportazione da `rqt_graph`.

### `fig:jetson_timing_breakdown` (Capitolo 7, riga 257)
* **Descrizione:** Grafico a barre con la suddivisione temporale dei millisecondi impiegati per ogni ciclo di controllo su NVIDIA Jetson Orin Nano.
* **Valori:**
  * Pre-processing (resize, normalizzazione): $\sim 1.2$ ms
  * Inferenza LR-ASPP in ONNX FP16: $10.41$ ms
  * Estrazione geometrica Cauchy-IRLS su 24 fasce: $0.08$ ms
  * Pubblicazione e comando ROS 2: $< 0.1$ ms
  * **Totale ciclo:** $\sim 11.8$ ms ($> 80$ Hz, ampiamente entro il budget real-time di 20 Hz).

### `fig:navigation_tracking_errors` (Capitolo 8, riga 190)
* **Descrizione:** Grafico temporale degli errori di inseguimento durante i test in campo del robot Demetra.
* **Elementi:** Due curve nel tempo $t \in [0, T]$:
  * Errore laterale $e_y(t)$ in metri (con deviazione standard e inviluppo $\pm 5$ cm).
  * Errore di heading $e_\theta(t)$ in gradi.

---

## 3. Foto Reali e Risultati Sperimentali (Dataset e Robot)

*Queste immagini documentano il lavoro empirico sul campo e provengono dai tuoi file o registrazioni.*

### `fig:crop_dataset_samples` (Capitolo 5, riga 19)
* **Descrizione:** Campioni visivi dei dataset multi-coltura con le corrispondenti maschere semantiche.
* **Layout consigliato:** Griglia $3 \times 2$ (o $3 \times 3$):
  * Riga 1: Lattuga (Immagine RGB + Maschera a 4 classi / binaria).
  * Riga 2: Pomodoro (Immagine RGB + Maschera SAM).
  * Riga 3: Cetriolo (Immagine RGB + Maschera SAM).

### `fig:sam3_annotation_pipeline` (Capitolo 5, riga 81)
* **Descrizione:** Diagramma/screenshot del processo di annotazione assistita con SAM 3.
* **Elementi:** L'immagine agricola grezza, il prompt testuale impostato (`"central farm path"`, `"plants"`, `"plastic mulch film"`), la maschera multi-classe prodotta e il passaggio di pulizia/salvataggio.

### `fig:geometric_strips_projection` (Capitolo 6, riga 126)
* **Descrizione:** Visualizzazione del partizionamento a fasce orizzontali sulla maschera segmentata.
* **Elementi:** La maschera binaria del terreno navigabile con le $N=24$ fasce orizzontali sovrapposte e l'individuazione degli intervalli laterali $[x_L, x_R]$ su ciascuna fascia.

### `fig:robust_edge_fitting` (Capitolo 6, riga 242)
* **Descrizione:** Esempio visivo del fitting robusto con Cauchy-IRLS.
* **Elementi:** Punti bordo rilevati (con eventuali foglie sporgenti considerate outlier e pesate vicino a 0), le due rette stimate $x_L(y)$ e $x_R(y)$, e la retta mediana risultante $x_C(y)$.

### `fig:robot_camera_setup` (Capitolo 7, riga 39)
* **Descrizione:** Fotografia reale del robot Demetra impiegato nelle prove sperimentali.
* **Elementi:** Il robot a 4 ruote sterzanti (4WS) inquadrato nel campo/serra, evidenziando il supporto della telecamera RGB montata anteriormente, la quota da terra e l'inclinazione (pitch) verso il filare.

### `fig:segmentation_comparison_crops` (Capitolo 8, riga 31)
* **Descrizione:** Confronto qualitativo delle predizioni dei tre modelli su una stessa scena complessa.
* **Layout:** Immagine originale RGB, Ground Truth manuale, output FCN-ResNet50, output DeepLabV3-ResNet50, output LR-ASPP-MobileNetV3.

### `fig:geometric_extraction_comparison` (Capitolo 8, riga 58)
* **Descrizione:** Confronto tra l'estrazione geometrica proposta (Dual-Edge IRLS) e i metodi baseline tradizionali (RANSAC sui contorni, istogramma a colonne).
* **Elementi:** Dimostrare visivamente come in presenza di foglie sporgenti asimmetriche il baricentro dell'istogramma subisca una deviazione laterale, mentre il metodo IRLS disaccoppiato mantenga la corretta traiettoria centrale.
