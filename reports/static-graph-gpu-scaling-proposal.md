# Proposta benchmark DDP

## 1. Obiettivo

Misurare di quanto si riduce il tempo di training quando lo stesso workload passa da 1 a 2 e 4 GPU nello stesso nodo. Il confronto è tra Flux.jl e PyTorch eager su un modello convoluzionale statico, con batch globale fisso. Il risultato è una baseline riproducibile (throughput, speedup, efficienza, memoria, correttezza) per valutare i futuri miglioramenti di Flux DDP.

Due domande guidano l'esperimento:

1. **Strong scaling:** quanto si guadagna con 2 e 4 GPU a lavoro globale fisso?
2. **Confronto tra framework:** come si confrontano throughput assoluto ed efficienza di scaling tra Flux e PyTorch?

Una campagna precedente su MNIST ha misurato ≈1,89× su 4 GPU, ma descrive un modello molto più piccolo e un'implementazione precedente (senza le guardie numeriche attuali): non è predittiva di questo esperimento.

## 2. Setup del benchmark

- **Dataset:** Tiny ImageNet-200, risoluzione nativa 64×64, training da zero; 100.000 immagini di training e 10.000 di validazione.
- **Modello:** ResNet-18 "small-image stem" della ricetta di riferimento (zeyuanyin/tiny-imagenet), testa a 200 classi. Si parte da ResNet-18, non ResNet-50, per limitare l'effort di implementazione e revisione.
- **Normalizzazione:** GroupNorm-32 (adattamento esplicito, motivato in §3).
- **Optimizer:** SGD con momentum 0.9 (vicino alla ricetta documentata) oppure Adam (continuità con gli esperimenti Flux). Stesso optimizer nei due framework, da concordare. Con batch 512 la lr della ricetta (0.2 a batch 256) va riscalata di conseguenza (linear scaling: 0.4), con warm-up documentato.
- **Batch globale:** 512, fisso in tutte le configurazioni (proposta primaria; 256 resta il fallback se il pilot di memoria non lascia margine, §4).
- **Precisione:** FP32 con TF32 esplicitamente disattivato; niente mixed precision, gradient accumulation o activation checkpointing.
- **PyTorch:** eager, senza `torch.compile`/TorchScript/CUDA Graph; restano i kernel nativi (cuDNN/cuBLAS) e il DDP nativo con bucketing e overlap.
- **Flux:** implementazione DDP attuale, incluse le guardie numeriche esistenti.

Matrice prevista (un processo per GPU):

| Framework | GPU | Batch locale | Lavoro per configurazione |
|---|---:|---:|---|
| Flux / PyTorch | 1 | 512 | stesso numero di update |
| Flux / PyTorch | 2 | 256 | stesso numero di update |
| Flux / PyTorch | 4 | 128 | stesso numero di update |

Tutte le configurazioni vedono gli stessi batch globali e lo stesso ordine dei campioni. Il learning rate non scala con il numero di GPU, perché il batch globale non cambia. Rispetto a 256, il batch 512 raddoppia il compute per passo a fronte di una comunicazione DDP (all-reduce) ~costante: a 4 GPU il batch locale 128 satura meglio le A100 rispetto a 64 e migliora l'overlap comunicazione/backward.

## 3. Perché GroupNorm al posto di BatchNorm

BatchNorm in training normalizza ogni canale sulle statistiche del batch locale. Con batch globale 512 e 4 rank, ogni layer normalizza su 128 immagini invece che su 512: al variare del numero di GPU cambia la funzione di training, e le differenze sulle curve possono riflettere la normalizzazione, non il DDP. I buffer running e il loro broadcast non risolvono: la normalizzazione in training resta locale.

GroupNorm calcola le statistiche dentro ogni singolo campione: è quindi indipendente da come il batch globale è partizionato e non aggiunge comunicazioni. SyncBatchNorm ripristina le statistiche globali ma aggiunge comunicazione e amplia lo scope di implementazione; resta un'estensione futura possibile.

Il modello diventa quindi **"Tiny ImageNet ResNet-18, small-image stem, GroupNorm-32"**: un adattamento documentato della ricetta di riferimento, non una sua riproduzione esatta.

## 4. Memoria: stime e pilot

Stime analitiche (nessuna misura eseguita). Il modello ha ≈11,27 M parametri; lo stato persistente è replicato su ogni GPU (≈129 MiB con SGD, ≈172 MiB con Adam). Le attivazioni sono la voce dominante e la principale incognita: il proxy degli output di convoluzione è ≈9,4 MiB per campione. Il caso critico è il single-GPU a batch locale 512 (≈4,7 GiB di soli output), con scenari illustrativi di ≈9,5–18,9 GiB includendo stato Adam e 2–4× le attivazioni; a 4 GPU (locale 128) gli stessi scenari scendono a ≈2,5–4,9 GiB. Il picco reale include retention per il backward, workspace, buffer di comunicazione e contesto CUDA, ed è maggiore; questi numeri non sono limiti misurati.

Con batch globale fisso la parte dipendente dalle attivazioni scende circa come 1/N, mentre lo stato replicato resta costante. Un breve **pilot di memoria** a 1/2/4 GPU, dopo l'approvazione, misurerà il picco e confermerà il batch definitivo: **512 è la proposta primaria**, con 256 come fallback se il single-GPU non lascia margine sufficiente.

## 5. Correttezza e riproducibilità

Perché il confronto sia valido: stessi pesi iniziali condivisi tra i framework, stesso ordine globale dei campioni, stessa loss (cross-entropy a media), stesse equazioni dell'optimizer e stesso schedule; stessi batch locali e stesso numero di update, con media dei gradienti esplicita su world size. Previsti check numerici a tolleranza: ogni configurazione DDP contro il riferimento single-GPU a batch globale, poi Flux contro PyTorch. Le curve di loss danno contesto ma non provano la correttezza. L'ultimo batch incompleto è gestito con la stessa politica in tutte le configurazioni (a batch 512: 195 batch pieni e stessi 160 campioni scartati per epoca).

## 6. Piano di misura

Due viste separate: **training-only** (batch già sul device: forward, backward, comunicazione, update) ed **end-to-end** (include pipeline di input e trasferimenti host→device). Warm-up escluso dal timing, sincronizzazione ai confini, durata dal rank più lento, almeno 5 ripetizioni indipendenti per configurazione, ordine dei run alternato tra framework e GPU.

Metriche: throughput (campioni/s), tempo per update, speedup rispetto al single-GPU dello stesso framework, efficienza S/N, memoria di picco e accuratezza di validazione; mediane con dispersione. Ambiente, versioni, hash dei sorgenti e comandi esatti saranno registrati nel report.

## 7. Decisioni da confermare

| Decisione | Proposta di partenza |
|---|---|
| Dataset | Tiny ImageNet-200, 64×64 nativo |
| Modello | ResNet-18 small-image stem, da zero, 200 classi |
| Normalizzazione | GroupNorm-32, adattamento documentato |
| Optimizer | SGD momentum o Adam |
| Batch globale | 512, subordinato al pilot di memoria (256 come fallback) |
| Campagna | 6 configurazioni a lavoro globale fisso; timing più budget per le curve |
| Deliverable | report riproducibile: throughput, speedup, efficienza, memoria, correttezza |

Il trade-off principale è tra fedeltà alla ricetta documentata (BatchNorm) e normalizzazione indipendente dal numero di rank (GroupNorm). GroupNorm offre l'esperimento controllato; se si preferisce riprodurre la ricetta BatchNorm serve una politica esplicita (SyncBatchNorm oppure BatchNorm locale con confondimento documentato). Sul batch, il trade-off è tra efficienza del workload DDP (locale 128 a 4 GPU) e margine di memoria del single-GPU: per questo la conferma finale arriva dal pilot.

## Riferimenti essenziali

1. Zeyuan Yin, *tiny-imagenet*: ricetta ResNet-18 small-image stem, batch 256, SGD. <https://github.com/zeyuanyin/tiny-imagenet>
2. Yuxin Wu, Kaiming He, *Group Normalization*, 2018. <https://arxiv.org/abs/1803.08494>
3. PyTorch, *DistributedDataParallel*: bucketing, overlap e semantiche di sincronizzazione. <https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html>

*Versione estesa con tutte le fonti e i calcoli: `reports/static-graph-gpu-scaling-proposal-detailed.md`. Nota: la ricetta di riferimento [1] riporta 59,57%–60,50% top-1 a 50/100/200 epoche con BatchNorm; quei valori non sono una soglia di accettazione per l'esperimento GroupNorm.*
