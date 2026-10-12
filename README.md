# InferSent Replication: Universal Sentence Representations from NLI Data

A PyTorch replication of **"Supervised Learning of Universal Sentence Representations from Natural Language Inference Data"** (Conneau, Kiela, Schwenk, Barrault, Bordes; EMNLP 2017).

The idea of the paper: train a sentence encoder on the supervised **Natural Language Inference (NLI)** task (SNLI), then **freeze** it and use its sentence vectors on many other tasks. The best encoder is a **BiLSTM with max-pooling** (called *InferSent*).

*Author: Janvi Porwal, B.Tech AI & DS (Final Year), Advanced Topics in ML mini project.*

---

## Results at a glance

Main model: BiLSTM-max, 4096-d sentence vectors, trained on SNLI, fixed GloVe 840B vectors. Trained on a Kaggle Tesla T4 GPU (about 2.9 hours, 12 epochs).

| Metric | Paper | This replication |
|---|---|---|
| SNLI dev accuracy | 85.0 | 83.9 |
| SNLI test accuracy | 84.5 | 84.1 |
| Transfer average, micro | 85.2 | 85.1 |
| Transfer average, macro | 83.7 | 84.0 |

### Transfer to 10 tasks (SentEval, logistic regression on frozen vectors)

| Task | Paper | Ours | Diff |
|---|---|---|---|
| MR | 79.9 | 78.2 | -1.7 |
| CR | 84.6 | 82.3 | -2.3 |
| SUBJ | 92.1 | 92.2 | +0.1 |
| MPQA | 89.8 | 88.5 | -1.3 |
| SST | 83.3 | 81.9 | -1.4 |
| TREC | 88.7 | 89.2 | +0.5 |
| MRPC (acc / F1) | 75.1 / 82.3 | 74.3 / 81.8 | -0.8 / -0.5 |
| SICK-R (Pearson) | 0.885 | 0.886 | +0.001 |
| SICK-E | 86.3 | 84.4 | -1.9 |
| STS14 (Pearson / Spearman) | .68 / .65 | .67 / .66 | -.01 / +.01 |

![Transfer accuracy: paper vs replication](results/figures/transfer_paper_vs_ours.png)

**Reading the numbers.** The averages and 6 of the 10 tasks are within about 1 point of the paper. The biggest gaps are CR (-2.3), SICK-E (-1.9), MR (-1.7), SST (-1.4) and MPQA (-1.3). Two runs of the same code gave results that differed by about 1 point on some tasks (for example SST 83.1 vs 81.9 and SICK-E 85.9 vs 84.4), so small gaps are within run-to-run noise. The CR gap was present in both runs and is larger than that noise; its cause is not known.

### SNLI training and errors

![Training curves](results/figures/training_curves.png)

Train accuracy reaches 96.6% while dev accuracy stays near 84%, so the model overfits the training pairs. The best dev accuracy (83.88) was reached at epoch 9.

Confusion matrix on the SNLI test set (rows = true label, columns = predicted):

| True \ Predicted | Entailment | Neutral | Contradiction | Correct |
|---|---|---|---|---|
| Entailment | 2923 | 349 | 96 | 86.8% |
| Neutral | 327 | 2545 | 347 | 79.1% |
| Contradiction | 143 | 304 | 2790 | 86.2% |

Neutral is the hardest class. Overall: 8,258 of 9,824 correct (84.1%).

### What does max-pooling select?

Each of the 4096 dimensions picks one word position. The plots compare the trained encoder with an untrained one (random weights), as in the appendix of the paper.

![Words selected by max-pooling](results/figures/maxpool_selection.png)
![Position of selected words](results/figures/maxpool_position.png)

---

## What is implemented

| Paper item | Status |
|---|---|
| SNLI training (Sec. 3.3 recipe), BiLSTM-max, Table 4 transfer row | **Run, results above** |
| Table 3: 7 encoders (LSTM, GRU, BiGRU-last, BiLSTM-mean, inner-attention, HConvNet, BiLSTM-max) | Code included; **only BiLSTM-max was run** |
| Fig. 5: embedding-size study | Code included; **not run** |
| Table 4: AllNLI and SST rows, untrained and bag-of-words baselines | Code included; **not run** |
| Table 5: COCO caption/image retrieval | Code included; **not run** |
| Sec. 5.1: Adam vs SGD | Switch included; **not run** |
| Appendix: max-pool analysis (trained vs untrained) | **Run** |

The parts marked "not run" were written from the paper's description but have not been executed, so treat them as untested. Rows of the paper's tables that come from other works (SkipThought, FastSent, SIF, ...) are only listed as reference rows in the notebook; they are not re-run.

## Method in short

1. Both sentences go through the **same encoder** and give vectors `u` (premise) and `v` (hypothesis).
2. The classifier input is `[u, v, |u - v|, u * v]`, followed by a layer of 512 units and 3 outputs (entailment / neutral / contradiction).
3. **BiLSTM-max encoder:** fixed GloVe vectors (300-d) -> bidirectional LSTM (2 x 2048) -> element-wise max over the words -> 4096-d sentence vector.
4. **Training (as in the paper):** SGD, learning rate 0.1, multiplied by 0.99 every epoch, divided by 5 when dev accuracy does not improve, batch size 64, stop when learning rate < 1e-5, GloVe frozen.
5. **Transfer:** the encoder is frozen; SentEval trains a logistic regression on its vectors for 10 tasks (MR, CR, SUBJ, MPQA, SST, TREC, MRPC, SICK-R, SICK-E, STS14).

### Differences from the paper (honest list)

- **Learning-rate rule:** the paper says the rate is divided by 5 if dev accuracy *decreases*; this code divides it when dev accuracy does not beat the best so far (option `LR_RULE='prev'` gives the paper's wording).
- **Classifier:** the paper says one hidden layer of 512 units; the default here uses stacked linear layers of 512 units (option `FC_STYLE='mlp1'` gives one hidden layer with tanh).
- **Mixed precision (fp16)** is used for speed (`USE_AMP`).
- Tokenisation uses NLTK's Treebank tokenizer; words without a GloVe vector are dropped.
- HConvNet channel widths and the inner-attention sizing are not fully specified in the paper; choices are documented in the notebook.
- One seed (1234) per run.

## How to run

### Kaggle (recommended)
1. Create a notebook, upload `notebooks/InferSent_Full_Replication_Kaggle.ipynb`.
2. Settings: **Accelerator = GPU (T4)**, **Internet = On**.
3. Run all cells (or *Save Version -> Save & Run All*). The notebook downloads SNLI, GloVe 840B (about 2 GB) and SentEval's data by itself. Expect about 5 hours for the main run.
4. Results (`results_<TAG>.json`, tables, figures, checkpoints) are written to `outputs/`.

### Google Colab
Select a GPU runtime, mount Google Drive and set `OUT_DIR` to a Drive folder so checkpoints survive disconnects. Everything else is the same.

### Running other experiments
Each run is one notebook with different settings in the `CFG` block at the top:

| Experiment | Setting |
|---|---|
| Another encoder (Table 3) | `ARCH = 'LSTM'`, `'GRU'`, `'BiGRU-last'`, `'BiLSTM-Mean'`, `'Inner-attention'`, `'HConvNet'` |
| Embedding size (Fig. 5) | `ENC_DIM = 512 / 1024 / 2048` |
| Train on AllNLI (SNLI + MultiNLI) | `TRAIN_DATA = 'AllNLI'` |
| Train on SST | `TRAIN_DATA = 'SST'` |
| COCO retrieval (Table 5) | `RUN_COCO = True` |
| Untrained / bag-of-words baselines | `SKIP_MAIN = True`, `RUN_BASELINES = True` |
| Adam instead of SGD | `OPTIMIZER = 'adam'` |
| Paper-literal classifier and lr rule | `FC_STYLE = 'mlp1'`, `LR_RULE = 'prev'` |
| Dropout (not in the paper) | `DROPOUT = 0.1` to `0.3` |
| Concatenated embeddings | `RUN_ENSEMBLE = True` |

The last section of the notebook collects all `results_*.json` files into tables next to the paper's numbers.

### Compatibility fixes for SentEval
SentEval is an old library; the notebook patches two issues automatically: `inspect.getargspec` (removed in Python 3.11) and ragged `np.array` calls in `sts.py` (numpy >= 1.24). The COCO retrieval part is my own GPU implementation of the SentEval protocol.

## Repository structure

```
infersent-replication/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── InferSent_Full_Replication_Kaggle.ipynb   (the executed notebook, outputs kept)
├── results/
│   ├── tables/
│   │   ├── table3_comparison.csv
│   │   ├── table4_full.csv
│   │   ├── table5_full.csv
│   │   └── improvement_summary.csv
│   └── figures/
│       ├── training_curves.png
│       ├── transfer_paper_vs_ours.png
│       ├── maxpool_selection.png
│       └── maxpool_position.png
└── docs/
    └── InferSent_Presentation.pptx
```

Model checkpoints (`*.pt`, about 200+ MB each) and the datasets are **not** stored in this repo (GitHub limits files to 100 MB); they are re-created by running the notebook.

## References

- A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes. *Supervised Learning of Universal Sentence Representations from Natural Language Inference Data.* EMNLP 2017, pages 670-680. [arXiv:1705.02364](https://arxiv.org/abs/1705.02364)
- Original code: [facebookresearch/InferSent](https://github.com/facebookresearch/InferSent) and the evaluation toolkit [facebookresearch/SentEval](https://github.com/facebookresearch/SentEval)
- SNLI (Bowman et al., 2015), MultiNLI (Williams et al., 2017), GloVe (Pennington et al., 2014)

```bibtex
@inproceedings{conneau2017supervised,
  title     = {Supervised Learning of Universal Sentence Representations from Natural Language Inference Data},
  author    = {Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Lo{\"\i}c and Bordes, Antoine},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages     = {670--680},
  year      = {2017},
  address   = {Copenhagen, Denmark}
}
```

## Acknowledgements
Built for the *Advanced Topics in ML* mini project. All credit for the method goes to the paper's authors; this repository is an independent re-implementation for learning purposes.
