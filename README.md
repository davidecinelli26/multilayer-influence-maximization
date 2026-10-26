# Influence maximization with a diversity constraint on a multilayer network

Given a social network observed before a time boundary, the project answers two questions: **whom to start from** to make a piece of information spread while reaching as many communities as possible (influence maximization with a diversity constraint), and **who will adopt** after the boundary (adoption prediction). The two parts close on each other: the prediction becomes the weight with which the optimizer picks the seeds.

## Authors

- [Davide Cinelli](https://github.com/davidecinelli26)
- [Giuseppe Cuiuri](https://github.com/GiuseppeCuiuri)

Developed for the *Social Network and Media Analysis* course (Prof. Andrea Tagarelli), MSc in Computer Engineering, DIMES, University of Calabria, A.Y. 2025/2026.

## Overview

The work runs on two datasets. A preliminary test, the *gate* (`cancello/`), measures whether on a network the neighbors matter beyond the degree (structural excess against a degree-preserving null model, stratified by time): on StackExchange it is zero, on Higgs Twitter +73%. Higgs is the working substrate, StackExchange stays as a negative control.

| | StackExchange | Higgs Twitter |
|---|---|---|
| channel | answers and comments | follower graph |
| nodes | 46,312 | 456,627 |
| edges | 230,641 | 14,855,842 |
| layers | 2 | 4 (follower, retweet, mention, reply) |
| object of diffusion | 105 technical tags | one event, July 2012 |
| boundary | January 2020 | hour 72 (night of July 3) |
| **structural excess** | **−0%** | **+73%** |

## The phases

1. **Gate** (`cancello/`): the structural excess on the two substrates.
2. **Phase 1, communities**: Leiden on the layers and on their fusion, with the weight of the interaction layers chosen on the transmissions before the boundary; the partition is judged out of sample with the modularity on the transmissions that happened after.
3. **Phase 2, probabilities**: `p(u,v)` estimated with credit distribution on the observed events and extended with a GraphSAGE to the edges without data; calibration of the regime with respect to the percolation threshold.
4. **Phase 3, seeds**: TIM with RR sketches and the diversity constraint of DTIM; budget × constraint weight surface.
5. **Validation**: the seeds against the true adopters after the boundary, with a paired cluster bootstrap.
6. **Who will adopt**: adoption classifier built as a ladder of feature sets (degree, exposure, community, spectrum, propagated, SIGN, R-SIGN), each step compared with the previous one.
7. **Prediction-weighted IM**: the roots of the RR sets drawn in proportion to the predicted probability of adopting, compared with the degree.

## Main results (Higgs Twitter)

- **Communities:** the calibrated fusion of the layers (λ = 0.003) reaches out-of-sample modularity `Q_tras` 0.6211, against 0.6196 for the follower layer alone (p < 0.001) and 0.4694 for the conventional fusion.
- **Probabilities:** the GNN beats the constant baseline by 86% in MSE.
- **Diversity:** at k = 50 the seeds go from 10 to 14 communities at zero cost, 18 for 0.27% of spread, 23 for 1.80%.
- **Validation:** against the true adopters, out-degree beats TIM+ by 7.66% at k = 20 and 3.34% at k = 50 (p ≤ 0.001); the diversity constraint at k = 20 costs −0.02% (p = 0.933).
- **Who will adopt:** AUC climbs from 0.6047 to 0.6782 (SIGN) with five positive steps under Bonferroni; R-SIGN (0.6795) stays within the noise between seeds.
- **Prediction-weighted IM:** +11.85% over the degree at k = 10 and +7.06% at k = 20 (p < 0.001), no difference at k = 50.

Every number, with the command that produced it, is in [`risultati/higgs/`](risultati/higgs/).

## Structure

```
asnm/                 the package
├── config.py         paths and constants
├── layers.py         the Livelli class (multilayer network), built from StackExchange
├── dumps.py          extraction of the XML files from the .7z archives  (StackExchange)
├── adoption.py       reading posts and tag adoptions                    (StackExchange)
├── splits.py         time boundary and target tags                      (StackExchange)
├── profiles.py       topical profiles of the users                      (StackExchange)
├── communities/      Phase 1                                            (StackExchange)
├── probabilities/    Phase 2: credit distribution, GraphSAGE, regime
├── seeds/            Phase 3: RR sets, greedy with diversity, competitive
├── evaluation/       Monte Carlo, bootstrap, validation                 (StackExchange)
└── higgs/            the whole pipeline on the Higgs substrate
    ├── dati.py             the four layers and the time window
    ├── comunita.py         candidate partitions (Leiden, Louvain)
    ├── accoppiamento.py    choice of the weight of the interaction layers
    ├── fase1.py            Q_tras and bootstrap over hours
    ├── probabilita.py      oriented credit distribution, node features
    ├── gnn.py              GraphSAGE on the edges
    ├── fase2.py            estimation, GNN, regime calibration
    ├── fase3.py            stability, ablation, k × alpha surface
    ├── stazionarieta.py    non-stationarity factor
    ├── validazione.py      the seeds against the true adopters
    ├── adozione/           the feature ladder (who will adopt)
    └── im_pesata.py        prediction-weighted IM
cancello/             the structural excess scripts on the two substrates
risultati/higgs/      the text outputs of every run on Higgs
relazione/            report and presentation (LaTeX, PDF, figure scripts)
app/                  Streamlit demo on the StackExchange substrate (optional)
data/raw/             the downloaded data          (not versioned)
data/cache/           intermediate artifacts       (not versioned)
vendor/               third-party code (M-EMCD)    (not versioned)
```

`seeds/greedy`, `seeds/rr`, `evaluation/montecarlo` and the two bootstraps do not depend on the substrate: they take a matrix `p` and a partition.

Module names, code comments, the report and the presentation are in Italian.

## How to run

```bash
pip install -r requirements.txt
```

**Higgs.** Download the two files from <https://snap.stanford.edu/data/> into `data/raw/higgs/`: `higgs-social_network.edgelist.gz` and `higgs-activity_time.txt.gz`. Then, in order:

```bash
python -m cancello.higgs_eccesso            # the gate
python -m asnm.higgs.dati                   # the four layers
python -m asnm.higgs.accoppiamento          # the layer weight (calibrated fusion)
python -m asnm.higgs.fase1                  # the partitions and the bootstrap
python -m asnm.higgs.fase2                  # credit distribution, GraphSAGE, regime
python -m asnm.higgs.stazionarieta          # the 4.55 factor
python -m asnm.higgs.fase3                  # stability, ablation, surface
python -m asnm.higgs.validazione --k=20     # validation (also --k=10, --k=50)
python -m asnm.higgs.adozione.esegui        # the feature ladder
python -m asnm.higgs.adozione.esegui_relazionale   # R-SIGN
python -m asnm.higgs.im_pesata --k=20       # weighted IM (also --k=10, --k=50)
```

Partitions, probabilities and RR sets end up in `data/cache/`: the first run of Phase 1 computes the partitions on twelve and a half million edges and takes about ten minutes, later runs read them back. The reference outputs are in `risultati/higgs/`.

**StackExchange** (negative control). The `.7z` archives of `security.stackexchange.com` and `computergraphics.stackexchange.com` (<https://archive.org/details/stackexchange>) go into `data/raw/`; the first command that touches a site extracts the XML files by itself.

```bash
python -m cancello.stackexchange_eccesso
python -m asnm.splits security
python -m asnm.communities.fase1 security --senza-emcd
python -m asnm.probabilities.fase2 security
python -m asnm.seeds.fase3 security
python -m asnm.evaluation.temporale security
```

Phase 1 on StackExchange also compares the **M-EMCD** ensemble consensus, distributed as a JAR by the research group (<http://people.dimes.unical.it/andreatagarelli/emcd/>). It is not redistributable and is not in the repository: download it separately into `vendor/`; it requires a JRE 8+ (set the `ASNM_JAVA` environment variable if `java` is not on the path). Without it, use `--senza-emcd`. Reference: A. Tagarelli, A. Amelio, F. Gullo, *Ensemble-based community detection in multilayer networks*, Data Mining and Knowledge Discovery 31(5):1506–1543, 2017.

## Reproducibility

Every experiment is a module runnable with `python -m`, with fixed seeds declared in the code (node split, model initialization, RR set sampling, Monte Carlo, bootstrap). The figures of the report are regenerated with `python relazione/figure.py` and `python relazione/schemi.py` from the outputs in `risultati/higgs/`.

## License

The code is under the [MIT](LICENSE) license. The datasets (SNAP Higgs Twitter, StackExchange data dump) and M-EMCD keep their own licenses.
