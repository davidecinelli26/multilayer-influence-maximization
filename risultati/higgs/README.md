# Results on Higgs Twitter

The text outputs of every run, exactly as the code printed them.
Every number in the report can be found here.

| file | command |
|---|---|
| `accoppiamento.txt` | `python -m asnm.higgs.accoppiamento` |
| `fase1.txt` | `python -m asnm.higgs.fase1` |
| `fase2.txt` | `python -m asnm.higgs.fase2` |
| `regime.txt` | `python -m asnm.higgs.fase2 --regime` |
| `stazionarieta.txt` | `python -m asnm.higgs.stazionarieta` |
| `fase3.txt` | `python -m asnm.higgs.fase3` |
| `validazione-k10.txt`, `-k20`, `-k50` | `python -m asnm.higgs.validazione --k=10` (20, 50) |
| `adozione.txt` | `python -m asnm.higgs.adozione.esegui` |
| `adozione-relazionale.txt` | `python -m asnm.higgs.adozione.esegui_relazionale` |
| `im-pesata-k10.txt`, `-k20`, `-k50` | `python -m asnm.higgs.im_pesata --k=10` (20, 50) |

The main numbers:

- gate: structural excess +42% aggregated, **+73%** stratified by hour
  (StackExchange: +10% and −0%);
- Phase 1: `Q_tras` 0.6211 for the calibrated fusion (λ = 0.003) against 0.6196
  for the follower layer alone (p < 0.001) and 0.4694 for the conventional fusion;
  Leiden and Louvain equivalent (p = 0.194);
- Phase 2: the GNN beats the constant baseline by 86% in MSE; regime factor 1.00;
  percolation threshold between ×1.5 and ×3; non-stationarity factor 4.55;
- Phase 3: at k = 50 from 10 to 14 communities at zero cost, 18 for 0.27%, 23
  for 1.80%; at k = 10 diversity increases the spread by 0.54%;
- validation: out-degree beats TIM+ by 7.66% at k = 20 and by 3.34% at
  k = 50 (p ≤ 0.001); the diversity constraint at k = 20 costs −0.02% (p = 0.933);
- who will adopt: AUC 0.6047 → 0.6265 → 0.6498 → 0.6648 → 0.6735 → 0.6782
  (SIGN), five positive steps under Bonferroni; R-SIGN 0.6795 (p = 0.218,
  within the noise between seeds);
- weighted IM: against the degree +11.85% at k = 10 and +7.06% at k = 20 (p < 0.001),
  −1.01% at k = 50 (p = 0.727).
