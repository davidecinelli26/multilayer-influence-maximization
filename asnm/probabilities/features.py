"""Attributi dei nodi per la GNN su StackExchange: grado e forza su ciascun
livello (in logaritmo) e il profilo tematico pre-confine compresso con SVD
troncata. I tag bersaglio sono esclusi secondo la politica di `splits`.
"""
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD

from asnm.layers import Livelli
from asnm.profiles import costruisci as costruisci_profili
from asnm.splits import Divisione

DIMENSIONI_TEMATICHE = 32


def costruisci(liv: Livelli, div: Divisione | None = None,
               dimensioni: int = DIMENSIONI_TEMATICHE) -> np.ndarray:
    """Matrice n x d degli attributi, normalizzata."""
    t0 = time.time()

    strutturali = np.zeros((liv.n, 2 * len(liv.nomi) + 1), np.float32)
    for k, nome in enumerate(liv.nomi):
        a = liv.csr_simmetrica(nome)
        strutturali[:, 2 * k] = np.diff(a.indptr)                  # grado
        strutturali[:, 2 * k + 1] = np.asarray(a.sum(1)).ravel()   # forza
    strutturali[:, -1] = strutturali[:, 0] + strutturali[:, 2]     # grado unione
    strutturali = np.log1p(strutturali)

    profili = costruisci_profili(liv, per_verita=False, div=div)
    d = min(dimensioni, max(profili.matrice.shape[1] - 1, 1))
    tematici = TruncatedSVD(n_components=d, random_state=0).fit_transform(
        profili.matrice).astype(np.float32)
    scala = np.abs(tematici).max() or 1.0

    x = np.hstack([strutturali, tematici / scala]).astype(np.float32)
    print(f"[features] {liv.sito}: {x.shape[0]:,} nodi x {x.shape[1]} attributi "
          f"({strutturali.shape[1]} strutturali + {d} tematici) "
          f"in {time.time()-t0:.0f}s")
    return x
