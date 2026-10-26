"""La rete di esposizione su StackExchange.

Per ogni adozione di un tag bersaglio nella finestra di verifica (u usa il
tag x al mese m), ogni vicino w di u che aveva gia' adottato x prima di m e'
un'esposizione; l'arco (u, w) prende peso 1 per ciascuna. Le esposizioni
restano separate per tag per il bootstrap a grappoli.
"""
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from asnm.layers import Livelli
from asnm.splits import Divisione, dividi


def eventi_per_tag(liv: Livelli, div: Divisione | None = None):
    """Le singole esposizioni, con il tag da cui provengono (il grappolo
    del bootstrap). Restituisce (u, w, indice_tag, nomi_tag, adozioni)."""
    d = div or dividi(liv.sito)
    t0 = time.time()

    # quando ciascun nodo ha adottato ciascun bersaglio, in tutta la storia
    quando = defaultdict(dict)
    for (utente, tag), m in {**d.train, **d.test}.items():
        i = liv.indice.get(utente)
        if i is not None and tag in d.bersagli:
            quando[i][tag] = m

    nomi = sorted(d.bersagli)
    indice_tag = {t: k for k, t in enumerate(nomi)}
    vicini = liv.vicinato()
    u_, w_, t_ = [], [], []
    adozioni = 0
    for (utente, tag), m in d.test.items():
        if tag not in d.bersagli:
            continue
        u = liv.indice.get(utente)
        if u is None:
            continue
        adozioni += 1
        k = indice_tag[tag]
        for w in vicini[u]:
            precedente = quando.get(int(w), {}).get(tag)
            if precedente is not None and precedente < m:
                u_.append(u)
                w_.append(int(w))
                t_.append(k)

    print(f"[exposure] {liv.sito}: {len(u_):,} esposizioni da {adozioni:,} adozioni "
          f"bersaglio su {len(nomi)} tag ({time.time()-t0:.0f}s)")
    return (np.array(u_, np.int32), np.array(w_, np.int32),
            np.array(t_, np.int32), nomi, adozioni)


def costruisci(liv: Livelli, div: Divisione | None = None):
    """Matrice simmetrica dei pesi di esposizione, piu' il numero di esposizioni."""
    righe, colonne, _, _, _ = eventi_per_tag(liv, div)
    n = liv.n
    mezza = sp.csr_matrix(
        (np.ones(len(righe), np.float64), (righe, colonne)), shape=(n, n)
    )
    rete = (mezza + mezza.T).tocsr()   # simmetrica: l'esposizione non ha verso
    rete.setdiag(0)
    rete.eliminate_zeros()

    attivi = int((np.diff(rete.indptr) > 0).sum())
    print(f"[exposure] {liv.sito}: rete su {attivi:,} nodi e {rete.nnz//2:,} archi")
    return rete, len(righe)


if __name__ == "__main__":
    import sys

    from asnm.config import SITO_PRINCIPALE
    from asnm.layers import costruisci as costruisci_livelli

    sito = sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE
    liv = costruisci_livelli(sito)
    rete, esposizioni = costruisci(liv)
    gradi = np.asarray(rete.sum(1)).ravel()
    coperti = gradi > 0
    print(f"\n  nodi con almeno un'esposizione : {coperti.sum():,} "
          f"({100*coperti.mean():.1f}% della rete)")
    print(f"  peso mediano per nodo coperto  : {np.median(gradi[coperti]):.0f}")
    print(f"  peso massimo su un arco        : {rete.data.max():.0f}")
