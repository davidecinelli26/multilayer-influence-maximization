"""Scelta del peso degli strati di interazione nella fusione multistrato.

La rete di lavoro e'

    A(lambda) = A_F / |A_F|  +  lambda * somma_l A_l / |A_l|,   l in {RT, MT, RE}

e lambda si sceglie su una griglia massimizzando la modularita' della
partizione Leiden sulle trasmissioni avvenute prima del taglio. La finestra
di verifica non entra nella scelta: il giudizio fuori campione lo da' la
Fase 1 con Q_tras.
"""
import time

import numpy as np
import scipy.sparse as sp

from asnm.communities.evaluate import modularita
from asnm.config import CACHE
from asnm.higgs import comunita
from asnm.higgs.dati import costruisci

LAMBDA = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def rete(liv, lam: float) -> sp.csr_matrix:
    """A(lambda), in float32."""
    f = liv.csr_simmetrica("F").astype(np.float32)
    tot = f / f.sum()
    if lam > 0:
        for nome in ("RT", "MT", "RE"):
            a = liv.csr_simmetrica(nome).astype(np.float32)
            s = a.sum()
            if s:
                tot = tot + (lam / s) * a
    return tot.tocsr()


def trasmissioni_precut(liv, fin):
    """Le trasmissioni prima del taglio: la rete di trasmissione della Fase 1,
    confinata alla finestra di addestramento."""
    u, v, _ = liv.archi["F"]
    a = fin.adozione
    dentro = (a[v] < fin.taglio) & (a[u] < a[v])
    m = sp.csr_matrix((np.ones(int(dentro.sum()), np.float64),
                       (u[dentro], v[dentro])), shape=(liv.n, liv.n))
    return (m + m.T).tocsr(), int(dentro.sum())


def esegui(valori=LAMBDA) -> dict:
    liv, fin = costruisci()
    criterio, quanti = trasmissioni_precut(liv, fin)
    print(f"[accoppiamento] criterio di taratura: {quanti:,} trasmissioni "
          f"precedenti al taglio\n")
    print(f"{'lambda':>8s} {'comunita':>9s} {'Q pre-taglio':>13s}   (in campione)")

    esiti = {}
    for lam in valori:
        t0 = time.time()
        percorso = CACHE / f"higgs_part_accoppiamento_{lam}.npy"
        if percorso.exists():
            cid = np.load(percorso)
        else:
            cid = comunita.leiden(rete(liv, lam))
            np.save(percorso, cid)
        _, etichette = np.unique(cid, return_inverse=True)
        q = modularita(criterio, etichette)
        esiti[lam] = (etichette, q)
        print(f"{lam:>8.3f} {etichette.max()+1:>9,} {q:>13.4f}   "
              f"[{time.time()-t0:.0f}s]")

    migliore = max(esiti, key=lambda l: esiti[l][1])
    print(f"\nlambda scelto: {migliore}. La scelta e' fatta senza guardare la "
          f"finestra di verifica;\nil giudizio fuori campione lo da' la Fase 1.")
    np.save(CACHE / "higgs_part_fusione-calibrata.npy", esiti[migliore][0])
    return {"esiti": esiti, "lambda": migliore}


if __name__ == "__main__":
    esegui()
