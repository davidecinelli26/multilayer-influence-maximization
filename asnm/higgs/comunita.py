"""Le partizioni candidate su Higgs (Fase 1), con Leiden e Louvain via igraph.

La fusione calibrata non si calcola qui: la produce `accoppiamento.py`, che
sceglie il peso degli strati di interazione sulle trasmissioni precedenti al
taglio. Le partizioni si calcolano una alla volta e finiscono su disco.
"""
import time

import numpy as np
import scipy.sparse as sp

SEME = 1


def _igraph(rete: sp.csr_matrix):
    """Grafo igraph non orientato e pesato, costruito direttamente dalla
    matrice sparsa (da una lista di tuple Python non finirebbe in tempi utili)."""
    import igraph as ig

    alto = sp.triu(rete, k=1).tocsr()
    return ig.Graph.Weighted_Adjacency(alto, mode="undirected", attr="weight")


def _etichette(partizione, n: int) -> np.ndarray:
    fuori = np.empty(n, np.int32)
    for k, membri in enumerate(partizione):
        fuori[membri] = k
    return fuori


def leiden(rete: sp.csr_matrix, seme: int = SEME, iterazioni: int = 2):
    import leidenalg as la

    g = _igraph(rete)
    p = la.find_partition(g, la.ModularityVertexPartition, weights="weight",
                          seed=seme, n_iterations=iterazioni)
    return _etichette(p, rete.shape[0])


def louvain(rete: sp.csr_matrix, seme: int = SEME):
    g = _igraph(rete)
    p = g.community_multilevel(weights="weight")
    return _etichette(p, rete.shape[0])


def _rete(liv, nome: str) -> sp.csr_matrix:
    """La rete su cui girare l'algoritmo, in float32."""
    if "calibrata" in nome:
        raise RuntimeError(
            "la fusione calibrata si costruisce in asnm/higgs/accoppiamento.py, "
            "che sceglie il coefficiente degli strati dentro la finestra di "
            "addestramento: eseguirlo prima, la partizione finisce in cache")
    m = fusione(liv) if nome.startswith("fusione") else liv.csr_simmetrica(
        nome.split()[0])
    return m.astype(np.float32)


def fusione(liv, nomi=None) -> sp.csr_matrix:
    """Somma dei livelli, ciascuno normalizzato per il proprio peso totale
    (la normalizzazione convenzionale: con essa un retweet pesa circa
    cinquecento archi di follow)."""
    nomi = liv.nomi if nomi is None else nomi
    tot = sp.csr_matrix((liv.n, liv.n))
    for nome in nomi:
        a = liv.csr_simmetrica(nome).astype(np.float64)
        s = a.sum()
        if s:
            tot = tot + a / s
    return tot.tocsr()


CANDIDATE = ("F follower", "RT retweet", "MT mention", "fusione 4 livelli",
             "fusione calibrata", "F Louvain")
"""Le candidate della Fase 1. `fusione calibrata` la produce `accoppiamento.py`."""


def partizione(liv, nome: str, cache=None) -> np.ndarray:
    """Una sola partizione, calcolata e messa su disco; la matrice viene
    liberata subito dopo."""
    import gc

    if cache is not None:
        percorso = cache / f"higgs_part_{nome.replace(' ', '-')}.npy"
        if percorso.exists():
            return np.load(percorso)

    t0 = time.time()
    rete = _rete(liv, nome)
    cid = louvain(rete) if nome.endswith("Louvain") else leiden(rete)
    del rete
    gc.collect()
    print(f"[comunita] {'Louvain' if nome.endswith('Louvain') else 'Leiden'} "
          f"su {nome:18s} {len(np.unique(cid)):>7,} comunita'  "
          f"[{time.time()-t0:.0f}s]")
    if cache is not None:
        np.save(percorso, cid)
    return cid


def candidate(liv, cache=None, nomi=CANDIDATE) -> dict:
    """Le partizioni da mettere a confronto nella Fase 1."""
    return {n: partizione(liv, n, cache) for n in nomi}
