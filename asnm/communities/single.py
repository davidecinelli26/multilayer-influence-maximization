"""Partizioni candidate con Louvain: su ciascun livello e sulla fusione pesata."""
import time

import numpy as np

from asnm.layers import Livelli

SEME = 1


def _a_vettore(comunita, n: int) -> np.ndarray:
    cid = np.empty(n, dtype=np.int32)
    for k, gruppo in enumerate(comunita):
        for x in gruppo:
            cid[x] = k
    return cid


def louvain(grafo, n: int, seme: int = SEME, risoluzione: float = 1.0) -> np.ndarray:
    import networkx as nx

    comunita = nx.community.louvain_communities(
        grafo, seed=seme, weight="weight", resolution=risoluzione
    )
    return _a_vettore(comunita, n)


def candidate(liv: Livelli) -> dict:
    """nome leggibile -> vettore delle comunita'."""
    fuori = {}
    for nome in liv.nomi:
        t0 = time.time()
        fuori[f"{nome} {liv.etichette.get(nome, '')}".strip()] = louvain(
            liv.grafo(nome), liv.n)
        print(f"[single] {nome}: Louvain in {time.time()-t0:.0f}s")
    t0 = time.time()
    fuori["L1+L2 fusione"] = louvain(liv.fusione(), liv.n)
    print(f"[single] fusione: Louvain in {time.time()-t0:.0f}s")
    return fuori
