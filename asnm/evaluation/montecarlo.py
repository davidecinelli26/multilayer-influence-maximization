"""Simulazione di Independent Cascade.

Ogni nodo appena attivato tenta una volta sola su ciascun vicino uscente e
riesce con probabilita' p(u,v). `diffusione` restituisce il numero medio di
attivati; `attivazione` la probabilita' stimata di attivazione di ciascun
nodo, che la validazione somma sui veri adottanti.
"""
import numpy as np
import scipy.sparse as sp


def _propaga(ind, ptr, dati, semi, rng) -> set:
    attivi = set(semi)
    pila = list(semi)
    while pila:
        x = pila.pop()
        a, b = ptr[x], ptr[x + 1]
        if a == b:
            continue
        for y in ind[a:b][rng.random(b - a) < dati[a:b]]:
            y = int(y)
            if y not in attivi:
                attivi.add(y)
                pila.append(y)
    return attivi


def attivazione(p: sp.csr_matrix, semi, simulazioni: int = 4000,
                seme: int = 77) -> np.ndarray:
    """pi(v) = probabilita' stimata che v venga attivato partendo dai seed."""
    rng = np.random.default_rng(seme)
    ind, ptr, dati = p.indices, p.indptr, p.data
    conteggio = np.zeros(p.shape[0], np.float64)
    semi = [int(s) for s in semi]
    for _ in range(simulazioni):
        attivi = _propaga(ind, ptr, dati, semi, rng)
        conteggio[np.fromiter(attivi, int, len(attivi))] += 1
    return conteggio / simulazioni


def diffusione(p: sp.csr_matrix, semi, simulazioni: int = 1200,
               seme: int = 99) -> float:
    """Numero medio di nodi attivati."""
    rng = np.random.default_rng(seme)
    ind, ptr, dati = p.indices, p.indptr, p.data
    semi = [int(s) for s in semi]
    totale = 0
    for _ in range(simulazioni):
        totale += len(_propaga(ind, ptr, dati, semi, rng))
    return totale / simulazioni


def analizza(p: sp.csr_matrix, semi, cid: np.ndarray,
             simulazioni: int = 2000, seme: int = 99):
    """Diffusione media, comunita' raggiunte in media e pi(v), in un solo
    passaggio di simulazioni."""
    rng = np.random.default_rng(seme)
    ind, ptr, dati = p.indices, p.indptr, p.data
    semi = [int(s) for s in semi]
    conteggio = np.zeros(p.shape[0], np.float64)
    totale = comunita = 0
    for _ in range(simulazioni):
        attivi = _propaga(ind, ptr, dati, semi, rng)
        indici = np.fromiter(attivi, int, len(attivi))
        conteggio[indici] += 1
        totale += len(attivi)
        comunita += len(set(cid[indici].tolist()))
    return totale / simulazioni, comunita / simulazioni, conteggio / simulazioni


def comunita_raggiunte(p: sp.csr_matrix, semi, cid: np.ndarray,
                       simulazioni: int = 1200, seme: int = 99) -> float:
    """Numero medio di comunita' distinte toccate dalla cascata."""
    rng = np.random.default_rng(seme)
    ind, ptr, dati = p.indices, p.indptr, p.data
    semi = [int(s) for s in semi]
    totale = 0
    for _ in range(simulazioni):
        attivi = _propaga(ind, ptr, dati, semi, rng)
        totale += len(set(cid[np.fromiter(attivi, int, len(attivi))].tolist()))
    return totale / simulazioni
