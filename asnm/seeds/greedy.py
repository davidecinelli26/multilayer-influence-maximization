"""Selezione dei seed: massima copertura degli insiemi RR con il vincolo di diversita'.

Il criterio di scelta a ogni passo e'

    s(v) = alpha * |R_v| / max_u |R_u|  +  (1 - alpha) * [c(v) non ancora toccata]

con |R_v| gli insiemi RR ancora scoperti che contengono v. Con alpha = 1 si
riottiene TIM+ esatto. Il premio di diversita' vale solo per le comunita'
con almeno `minimo` nodi.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class Selezione:
    semi: list = field(default_factory=list)
    guadagni: list = field(default_factory=list)   # insiemi RR coperti da ciascuno
    comunita: list = field(default_factory=list)   # comunita' aperta da ciascuno

    @property
    def comunita_distinte(self) -> int:
        return len({c for c in self.comunita if c >= 0})

    @property
    def copertura(self) -> int:
        return int(sum(self.guadagni))

    @property
    def senza_guadagno(self) -> int:
        """Seed scelti solo per la diversita', che non coprono nulla di nuovo."""
        return int(sum(1 for g in self.guadagni if g == 0))


MINIMO = 50
"""Dimensione minima di una comunita' perche' aprirla dia il premio di
diversita' (su StackExchange; 200 su Higgs). Senza soglia il vincolo spende
seed per aprire comunita' da pochi nodi."""


def seleziona(insiemi: sp.csc_matrix, k: int, alpha: float = 1.0,
              cid: np.ndarray | None = None,
              dimensioni: np.ndarray | None = None,
              minimo: int = MINIMO) -> Selezione:
    """`insiemi` e' (numero di insiemi RR) x (numero di nodi), in formato CSC.

    Se `dimensioni` e' dato, solo le comunita' con almeno `minimo` nodi
    danno il premio di diversita'.
    """
    if alpha < 1.0 and cid is None:
        raise ValueError("il vincolo di diversita' richiede una partizione")
    n_insiemi, n = insiemi.shape
    vivo = np.ones(n_insiemi, bool)
    scelti = Selezione()
    toccate: set = set()

    premiabile = None
    if dimensioni is not None and minimo > 1:
        premiabile = np.asarray(dimensioni)[cid] >= minimo

    for _ in range(k):
        copertura = insiemi.T.dot(vivo.astype(np.int32)).astype(np.float64)
        massimo = copertura.max()
        if massimo <= 0:
            break

        punteggio = alpha * (copertura / massimo)
        if alpha < 1.0:
            nuova = np.ones(n)
            if toccate:
                nuova[np.isin(cid, list(toccate))] = 0.0
            if premiabile is not None:
                nuova[~premiabile] = 0.0
            punteggio = punteggio + (1 - alpha) * nuova
        if scelti.semi:
            punteggio[np.array(scelti.semi, dtype=int)] = -np.inf

        v = int(punteggio.argmax())
        scelti.semi.append(v)
        scelti.guadagni.append(int(copertura[v]))
        if cid is not None:
            scelti.comunita.append(int(cid[v]))
            toccate.add(int(cid[v]))
        else:
            scelti.comunita.append(-1)
        vivo[insiemi[:, v].nonzero()[0]] = False

    return scelti


def per_grado(p: sp.csr_matrix, k: int) -> list:
    """Baseline: i k nodi di grado uscente piu' alto."""
    gradi = np.diff(p.indptr)
    return [int(i) for i in np.argsort(-gradi)[:k]]


def a_caso(n: int, k: int, seme: int = 0) -> list:
    return [int(i) for i in np.random.default_rng(seme).choice(n, k, replace=False)]
