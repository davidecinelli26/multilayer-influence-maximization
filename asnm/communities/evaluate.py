"""I criteri con cui si valuta una partizione nella Fase 1.

Modularita' sulla rete di esposizione (criterio decisionale, corretto per il
caso per definizione), AMI col tag dominante degli utenti (criterio
tematico). NMI e rapporto intra/caso si calcolano solo per mostrare che
premiano le partizioni degeneri.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from asnm.profiles import Profili


def modularita(rete: sp.csr_matrix, cid: np.ndarray) -> float:
    """Modularita' di una partizione su un grafo pesato non orientato:

        Q = intra/(2m) - somma_c (S_c/(2m))^2
    """
    a = rete.tocoo()
    due_m = a.data.sum()
    if due_m == 0:
        return float("nan")
    intra = a.data[cid[a.row] == cid[a.col]].sum()
    gradi = np.asarray(rete.sum(axis=1)).ravel()
    forza = np.bincount(cid, weights=gradi, minlength=cid.max() + 1)
    return float(intra / due_m - np.sum((forza / due_m) ** 2))


def statistiche_intra(rete: sp.csr_matrix, cid: np.ndarray) -> tuple[float, float]:
    """Quota di esposizioni intra-comunita', e quanto darebbe un'assegnazione
    casuale a comunita' delle stesse dimensioni (un nullo che ignora i gradi)."""
    a = rete.tocoo()
    quota = float(a.data[cid[a.row] == cid[a.col]].sum() / a.data.sum())
    coperti = cid[np.diff(rete.indptr) > 0]
    dimensioni = np.bincount(coperti)
    p_caso = float(np.sum((dimensioni / len(coperti)) ** 2))
    return quota, p_caso


@dataclass
class Valutazione:
    nome: str
    comunita: int
    ami: float
    nmi: float
    q_esp: float
    kappa: float
    rapporto: float

    def riga(self) -> str:
        r = "inf" if not np.isfinite(self.rapporto) else f"{self.rapporto:.2f}x"
        return (f"{self.nome:22s} {self.comunita:>6,} {self.ami:>8.4f} "
                f"{self.nmi:>8.4f} {self.q_esp:>9.4f} {self.kappa:>8.4f} {r:>9s}")

    @staticmethod
    def intestazione() -> str:
        return (f"{'partizione':22s} {'com.':>6s} {'AMI':>8s} {'NMI':>8s} "
                f"{'Q_esp':>9s} {'kappa':>8s} {'rapporto':>9s}\n"
                f"{'':22s} {'':>6s} {'decis.':>8s} {'(rotta)':>8s} "
                f"{'DECIS.':>9s} {'':>8s} {'(rotta)':>9s}")


def valuta(nome: str, cid: np.ndarray, rete: sp.csr_matrix,
           profili: Profili, min_usi: int = 5) -> Valutazione:
    cid = np.asarray(cid)
    _, etichette = np.unique(cid, return_inverse=True)   # compatta gli id

    nodi, tema = profili.dominante(min_usi)
    if len(nodi) >= 100:
        mia = etichette[nodi]
        ami = adjusted_mutual_info_score(mia, tema)
        nmi = normalized_mutual_info_score(mia, tema)
    else:
        ami = nmi = float("nan")

    q = modularita(rete, etichette)
    quota, caso = statistiche_intra(rete, etichette)
    kappa = (quota - caso) / (1 - caso) if caso < 1 else float("nan")
    rapporto = quota / caso if caso > 0 else float("inf")
    return Valutazione(nome, len(set(etichette.tolist())), ami, nmi, q, kappa, rapporto)


def controllo_singoletti(rete: sp.csr_matrix, profili: Profili, n: int) -> Valutazione:
    """Il controllo a singoletti: Q_esp e AMI devono valere zero."""
    v = valuta("(controllo) singoletti", np.arange(n), rete, profili)
    v.nome = "(controllo) singoletti"
    return v
