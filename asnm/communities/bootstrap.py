"""Bootstrap appaiato a grappoli per la Fase 1.

L'unita' di ricampionamento e' il grappolo (il tag su StackExchange, l'ora
su Higgs): le esposizioni dello stesso grappolo sono la stessa cascata e non
sono indipendenti. La stessa replica alimenta tutte le partizioni e si
prendono i percentili della differenza. Il calcolo e' vettorizzato: per una
partizione la modularita' su un multinsieme di esposizioni e'

    Q = intra/E - somma_c (S_c / 2E)^2

con E le esposizioni, intra quelle con i due estremi nella stessa comunita'
e S_c quante volte la comunita' c compare come estremo; tutti e tre sono
somme sui grappoli, quindi una replica costa un prodotto matrice-vettore.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Preparata:
    """Aggregati per tag di una partizione: bastano a ricalcolare Q_esp."""
    intra: np.ndarray      # per tag, esposizioni intra-comunita'
    esposizioni: np.ndarray  # per tag, esposizioni totali
    contributi: sp.csr_matrix  # per tag x comunita', quante volte compare come estremo


def prepara(cid: np.ndarray, u: np.ndarray, w: np.ndarray,
            tag: np.ndarray, n_tag: int) -> Preparata:
    _, etichette = np.unique(cid, return_inverse=True)
    cu, cw = etichette[u], etichette[w]
    n_com = etichette.max() + 1

    intra = np.bincount(tag[cu == cw], minlength=n_tag).astype(np.float64)
    esposizioni = np.bincount(tag, minlength=n_tag).astype(np.float64)

    # ogni esposizione mette due estremi in gioco: quello di u e quello di w
    righe = np.concatenate([tag, tag])
    colonne = np.concatenate([cu, cw])
    contributi = sp.csr_matrix(
        (np.ones(len(righe), np.float64), (righe, colonne)),
        shape=(n_tag, n_com),
    )
    return Preparata(intra, esposizioni, contributi)


def q_da_pesi(p: Preparata, pesi: np.ndarray) -> float:
    """Q_esp quando ogni tag compare `pesi[t]` volte."""
    e = float(pesi @ p.esposizioni)
    if e == 0:
        return float("nan")
    intra = float(pesi @ p.intra)
    forza = p.contributi.T @ pesi          # S_c
    return intra / e - float(np.sum((forza / (2 * e)) ** 2))


@dataclass
class Esito:
    nome: str
    osservato: float
    differenza: float
    basso: float
    alto: float
    p: float
    confronti: int = 1      # per la correzione di Bonferroni

    @property
    def soglia_corretta(self) -> float:
        return 0.05 / max(self.confronti, 1)

    @property
    def significativa(self) -> bool:
        return self.p < 0.05

    @property
    def regge_alla_correzione(self) -> bool:
        return self.p < self.soglia_corretta

    def riga(self) -> str:
        if not self.significativa:
            etichetta = "equivalenti"
        elif self.regge_alla_correzione:
            etichetta = "SIGNIFICATIVA"
        else:
            etichetta = f"al limite (cade a {self.soglia_corretta:.4f})"
        return (f"{self.nome:24s} {self.osservato:>8.4f} {self.differenza:>+9.4f} "
                f"  [{self.basso:>+8.4f}, {self.alto:>+8.4f}]  {self.p:>6.3f}  "
                f"{etichetta}")

    @staticmethod
    def intestazione() -> str:
        return (f"{'partizione':24s} {'Q_esp':>8s} {'differenza':>9s} "
                f"  {'IC 95% sulla differenza':^22s}  {'p':>6s}  esito")


def confronta(partizioni: dict, u, w, tag, n_tag: int,
              riferimento: str, repliche: int = 2000, seme: int = 7) -> list:
    """Bootstrap appaiato di tutte le partizioni contro una di riferimento."""
    preparate = {n: prepara(c, u, w, tag, n_tag) for n, c in partizioni.items()}
    unitari = np.ones(n_tag)
    osservati = {n: q_da_pesi(p, unitari) for n, p in preparate.items()}

    rng = np.random.default_rng(seme)
    campioni = {n: np.empty(repliche) for n in preparate}
    for b in range(repliche):
        scelti = rng.integers(0, n_tag, n_tag)
        pesi = np.bincount(scelti, minlength=n_tag).astype(np.float64)
        for n, p in preparate.items():
            campioni[n][b] = q_da_pesi(p, pesi)

    base = campioni[riferimento]
    confronti = len(preparate) - 1
    esiti = []
    for n in preparate:
        if n == riferimento:
            continue
        d = campioni[n] - base
        basso, alto = np.percentile(d, [2.5, 97.5])
        p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        esiti.append(Esito(n, osservati[n], osservati[n] - osservati[riferimento],
                           basso, alto, min(p, 1.0), confronti))
    esiti.insert(0, Esito(f"{riferimento}  (riferimento)", osservati[riferimento],
                          0.0, 0.0, 0.0, 1.0, confronti))
    return esiti
