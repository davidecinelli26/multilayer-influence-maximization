"""Bootstrap appaiato a grappoli su punteggi additivi per unita' (tag o ora).

La stessa replica alimenta tutte le strategie e si prendono i percentili
della differenza rispetto al riferimento; il p-value e' a due code. La
soglia di significativita' e' corretta con Bonferroni per il numero di
confronti.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Confronto:
    nome: str
    totale: float
    differenza: float
    basso: float
    alto: float
    p: float
    confronti: int = 1
    base: float = 0.0

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
            esito = "equivalente"
        elif self.regge_alla_correzione:
            esito = "DIVERSA"
        else:
            esito = f"al limite (cade a {self.soglia_corretta:.4f})"
        rel = f"{100*self.differenza/self.base:+.2f}%" if self.base else "-"
        return (f"{self.nome:26s} {self.totale:>9.1f} {self.differenza:>+9.1f} "
                f"{rel:>8s}  [{self.basso:>+7.1f}, {self.alto:>+7.1f}] "
                f"{self.p:>6.3f}  {esito}")

    @staticmethod
    def intestazione() -> str:
        return (f"{'strategia':26s} {'punteggio':>9s} {'differenza':>9s} "
                f"{'relativa':>8s}  {'IC 95%':^19s} {'p':>6s}  esito")


def confronta(punteggi: dict, riferimento: str, repliche: int = 3000,
              seme: int = 3) -> list:
    """`punteggi` associa a ogni strategia un vettore di punteggi per unita'
    (tag o ora)."""
    nomi = list(punteggi)
    matrice = np.vstack([punteggi[n] for n in nomi])      # strategie x unita'
    n_tag = matrice.shape[1]
    totali = matrice.sum(axis=1)

    rng = np.random.default_rng(seme)
    scelte = rng.integers(0, n_tag, (repliche, n_tag))
    pesi = np.stack([np.bincount(r, minlength=n_tag) for r in scelte]).astype(float)
    campioni = pesi @ matrice.T                            # repliche x strategie

    i_rif = nomi.index(riferimento)
    confronti = len(nomi) - 1
    base = float(totali[i_rif])
    esiti = [Confronto(f"{riferimento}  (riferimento)", base,
                       0.0, 0.0, 0.0, 1.0, confronti, base)]
    for i, n in enumerate(nomi):
        if i == i_rif:
            continue
        d = campioni[:, i] - campioni[:, i_rif]
        basso, alto = np.percentile(d, [2.5, 97.5])
        p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        esiti.append(Confronto(n, float(totali[i]),
                               float(totali[i] - totali[i_rif]),
                               float(basso), float(alto), min(p, 1.0),
                               confronti, base))
    return esiti
