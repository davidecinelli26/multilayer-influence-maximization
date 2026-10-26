"""Credit distribution su StackExchange: p(u,v) stimata dai log di adozione.

Variante statica di Bernoulli (Goyal, Bonchi, Lakshmanan, 2010):

    p(u -> v) = A(u -> v) / (A(u) + s)

con A(u) le adozioni di u prima del confine, A(u -> v) i tag che v ha
adottato dopo u, e s uno smoothing sul denominatore. I tag bersaglio sono
esclusi secondo la politica di `splits`.
"""
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from asnm.layers import Livelli
from asnm.splits import Divisione, dividi

SMOOTHING = 5.0   # al denominatore: contrae le stime basate su poche adozioni


@dataclass
class Stima:
    u: np.ndarray            # archi non orientati, estremo minore
    v: np.ndarray            # estremo maggiore
    ricco: np.ndarray        # l'arco ha almeno un evento utilizzabile?
    p_uv: np.ndarray         # p(u -> v), definita solo sugli archi ricchi
    p_vu: np.ndarray         # p(v -> u)
    adozioni: np.ndarray     # A(x) per nodo, per diagnostica
    n: int

    @property
    def quota_ricchi(self) -> float:
        return float(self.ricco.mean())

    def p_ricche(self) -> np.ndarray:
        """Tutte le p stimate, in un vettore solo."""
        return np.concatenate([self.p_uv[self.ricco], self.p_vu[self.ricco]])

    def matrice(self, p_poveri=None, scala: float = 1.0) -> sp.csr_matrix:
        """Matrice diretta n x n con p(u,v) su ogni arco.

        `p_poveri` sono i valori da usare sugli archi senza eventi: un
        vettore lungo `2 * (numero di archi poveri)` -- prima le direzioni
        u->v, poi le v->u -- oppure uno scalare.
        """
        poveri = ~self.ricco
        n_poveri = int(poveri.sum())
        if p_poveri is None:
            p_poveri = float(np.median(self.p_uv[self.ricco]))
        valori_poveri = (np.full(2 * n_poveri, float(p_poveri))
                         if np.isscalar(p_poveri) else np.asarray(p_poveri))
        if len(valori_poveri) != 2 * n_poveri:
            raise ValueError(
                f"p_poveri ha {len(valori_poveri)} valori, ne servono {2*n_poveri}")

        sorgenti = np.concatenate([self.u[self.ricco], self.v[self.ricco],
                                   self.u[poveri], self.v[poveri]])
        destinazioni = np.concatenate([self.v[self.ricco], self.u[self.ricco],
                                       self.v[poveri], self.u[poveri]])
        pesi = np.concatenate([self.p_uv[self.ricco], self.p_vu[self.ricco],
                               valori_poveri[:n_poveri],
                               valori_poveri[n_poveri:]]) * scala
        tieni = pesi > 1e-6
        return sp.csr_matrix(
            (pesi[tieni], (sorgenti[tieni], destinazioni[tieni])),
            shape=(self.n, self.n),
        )

    def riassunto(self) -> str:
        p = np.concatenate([self.p_uv[self.ricco], self.p_vu[self.ricco]])
        return (
            f"archi non orientati {len(self.u):,}: "
            f"ricchi {int(self.ricco.sum()):,} ({100*self.quota_ricchi:.1f}%), "
            f"poveri {int((~self.ricco).sum()):,} "
            f"({100*(1-self.quota_ricchi):.1f}%)\n"
            f"  p sui ricchi: media {p.mean():.4f}  mediana {np.median(p):.4f}  "
            f"p99 {np.percentile(p, 99):.4f}  max {p.max():.4f}  "
            f"nulla nel {100*(p == 0).mean():.1f}% dei casi"
        )


def _archi_unione(liv: Livelli):
    unione = sum((liv.csr_simmetrica(n) for n in liv.nomi),
                 sp.csr_matrix((liv.n, liv.n)))
    a = sp.triu(unione.tocsr(), k=1).tocoo()
    return a.row.astype(np.int32), a.col.astype(np.int32)


def stima(liv: Livelli, div: Divisione | None = None,
          smoothing: float = SMOOTHING) -> Stima:
    d = div or dividi(liv.sito)
    t0 = time.time()

    # adozioni pre-confine ammesse dalla politica, per nodo
    quando = defaultdict(dict)
    for (utente, tag), m in d.train.items():
        i = liv.indice.get(utente)
        if i is not None and d.ammesso_per_stima(tag):
            quando[i][tag] = m
    totali = np.zeros(liv.n, np.float64)
    for i, tg in quando.items():
        totali[i] = len(tg)

    u, v = _archi_unione(liv)
    p_uv = np.zeros(len(u), np.float64)
    p_vu = np.zeros(len(u), np.float64)
    ricco = np.zeros(len(u), bool)

    for k in range(len(u)):
        a, b = quando.get(int(u[k])), quando.get(int(v[k]))
        if not a or not b:
            continue
        # si itera sul dizionario piu' piccolo: i profili sono molto sbilanciati
        piccolo, grande, scambiati = (a, b, False) if len(a) <= len(b) else (b, a, True)
        dopo_v = dopo_u = 0
        for tag, m1 in piccolo.items():
            m2 = grande.get(tag)
            if m2 is None or m1 == m2:
                continue
            primo_e_piccolo = (m1 < m2) != scambiati
            if primo_e_piccolo:
                dopo_v += 1
            else:
                dopo_u += 1
        if dopo_u or dopo_v:
            ricco[k] = True
            p_uv[k] = dopo_v / (totali[u[k]] + smoothing)
            p_vu[k] = dopo_u / (totali[v[k]] + smoothing)

    print(f"[credit] {liv.sito}: stima su {len(u):,} archi in {time.time()-t0:.0f}s")
    return Stima(u, v, ricco, p_uv, p_vu, totali, liv.n)


if __name__ == "__main__":
    import sys

    from asnm.config import SITO_PRINCIPALE
    from asnm.layers import costruisci as costruisci_livelli
    from asnm.splits import MASCHERA_SEVERA

    sito = sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE
    liv = costruisci_livelli(sito)
    for s in (0.0, SMOOTHING):
        st = stima(liv, smoothing=s)
        print(f"\nsmoothing s = {s}  (maschera severa: "
              f"{'attiva' if MASCHERA_SEVERA else 'no'})")
        print("  " + st.riassunto().replace("\n", "\n  "))
