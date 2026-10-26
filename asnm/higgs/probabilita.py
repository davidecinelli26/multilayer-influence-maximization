"""Fase 2 su Higgs: credit distribution orientata e attributi dei nodi.

    p(u -> v) = trasmissioni(u -> v) / (attivita'(u) + s)

dove le trasmissioni sono gli eventi precedenti al taglio in cui v ha
ritwittato, menzionato o risposto a u, e l'attivita' di u e' il numero di
volte in cui u compare nel log prima del taglio. La stima e' definita solo
sugli archi di follow, nel verso del follow. Gli archi con almeno una
trasmissione sono i «ricchi»; sugli altri la p la estende la GNN.
"""
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

SMOOTHING = 5.0


@dataclass
class StimaDiretta:
    """La stima su archi orientati, con la stessa interfaccia della `Stima`
    di StackExchange: `matrice()`, `p_ricche()`, `riassunto()`."""

    src: np.ndarray          # influenzatore
    dst: np.ndarray          # influenzato
    ricco: np.ndarray        # l'arco ha almeno una trasmissione osservata?
    p: np.ndarray            # p(src -> dst), definita sui ricchi
    attivita: np.ndarray     # per nodo, comparse nel log pre-taglio
    n: int

    @property
    def quota_ricchi(self) -> float:
        return float(self.ricco.mean())

    def p_ricche(self) -> np.ndarray:
        return self.p[self.ricco]

    def matrice(self, p_poveri=None, scala: float = 1.0) -> sp.csr_matrix:
        """Matrice orientata n x n. `p_poveri` copre gli archi senza dati."""
        poveri = ~self.ricco
        n_poveri = int(poveri.sum())
        if p_poveri is None:
            p_poveri = float(np.median(self.p_ricche()))
        valori = (np.full(n_poveri, float(p_poveri))
                  if np.isscalar(p_poveri) else np.asarray(p_poveri, np.float64))
        if len(valori) != n_poveri:
            raise ValueError(
                f"p_poveri ha {len(valori)} valori, ne servono {n_poveri}")

        pesi = np.empty(len(self.src), np.float64)
        pesi[self.ricco] = self.p[self.ricco]
        pesi[poveri] = valori
        pesi *= scala
        tieni = pesi > 1e-6
        return sp.csr_matrix(
            (pesi[tieni], (self.src[tieni], self.dst[tieni])),
            shape=(self.n, self.n))

    def riassunto(self) -> str:
        pr = self.p_ricche()
        return (
            f"archi orientati {len(self.src):,}: ricchi {int(self.ricco.sum()):,} "
            f"({100*self.quota_ricchi:.2f}%)\n"
            f"  p sui ricchi: media {pr.mean():.4f}  mediana {np.median(pr):.4f}  "
            f"p99 {np.percentile(pr, 99):.4f}  max {pr.max():.4f}")


def stima(liv, fin, smoothing: float = SMOOTHING) -> StimaDiretta:
    """Credit distribution sulle trasmissioni osservate prima del taglio."""
    t0 = time.time()
    from asnm.higgs.dati import attivita as leggi_attivita

    a, b, ts, _ = leggi_attivita()
    prima = ts < fin.taglio
    a, b = a[prima], b[prima]

    # denominatore: quante volte il nodo compare nel log, in qualunque ruolo
    conta = np.bincount(np.concatenate([a, b]), minlength=liv.n).astype(np.float64)

    # numeratore: b -> a, da chi e' stato ritwittato a chi ha ritwittato.
    # L'incrocio con gli archi di follow si fa su chiavi intere ordinate.
    tenere = a != b
    chiavi_t, conteggi_t = np.unique(
        b[tenere].astype(np.int64) * liv.n + a[tenere], return_counts=True)

    src, dst, _ = liv.archi["F"]
    chiavi_f = src.astype(np.int64) * liv.n + dst
    posizione = np.searchsorted(chiavi_t, chiavi_f)
    dentro = posizione < len(chiavi_t)
    trovato = np.zeros(len(chiavi_f), bool)
    trovato[dentro] = chiavi_t[posizione[dentro]] == chiavi_f[dentro]
    osservate = np.zeros(len(chiavi_f), np.float64)
    osservate[trovato] = conteggi_t[posizione[trovato]]
    ricco = trovato
    p = np.zeros(len(src), np.float64)
    p[ricco] = osservate[ricco] / (conta[src[ricco]] + smoothing)

    fuori = int(len(chiavi_t) - ricco.sum())
    print(f"[credit-higgs] {int(ricco.sum()):,} archi di follow con "
          f"trasmissione osservata; {fuori:,} trasmissioni fuori dal grafo "
          f"di follow (si ritwitta anche chi non si segue)  "
          f"[{time.time()-t0:.0f}s]")
    return StimaDiretta(src, dst, ricco, p, conta, liv.n)


# ------------------------------------------------------------- attributi
def attributi(liv, fin, dimensioni_svd: int = 32) -> np.ndarray:
    """Gli attributi dei nodi, tutti calcolabili prima del taglio: gradi
    entranti e uscenti sui quattro livelli, attivita' pre-taglio (se e quando),
    e le dimensioni spettrali del grafo dei follower (SVD troncata)."""
    from sklearn.decomposition import TruncatedSVD

    t0 = time.time()
    pezzi = []
    for nome in liv.nomi:
        m = liv.csr(nome)
        pezzi.append(np.log1p(np.asarray(m.sum(1)).ravel()))
        pezzi.append(np.log1p(np.asarray(m.sum(0)).ravel()))

    attivo = fin.gia_attivi.astype(np.float32)
    ore = np.where(attivo > 0,
                   np.clip((fin.adozione - fin.t0) / 3600.0, 0, 72), 72.0)
    pezzi += [attivo, ore / 72.0]

    base = np.stack(pezzi, 1).astype(np.float32)

    unione = liv.csr_simmetrica("F")
    svd = TruncatedSVD(n_components=dimensioni_svd, random_state=0)
    spettrale = svd.fit_transform(unione).astype(np.float32)
    spettrale /= (np.abs(spettrale).max(0, keepdims=True) + 1e-9)

    x = np.hstack([base, spettrale])
    print(f"[attributi] {x.shape[1]} attributi per {x.shape[0]:,} nodi "
          f"({base.shape[1]} strutturali + {dimensioni_svd} spettrali), "
          f"varianza spiegata {svd.explained_variance_ratio_.sum():.3f}  "
          f"[{time.time()-t0:.0f}s]")
    return x
