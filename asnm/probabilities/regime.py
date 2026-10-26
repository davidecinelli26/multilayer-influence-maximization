"""Taratura del regime della cascata.

Sotto Independent Cascade la diffusione ha una soglia di percolazione; sopra
soglia gli insiemi RR diventano enormi e il greedy sceglie fra candidati
equivalenti. Il controllo e' la dimensione media degli insiemi RR al variare
di un fattore di scala su p: si tiene il fattore piu' grande che la mantiene
fra MINIMA_UTILE e SOTTOCRITICO. `pareggia` porta due modelli di p diversi
alla stessa dimensione media, per confrontarli a parita' di regime.
"""
from dataclasses import dataclass

import numpy as np

from asnm.seeds import rr

SCALE = (1.0, 0.5, 0.25, 0.12, 0.06, 0.03)
SOTTOCRITICO = 30.0   # dimensione media degli insiemi RR: sopra, regime critico
MINIMA_UTILE = 3.0    # sotto, gli insiemi sono quasi tutti singoletti


@dataclass
class Punto:
    scala: float
    p_media: float
    rr_medio: float

    @property
    def regime(self) -> str:
        if self.rr_medio > 200:
            return "sopracritico"
        if self.rr_medio > SOTTOCRITICO:
            return "critico"
        if self.rr_medio < MINIMA_UTILE:
            return "quasi spento"
        return "sottocritico"

    def riga(self) -> str:
        return (f"{self.scala:>8.2f} {self.p_media:>10.5f} {self.rr_medio:>11.1f}  "
                f"{self.regime}")


def esplora(st, scale=SCALE, campione: int = 600, p_poveri=None) -> list:
    """Dimensione media degli insiemi RR per ogni fattore di scala, in salita
    fino al primo punto sopracritico. `p_poveri` deve essere la p che si
    usera' davvero."""
    if p_poveri is None:
        p_poveri = float(np.median(st.p_ricche()))
    punti = []
    for s in sorted(scale):
        matrice = st.matrice(p_poveri=p_poveri, scala=s)
        punti.append(Punto(s, float(matrice.data.mean()),
                           rr.dimensione_media(matrice, campione)))
        # sopra soglia un solo insieme RR puo' toccare centomila nodi
        if punti[-1].regime == "sopracritico":
            break
    return punti


def pareggia(st, p_poveri, bersaglio: float, campione: int = 400,
             tolleranza: float = 0.15, passi: int = 14) -> float:
    """La scala che porta gli insiemi RR alla dimensione media `bersaglio`,
    per confrontare due modelli di p a parita' di regime. Bisezione
    sull'esponente."""
    basso, alto = 1e-3, 32.0
    for _ in range(passi):
        mezzo = float(np.sqrt(basso * alto))
        d = rr.dimensione_media(st.matrice(p_poveri=p_poveri, scala=mezzo),
                                campione)
        if abs(d - bersaglio) <= tolleranza * bersaglio:
            return mezzo
        if d < bersaglio:
            basso = mezzo
        else:
            alto = mezzo
    return float(np.sqrt(basso * alto))


def scegli(punti: list) -> float:
    """Il fattore piu' grande che tiene il sistema sottocritico."""
    validi = [p for p in punti if MINIMA_UTILE <= p.rr_medio <= SOTTOCRITICO]
    if not validi:
        raise RuntimeError(
            "nessun fattore porta gli insiemi RR in un regime utilizzabile: "
            "allarga `SCALE`")
    return max(validi, key=lambda p: p.scala).scala


if __name__ == "__main__":
    import sys

    from asnm.config import SITO_PRINCIPALE
    from asnm.layers import costruisci as costruisci_livelli
    from asnm.probabilities.credit import stima

    sito = sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE
    liv = costruisci_livelli(sito)
    st = stima(liv)
    print("\n" + st.riassunto())
    print(f"\n{'fattore':>8s} {'p media':>10s} {'RR medio':>11s}  regime")
    punti = esplora(st)
    for p in punti:
        print(p.riga())
    print(f"\nfattore scelto: {scegli(punti)}")
