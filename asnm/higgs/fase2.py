"""Fase 2 su Higgs, dall'inizio alla fine.

    python -m asnm.higgs.fase2 [--epoche=N]
    python -m asnm.higgs.fase2 --regime      (rifa' solo la taratura)

Stima p con la credit distribution, addestra le due varianti di GraphSAGE,
tara il regime sulla p che verra' usata davvero e salva tutto in cache.
"""
import sys

import numpy as np

from asnm.config import CACHE
from asnm.higgs import gnn, probabilita
from asnm.higgs.dati import TAGLIO_ORE, costruisci
from asnm.probabilities import regime

PERCORSO = CACHE / f"higgs_probabilita_{TAGLIO_ORE}h.npz"

SCALE = (16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.55)
"""Griglia dei fattori di scala da esplorare; si percorre in salita e ci si
ferma al primo punto sopracritico."""


def esegui(epoche: int = gnn.EPOCHE) -> dict:
    liv, fin = costruisci()
    st = probabilita.stima(liv, fin)
    print("\n" + st.riassunto())

    x = probabilita.attributi(liv, fin)
    print()
    p_uniforme, d_uniforme = gnn.addestra(st, x, pesata=False, epoche=epoche)
    p_pesata, d_pesata = gnn.addestra(st, x, pesata=True, epoche=epoche)

    # la taratura viene dopo la GNN: la scala va scelta sulla matrice che si
    # usera' davvero, non sulla mediana costante
    print(f"\nTARATURA DEL REGIME (sulla p della GNN, non sulla mediana)\n"
          f"{'fattore':>8s} {'p media':>10s} {'RR medio':>11s}  regime")
    punti = regime.esplora(st, scale=SCALE, campione=1200, p_poveri=p_uniforme)
    for punto in punti:
        print(punto.riga())
    scala = regime.scegli(punti)
    print(f"\nfattore scelto: {scala}.")

    meglio = 100 * (1 - d_uniforme["mse"] / d_uniforme["mse_costante"])
    delta_hub = 100 * (1 - d_pesata["mse_hub"] / d_uniforme["mse_hub"])
    print(f"\nLa GNN batte la costante del {meglio:.0f}% di MSE.")
    if delta_hub > 1:
        print(f"La loss pesata migliora sugli archi verso gli hub del "
              f"{delta_hub:.1f}%: l'ipotesi regge sulla predizione, resta da "
              f"vedere se arriva ai seed.")
    else:
        print(f"La loss pesata NON migliora sugli hub ({delta_hub:+.1f}%): "
              f"l'ipotesi e' respinta gia' sulla predizione.")

    np.savez(PERCORSO, src=st.src, dst=st.dst, ricco=st.ricco, p=st.p,
             attivita=st.attivita, n=np.array([st.n]), scala=np.array([scala]),
             p_gnn_uniforme=p_uniforme, p_gnn_pesata=p_pesata,
             p_costante=np.full(int((~st.ricco).sum()),
                                float(np.median(st.p_ricche())), np.float32))
    print(f"\ncache scritta in {PERCORSO.name} "
          f"({PERCORSO.stat().st_size/1e6:.0f} MB)")
    return {"stima": st, "scala": scala, "attributi": x,
            "diagnostica": {"uniforme": d_uniforme, "pesata": d_pesata}}


def carica():
    """(StimaDiretta, scala, varianti di p sui poveri) dalla cache."""
    d = np.load(PERCORSO, allow_pickle=False)
    st = probabilita.StimaDiretta(d["src"], d["dst"], d["ricco"], d["p"],
                                  d["attivita"], int(d["n"][0]))
    varianti = {"gnn uniforme": d["p_gnn_uniforme"],
                "gnn pesata": d["p_gnn_pesata"],
                "costante": d["p_costante"]}
    return st, float(d["scala"][0]), varianti


def ritara(campione: int = 1500) -> float:
    """Rifa' solo la taratura del regime e aggiorna la cache (la scala non
    entra nell'addestramento della GNN)."""
    d = dict(np.load(PERCORSO, allow_pickle=False))
    st = probabilita.StimaDiretta(d["src"], d["dst"], d["ricco"], d["p"],
                                  d["attivita"], int(d["n"][0]))
    print(f"{'fattore':>8s} {'p media':>10s} {'RR medio':>11s}  regime")
    punti = regime.esplora(st, scale=SCALE, campione=campione,
                           p_poveri=d["p_gnn_uniforme"])
    for p in punti:
        print(p.riga())
    scala = regime.scegli(punti)
    d["scala"] = np.array([scala])
    np.savez(PERCORSO, **d)
    print(f"\nfattore scelto: {scala} (cache aggiornata)")
    return scala


if __name__ == "__main__":
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    if "--regime" in sys.argv[1:]:
        ritara(int(opzioni.get("campione", 1500)))
    else:
        esegui(epoche=int(opzioni.get("epoche", gnn.EPOCHE)))
