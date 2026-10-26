"""Fase 2 su StackExchange, dall'inizio alla fine.

    python -m asnm.probabilities.fase2 [sito]

Stima p con la credit distribution, tara il regime, addestra le due
varianti di GraphSAGE e salva tutto in cache per la Fase 3.
"""
import sys

import numpy as np

from asnm.config import CACHE, SITO_PRINCIPALE
from asnm.layers import costruisci as costruisci_livelli
from asnm.probabilities import credit, features, gnn, regime
from asnm.splits import dividi


def _percorso(sito):
    return CACHE / f"probabilita_{sito}.npz"


def esegui(sito: str, epoche: int = gnn.EPOCHE) -> dict:
    liv = costruisci_livelli(sito)
    div = dividi(sito)

    st = credit.stima(liv, div)
    print("\n" + st.riassunto())

    print(f"\nTARATURA DEL REGIME\n{'fattore':>8s} {'p media':>10s} "
          f"{'RR medio':>11s}  regime")
    punti = regime.esplora(st)
    for p in punti:
        print(p.riga())
    scala = regime.scegli(punti)
    print(f"\nfattore scelto: {scala}. Sopra soglia ogni confronto fra insiemi "
          f"di seed misurerebbe rumore, non un effetto.")

    x = features.costruisci(liv, div)
    print()
    p_uniforme, d_uniforme = gnn.addestra(st, x, pesata=False, epoche=epoche)
    p_pesata, d_pesata = gnn.addestra(st, x, pesata=True, epoche=epoche)

    meglio_costante = 100 * (1 - d_uniforme["mse"] / d_uniforme["mse_costante"])
    delta_hub = 100 * (1 - d_pesata["mse_hub"] / d_uniforme["mse_hub"])
    print(f"\nLa GNN batte la costante del {meglio_costante:.0f}% di MSE: la "
          f"credit distribution si estende agli archi senza dati meglio di un "
          f"valore fisso.")
    if delta_hub > 1:
        print(f"La loss pesata migliora sugli archi verso gli hub del "
              f"{delta_hub:.1f}%: l'ipotesi sul meccanismo regge sulla "
              f"predizione, e resta da vedere se arriva ai seed.")
    else:
        print(f"La loss pesata NON migliora sugli hub ({delta_hub:+.1f}% "
              f"rispetto all'uniforme): l'ipotesi che l'MSE uniforme trascuri "
              f"gli hub e' respinta gia' sulla predizione, prima ancora "
              f"dell'ablazione sui seed. Si porta avanti la uniforme.")
    print("Quale delle due arrivi a cambiare i seed lo dice la Fase 3.")

    poveri = ~st.ricco
    np.savez_compressed(
        _percorso(sito),
        u=st.u, v=st.v, ricco=st.ricco, p_uv=st.p_uv, p_vu=st.p_vu,
        adozioni=st.adozioni, n=np.array([st.n]), scala=np.array([scala]),
        p_gnn_uniforme=p_uniforme, p_gnn_pesata=p_pesata,
        p_costante=np.full(2 * int(poveri.sum()),
                           float(np.median(st.p_uv[st.ricco]))),
    )
    print(f"\ncache scritta in {_percorso(sito).name}")
    return {"stima": st, "scala": scala, "attributi": x,
            "p_uniforme": p_uniforme, "p_pesata": p_pesata,
            "diagnostica": {"uniforme": d_uniforme, "pesata": d_pesata}}


def carica(sito: str):
    """(Stima, scala, dizionario delle p sui poveri) dalla cache."""
    d = np.load(_percorso(sito), allow_pickle=False)
    st = credit.Stima(d["u"], d["v"], d["ricco"], d["p_uv"], d["p_vu"],
                      d["adozioni"], int(d["n"][0]))
    varianti = {"gnn uniforme": d["p_gnn_uniforme"],
                "gnn pesata": d["p_gnn_pesata"],
                "costante": d["p_costante"]}
    return st, float(d["scala"][0]), varianti


if __name__ == "__main__":
    esegui(sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE)
