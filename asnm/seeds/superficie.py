"""La superficie k x alpha su StackExchange: comunita' toccate e costo della
diversita' per ogni budget e ogni peso del vincolo.

    python -m asnm.seeds.superficie [sito] [--rr=120000]
"""
import sys

import numpy as np

from asnm.communities import fase1
from asnm.config import SITO_PRINCIPALE
from asnm.evaluation import montecarlo
from asnm.layers import costruisci as costruisci_livelli
from asnm.probabilities.fase2 import carica as carica_probabilita
from asnm.seeds import fase3, greedy

BUDGET = (5, 10, 20, 50)
ALPHA = (1.00, 0.90, 0.80, 0.70, 0.50)
SIMULAZIONI = 2500


def esegui(sito: str, quanti: int = 120_000,
           nome_partizione: str = "L1+L2 fusione") -> dict:
    liv = costruisci_livelli(sito)
    st, scala, varianti = carica_probabilita(sito)
    cid = np.unique(fase1.candidate(liv)[nome_partizione], return_inverse=True)[1]
    dimensioni = np.bincount(cid)
    significative = int((dimensioni >= greedy.MINIMO).sum())
    p = st.matrice(p_poveri=varianti["gnn uniforme"], scala=scala)
    insiemi = fase3._insiemi(p, quanti, fase3.SEME_A, "gnn-uniforme", sito)

    print(f"[superficie] {sito}: {significative} comunita' significative su "
          f"{len(dimensioni)}; il vincolo ha spazio solo finche' k le e' "
          f"confrontabile")
    print(f"\n{'k':>4s} {'alpha':>6s} {'com. seed':>10s} {'diffusione':>11s} "
          f"{'costo':>8s}  lettura")

    superficie = {}
    for k in BUDGET:
        base = None
        for a in ALPHA:
            s = greedy.seleziona(insiemi, k, alpha=a, cid=cid,
                                 dimensioni=dimensioni)
            diff = montecarlo.diffusione(p, s.semi, SIMULAZIONI)
            if base is None:
                base = diff
                nota = (f"TIM+ concentra in {s.comunita_distinte} "
                        f"comunit{'a' if s.comunita_distinte == 1 else 'a'}")
            else:
                nota = ""
            superficie[(k, a)] = (s.comunita_distinte, diff, diff / base - 1)
            print(f"{k:>4d} {a:>6.2f} {s.comunita_distinte:>10} {diff:>11.1f} "
                  f"{100*(diff/base-1):>7.2f}%  {nota}")
        print()

    print("Il costo della diversita' cala al crescere del budget: con pochi "
          "seed\nspargerli toglie copertura, con molti il greedy li avrebbe "
          "sparsi comunque.")
    return superficie


if __name__ == "__main__":
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(argomenti[0] if argomenti else SITO_PRINCIPALE,
           quanti=int(opzioni.get("rr", 120_000)))
