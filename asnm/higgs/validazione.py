"""Validazione sulle adozioni vere.

    python -m asnm.higgs.validazione [--k=50] [--sim=3000] [--reale=4.55]

I seed si scelgono con i soli dati precedenti all'ora 72; si stima con
Monte Carlo la probabilita' pi(v) che ogni nodo venga attivato; il punteggio
di una strategia e' la somma di pi(v) sui 269.081 utenti che hanno davvero
adottato dopo il taglio. Bootstrap appaiato a grappoli sull'ora. Il controllo
finale rivaluta gli stessi seed alla trasmissivita' misurata del periodo di
verifica (fattore 4,55, misurato in `stazionarieta.py`).
"""
import sys
import time

import numpy as np
import scipy.sparse as sp

from asnm.config import CACHE
from asnm.evaluation import bootstrap, montecarlo
from asnm.higgs import comunita, fase3
from asnm.higgs.dati import costruisci
from asnm.higgs.fase2 import carica as carica_probabilita
from asnm.probabilities import regime
from asnm.seeds import greedy, rr

SIMULAZIONI = 3000
ALPHA = (0.90, 0.80, 0.70, 0.50)
FATTORE_REALE = 4.55
"""Quanto la finestra di verifica e' piu' trasmissiva di quella di
addestramento (misurato in `stazionarieta.py`)."""


def punteggi_per_ora(pi: np.ndarray, ora_bersaglio: np.ndarray,
                     n_ore: int) -> np.ndarray:
    """Somma di pi(v) sui veri adottanti, separata per ora di adozione."""
    m = ora_bersaglio >= 0
    return np.bincount(ora_bersaglio[m], weights=pi[m], minlength=n_ore)


def esegui(k: int = 50, quanti: int = 120_000, simulazioni: int = SIMULAZIONI,
           nome_partizione: str = "fusione calibrata",
           fattore_reale: float = FATTORE_REALE,
           simulazioni_reale: int = 250) -> dict:
    liv, fin = costruisci()
    st, scala, varianti = carica_probabilita()
    cid = np.unique(comunita.partizione(liv, nome_partizione, CACHE),
                    return_inverse=True)[1]
    dimensioni = np.bincount(cid)

    p = st.matrice(p_poveri=varianti["gnn uniforme"], scala=scala)
    insiemi = fase3._insiemi(p, quanti, fase3.SEME_A, "gnn-uniforme", scala)

    bersagli = fin.bersagli
    ora_bersaglio = np.where(bersagli,
                             (fin.adozione - fin.taglio) // 3600, -1).astype(int)
    n_ore = int(ora_bersaglio.max()) + 1
    totale_veri = int(bersagli.sum())
    print(f"[validazione] {totale_veri:,} adottanti veri dopo il taglio, "
          f"distribuiti su {n_ore} ore. Budget k={k}.")

    strategie = {"TIM+ (alpha=1)": greedy.seleziona(insiemi, k, alpha=1.0).semi}
    for a in ALPHA:
        strategie[f"diversita' alpha={a:.2f}"] = greedy.seleziona(
            insiemi, k, alpha=a, cid=cid, dimensioni=dimensioni,
            minimo=fase3.MINIMO).semi
    # variante con il divieto di scegliere chi ha gia' adottato: si azzerano
    # le sue colonne, cosi' esce dai candidati ma i suoi insiemi RR restano
    # copribili dagli altri
    maschera = sp.diags((~fin.gia_attivi).astype(np.int8))
    fra_non_attivi = (insiemi @ maschera).tocsc()
    fra_non_attivi.eliminate_zeros()
    strategie["TIM+ fra i non attivi"] = greedy.seleziona(
        fra_non_attivi, k, alpha=1.0).semi
    strategie["diversita' fra i non attivi"] = greedy.seleziona(
        fra_non_attivi, k, alpha=0.80, cid=cid, dimensioni=dimensioni,
        minimo=fase3.MINIMO).semi

    # ablazione della GNN sul criterio esterno: seed scelti con una p
    # costante, portata alla stessa dimensione media degli insiemi RR
    scala_cost = regime.pareggia(
        st, varianti["costante"],
        rr.dimensione_media(p, 400))
    p_cost = st.matrice(p_poveri=varianti["costante"], scala=scala_cost)
    insiemi_cost = fase3._insiemi(p_cost, quanti, fase3.SEME_A, "costante",
                                  scala_cost)
    strategie["TIM+ con p costante"] = greedy.seleziona(
        insiemi_cost, k, alpha=1.0).semi

    strategie["grado uscente"] = greedy.per_grado(p, k)
    strategie["a caso"] = greedy.a_caso(liv.n, k)
    strategie["a caso fra gli attivi"] = [
        int(i) for i in np.random.default_rng(4).choice(
            np.flatnonzero(fin.gia_attivi), k, replace=False)]

    punteggi, totali, comunita_semi = {}, {}, {}
    for nome, semi in strategie.items():
        t0 = time.time()
        pi = montecarlo.attivazione(p, semi, simulazioni)
        punteggi[nome] = punteggi_per_ora(pi, ora_bersaglio, n_ore)
        totali[nome] = float(punteggi[nome].sum())
        comunita_semi[nome] = len({int(cid[s]) for s in semi})
        print(f"  {nome:26s} punteggio {totali[nome]:>8.1f}  "
              f"comunita' dei seed {comunita_semi[nome]:>3}  "
              f"[{time.time()-t0:.0f}s]")

    print(f"\nRIFERIMENTO: partenza da tutti i {int(fin.gia_attivi.sum()):,} "
          "utenti gia' attivi al taglio.")
    pi_attivi = montecarlo.attivazione(
        p, np.flatnonzero(fin.gia_attivi), max(simulazioni // 10, 200))
    spiegati = float(punteggi_per_ora(pi_attivi, ora_bersaglio, n_ore).sum())
    print(f"  il modello spiega {spiegati:,.0f} adozioni vere su "
          f"{totale_veri:,} ({100*spiegati/totale_veri:.1f}%).")

    riferimento = "TIM+ (alpha=1)"
    print("\nBOOTSTRAP APPAIATO A GRAPPOLI SULL'ORA (unita' = l'ora intera)")
    print(bootstrap.Confronto.intestazione())
    esiti = bootstrap.confronta(punteggi, riferimento)
    for e in esiti:
        print(e.riga())

    # ------------------------------------------------------- robustezza
    # gli stessi seed, valutati alla trasmissivita' misurata del periodo di
    # verifica (sopra soglia: si valuta, non si sceglie)
    esiti_reale = None
    if fattore_reale:
        print("\nCONTROLLO: gli stessi seed, valutati alla trasmissivita' "
              f"misurata del periodo\ndi verifica (x{fattore_reale:.2f}, "
              "sopra soglia di percolazione)")
        p_reale = st.matrice(p_poveri=varianti["gnn uniforme"],
                             scala=scala * fattore_reale)
        punteggi_reale = {}
        for nome, semi in strategie.items():
            pi = montecarlo.attivazione(p_reale, semi, simulazioni_reale)
            punteggi_reale[nome] = punteggi_per_ora(pi, ora_bersaglio, n_ore)
        print(bootstrap.Confronto.intestazione())
        esiti_reale = bootstrap.confronta(punteggi_reale, riferimento)
        for e in esiti_reale:
            print(e.riga())

    return {"punteggi": punteggi, "totali": totali, "esiti": esiti,
            "esiti_reale": esiti_reale, "strategie": strategie,
            "tetto": spiegati, "veri": totale_veri}


if __name__ == "__main__":
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(k=int(opzioni.get("k", 50)), quanti=int(opzioni.get("rr", 120_000)),
           simulazioni=int(opzioni.get("sim", SIMULAZIONI)),
           fattore_reale=float(opzioni.get("reale", FATTORE_REALE)),
           simulazioni_reale=int(opzioni.get("simreale", 150)),
           nome_partizione=opzioni.get("partizione", "fusione calibrata"))
