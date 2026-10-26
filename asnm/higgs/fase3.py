"""Fase 3 su Higgs: i seed e il costo della diversita'.

    python -m asnm.higgs.fase3 [--rr=120000] [--k=50] [--sim=1200]

1. controllo di stabilita' (stesso modello, altro seme) e ablazione della
   GNN contro una p costante, a regime pareggiato;
2. superficie k x alpha: comunita' toccate e diffusione attesa per ogni
   budget e ogni peso del vincolo di diversita'.
"""
import sys

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from asnm.config import CACHE
from asnm.evaluation import montecarlo
from asnm.higgs import comunita
from asnm.higgs.dati import costruisci
from asnm.higgs.fase2 import carica as carica_probabilita
from asnm.probabilities import regime
from asnm.seeds import greedy, rr

ALPHA = (1.00, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30)
BUDGET = (5, 10, 20, 50)
SEME_A, SEME_B = 1, 707
MINIMO = 200
"""Soglia di dimensione per il premio di diversita' (lo 0,04% della rete, la
stessa frazione dei 50 nodi su StackExchange)."""


def _insiemi(p, quanti, seme, etichetta, scala=None):
    """Insiemi RR dalla cache o generati; la scala di p e' parte del nome del
    file, cosi' un cambio di taratura non rilegge insiemi di un altro regime."""
    marca = etichetta if scala is None else f"{etichetta}_s{scala:g}"
    percorso = CACHE / f"higgs_rr_{marca}_{quanti}_{seme}.npz"
    if percorso.exists():
        d = np.load(percorso, allow_pickle=False)
        m = sp.csr_matrix((d["dati"], d["indici"], d["ptr"]),
                          shape=tuple(d["forma"])).tocsc()
        print(f"[rr] {etichetta} seme {seme}: dalla cache "
              f"({m.nnz/m.shape[0]:.2f} nodi per insieme)")
        return m
    m, _ = rr.genera(p, quanti, seme)
    c = m.tocsr()
    np.savez_compressed(percorso, dati=c.data, indici=c.indices, ptr=c.indptr,
                        forma=np.array(c.shape))
    return m


def esegui(quanti: int = 120_000, k: int = 50,
           nome_partizione: str = "fusione calibrata",
           simulazioni: int = 1200) -> dict:
    liv, fin = costruisci()
    st, scala, varianti = carica_probabilita()
    cid = np.unique(comunita.partizione(liv, nome_partizione, CACHE),
                    return_inverse=True)[1]
    dimensioni = np.bincount(cid)
    significative = dimensioni >= MINIMO
    print(f"[fase3] partizione '{nome_partizione}': {len(dimensioni):,} "
          f"comunita', {int(significative.sum())} con almeno {MINIMO} nodi, "
          f"che coprono il {100*dimensioni[significative].sum()/liv.n:.1f}% "
          f"della rete. p riscalata di {scala}.")

    # ogni variante di p viene portata alla stessa dimensione media degli
    # insiemi RR, cosi' l'ablazione confronta i modelli e non il regime
    bersaglio = rr.dimensione_media(
        st.matrice(p_poveri=varianti["gnn uniforme"], scala=scala), 400)
    scale = {"gnn uniforme": scala}
    matrici = {"gnn uniforme": st.matrice(p_poveri=varianti["gnn uniforme"],
                                          scala=scala)}
    for nome, v in varianti.items():
        if nome == "gnn uniforme":
            continue
        scale[nome] = regime.pareggia(st, v, bersaglio)
        matrici[nome] = st.matrice(p_poveri=v, scala=scale[nome])
    print("[fase3] dimensione media degli insiemi RR pareggiata a "
          f"{bersaglio:.1f} nodi; scale: "
          + ", ".join(f"{n} {s:.3f}" for n, s in scale.items()))

    # ---------------------------------------------- 1. stabilita' e ablazione
    print("\n1. CONTROLLO DI STABILITA' E ABLAZIONE DELLA GNN")
    insiemi, scelte, coperture = {}, {}, {}
    for nome in ("gnn uniforme", "costante"):
        for seme in (SEME_A, SEME_B):
            m = _insiemi(matrici[nome], quanti, seme, nome.replace(" ", "-"),
                         scale[nome])
            insiemi[(nome, seme)] = m
            scelte[(nome, seme)] = greedy.seleziona(m, k, alpha=1.0).semi
            coperture[(nome, seme)] = np.asarray(m.sum(0)).ravel()

    def confronto(x, y, etichetta, isola):
        a, b = scelte[x], scelte[y]
        m = (coperture[x] > 0) | (coperture[y] > 0)
        rho = spearmanr(coperture[x][m], coperture[y][m]).statistic
        sov = [f"{len(set(a[:q]) & set(b[:q]))}/{q}" for q in (10, 25, k)]
        print(f"  {etichetta:34s} {sov[0]:>7s} {sov[1]:>7s} {sov[2]:>7s} "
              f"{rho:>9.4f}   {isola}")
        return len(set(a[:k]) & set(b[:k]))

    print(f"  {'confronto':34s} {'k=10':>7s} {'k=25':>7s} {'k='+str(k):>7s} "
          f"{'Spearman':>9s}   che cosa isola")
    r1 = confronto(("gnn uniforme", SEME_A), ("gnn uniforme", SEME_B),
                   "GNN contro se stessa, altro seme", "rumore")
    r2 = confronto(("costante", SEME_A), ("costante", SEME_B),
                   "costante contro se stessa, altro seme", "rumore")
    effetto = confronto(("gnn uniforme", SEME_A), ("costante", SEME_A),
                        "GNN contro costante", "EFFETTO DELLA GNN")
    # due pavimenti di rumore: il permissivo e' il massimo fra le due
    # auto-consistenze, il severo il minimo; si giudica sul severo
    permissivo, severo = max(r1, r2), min(r1, r2)
    print(f"\n  Pavimento permissivo {permissivo}/{k}, severo {severo}/{k}, "
          f"effetto della GNN {effetto}/{k}.")
    if effetto < severo:
        print("  L'effetto supera anche il pavimento severo: i due modelli "
              "scelgono seed diversi\n  piu' di quanto lo stesso modello "
              "differisca da se' stesso.")
    else:
        print("  L'effetto sta dentro il pavimento severo: la sovrapposizione "
              "non risolve nulla,\n  e la domanda va girata alla diffusione "
              "attesa e alla validazione.")

    print(f"\n  {'insieme di seed':34s} {'sotto il modello GNN':>22s} "
          f"{'sotto la costante':>19s}")
    for etichetta, chiave in (("scelti con la GNN", ("gnn uniforme", SEME_A)),
                              ("scelti con la costante", ("costante", SEME_A))):
        a = montecarlo.diffusione(matrici["gnn uniforme"], scelte[chiave],
                                  simulazioni)
        b = montecarlo.diffusione(matrici["costante"], scelte[chiave], simulazioni)
        print(f"  {etichetta:34s} {a:>22.1f} {b:>19.1f}")
    grado = greedy.per_grado(matrici["gnn uniforme"], k)
    print(f"  {'i '+str(k)+' di grado massimo':34s} "
          f"{montecarlo.diffusione(matrici['gnn uniforme'], grado, simulazioni):>22.1f} "
          f"{montecarlo.diffusione(matrici['costante'], grado, simulazioni):>19.1f}")

    # ------------------------------------------------ 2. la superficie k x alpha
    print(f"\n2. SUPERFICIE k x ALPHA (partizione '{nome_partizione}')")
    print(f"  {'k':>4s} {'alpha':>6s} {'com. seed':>10s} {'diffusione':>11s} "
          f"{'costo':>8s}  lettura")
    p_lavoro = matrici["gnn uniforme"]
    m_lavoro = insiemi[("gnn uniforme", SEME_A)]
    superficie = {}
    for budget in BUDGET:
        base = None
        for a in ALPHA:
            s = greedy.seleziona(m_lavoro, budget, alpha=a, cid=cid,
                                 dimensioni=dimensioni, minimo=MINIMO)
            diff = montecarlo.diffusione(p_lavoro, s.semi, simulazioni)
            nota = ""
            if base is None:
                base = diff
                nota = f"TIM+ concentra in {s.comunita_distinte}"
            superficie[(budget, a)] = (s, diff, diff / base - 1)
            print(f"  {budget:>4d} {a:>6.2f} {s.comunita_distinte:>10} "
                  f"{diff:>11.1f} {100*(diff/base-1):>7.2f}%  {nota}")
        print()

    print("  Il tasso di scambio e' (1-alpha)/alpha: quanta copertura in piu' "
          "deve avere\n  un nodo ridondante per battere uno che apre una "
          "comunita' nuova.")
    return {"superficie": superficie, "scelte": scelte, "matrici": matrici,
            "cid": cid, "dimensioni": dimensioni, "insiemi": insiemi}


if __name__ == "__main__":
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(quanti=int(opzioni.get("rr", 120_000)), k=int(opzioni.get("k", 50)),
           nome_partizione=opzioni.get("partizione", "fusione calibrata"),
           simulazioni=int(opzioni.get("sim", 1200)))
