"""Fase 3 su StackExchange: stabilita', ablazione della GNN, curva alpha.

    python -m asnm.seeds.fase3 [sito] [--rr=120000] [--k=50]
"""
import sys

import numpy as np
from scipy.stats import spearmanr

from asnm.communities import fase1
from asnm.config import CACHE, SITO_PRINCIPALE
from asnm.evaluation import montecarlo
from asnm.layers import costruisci as costruisci_livelli
from asnm.probabilities.fase2 import carica as carica_probabilita
from asnm.seeds import greedy, rr

ALPHA = (1.00, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30)
SEME_A, SEME_B = 1, 707


def _insiemi(p, quanti, seme, etichetta, sito):
    percorso = CACHE / f"rr_{sito}_{etichetta}_{quanti}_{seme}.npz"
    if percorso.exists():
        d = np.load(percorso, allow_pickle=False)
        import scipy.sparse as sp
        m = sp.csr_matrix((d["dati"], d["indici"], d["ptr"]),
                          shape=tuple(d["forma"])).tocsc()
        print(f"[rr] {etichetta} seme {seme}: letti dalla cache "
              f"({m.nnz/m.shape[0]:.2f} nodi per insieme)")
        return m
    m, _ = rr.genera(p, quanti, seme)
    c = m.tocsr()
    np.savez_compressed(percorso, dati=c.data, indici=c.indices, ptr=c.indptr,
                        forma=np.array(c.shape))
    return m


def _sovrapposizione(a: list, b: list, k: int) -> str:
    return f"{len(set(a[:k]) & set(b[:k]))}/{k}"


def esegui(sito: str, quanti: int = 120_000, k: int = 50,
           nome_partizione: str = "L1+L2 fusione") -> dict:
    liv = costruisci_livelli(sito)
    st, scala, varianti = carica_probabilita(sito)
    partizioni = fase1.candidate(liv)

    # la partizione di lavoro: la migliore per Q_esp secondo la Fase 1
    cid = np.unique(partizioni[nome_partizione], return_inverse=True)[1]
    print(f"[fase3] vincolo di diversita' sulla partizione '{nome_partizione}' "
          f"({cid.max()+1:,} comunita'), p riscalata di {scala}")

    dimensioni = np.bincount(cid)
    significative = int((dimensioni >= greedy.MINIMO).sum())
    print(f"[fase3] delle {len(dimensioni)} comunita', {significative} hanno "
          f"almeno {greedy.MINIMO} nodi e coprono il "
          f"{100*dimensioni[dimensioni >= greedy.MINIMO].sum()/dimensioni.sum():.1f}% "
          f"della rete: solo quelle danno il premio di diversita'.")

    matrici = {n: st.matrice(p_poveri=v, scala=scala) for n, v in varianti.items()}

    # ---------------------------------------------- 1. controllo di stabilita'
    print("\n1. CONTROLLO DI STABILITA' E ABLAZIONE DELLA GNN")
    insiemi = {}
    for nome in ("gnn uniforme", "costante"):
        for seme in (SEME_A, SEME_B):
            insiemi[(nome, seme)] = _insiemi(matrici[nome], quanti, seme,
                                             nome.replace(" ", "-"), sito)
    scelte, coperture = {}, {}
    for chiave, m in insiemi.items():
        scelte[chiave] = greedy.seleziona(m, k, alpha=1.0).semi
        coperture[chiave] = np.asarray(m.sum(0)).ravel()

    def confronto(x, y, etichetta, isola):
        a, b = scelte[x], scelte[y]
        m = (coperture[x] > 0) | (coperture[y] > 0)
        rho = spearmanr(coperture[x][m], coperture[y][m]).statistic
        print(f"  {etichetta:34s} {_sovrapposizione(a,b,10):>7s} "
              f"{_sovrapposizione(a,b,25):>7s} {_sovrapposizione(a,b,k):>7s} "
              f"{rho:>9.4f}   {isola}")
        return len(set(a[:k]) & set(b[:k]))

    print(f"  {'confronto':34s} {'k=10':>7s} {'k=25':>7s} {'k=' + str(k):>7s} "
          f"{'Spearman':>9s}   che cosa isola")
    r1 = confronto(("gnn uniforme", SEME_A), ("gnn uniforme", SEME_B),
                   "GNN contro se stessa, altro seme", "rumore")
    r2 = confronto(("costante", SEME_A), ("costante", SEME_B),
                   "costante contro se stessa, altro seme", "rumore")
    effetto = confronto(("gnn uniforme", SEME_A), ("costante", SEME_A),
                        "GNN contro costante", "EFFETTO DELLA GNN")
    pavimento = max(r1, r2)
    print(f"\n  Pavimento di rumore {pavimento}/{k}, effetto della GNN "
          f"{effetto}/{k}.")

    print(f"\n  {'insieme di seed':34s} {'sotto il modello GNN':>22s} "
          f"{'sotto la costante':>19s}")
    for etichetta, chiave in (("scelti con la GNN", ("gnn uniforme", SEME_A)),
                              ("scelti con la costante", ("costante", SEME_A))):
        a = montecarlo.diffusione(matrici["gnn uniforme"], scelte[chiave])
        b = montecarlo.diffusione(matrici["costante"], scelte[chiave])
        print(f"  {etichetta:34s} {a:>22.1f} {b:>19.1f}")
    grado = greedy.per_grado(matrici["gnn uniforme"], k)
    print(f"  {'i ' + str(k) + ' di grado massimo':34s} "
          f"{montecarlo.diffusione(matrici['gnn uniforme'], grado):>22.1f} "
          f"{montecarlo.diffusione(matrici['costante'], grado):>19.1f}")

    # ------------------------------------------------------- 2. la curva alpha
    print(f"\n2. CURVA ALPHA (partizione '{nome_partizione}', k={k})")
    print(f"  {'alpha':>6s} {'com. seed':>10s} {'di cui vere':>12s} "
          f"{'diffusione':>11s} {'costo':>8s} {'com. raggiunte':>15s} {'guadagno':>9s}")
    p_lavoro = matrici["gnn uniforme"]
    m_lavoro = insiemi[("gnn uniforme", SEME_A)]
    riferimento = None
    curva = {}
    for a in ALPHA:
        s = greedy.seleziona(m_lavoro, k, alpha=a, cid=cid,
                             dimensioni=dimensioni)
        diff = montecarlo.diffusione(p_lavoro, s.semi)
        raggiunte = montecarlo.comunita_raggiunte(p_lavoro, s.semi, cid)
        if riferimento is None:
            riferimento = (diff, raggiunte)
        curva[a] = (s, diff, raggiunte)
        vere = sum(1 for c in set(s.comunita) if dimensioni[c] >= greedy.MINIMO)
        print(f"  {a:>6.2f} {s.comunita_distinte:>10} {vere:>12} "
              f"{diff:>11.1f} {100*(diff/riferimento[0]-1):>7.1f}% "
              f"{raggiunte:>15.2f} {100*(raggiunte/riferimento[1]-1):>8.1f}%")

    print("\n  Il tasso di scambio e' (1-alpha)/alpha: quanta copertura in piu' "
          "deve avere\n  un nodo ridondante per battere uno che apre una comunita' "
          "nuova.")
    for a in (0.90, 0.80, 0.50):
        print(f"    alpha={a:.2f} -> {(1-a)/a:.3f}"
              + ("  (>= 1: irraggiungibile, la diversita' domina sempre)"
                 if (1 - a) / a >= 1 else ""))
    return {"curva": curva, "scelte": scelte, "matrici": matrici, "cid": cid}


if __name__ == "__main__":
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(argomenti[0] if argomenti else SITO_PRINCIPALE,
           quanti=int(opzioni.get("rr", 120_000)), k=int(opzioni.get("k", 50)),
           nome_partizione=opzioni.get("partizione", "L1+L2 fusione"))
