"""La scala a pioli, dall'inizio alla fine.

    python -m asnm.higgs.adozione.esegui [--rep=2000]

Gradini: grado, esposizione, comunita', spettro (stessi alberi), propagati
(stessi alberi su [x, A x, A^2 x]), SIGN (MLP sugli stessi attributi
propagati). Ogni salto e' confrontato col precedente nel bootstrap; l'ultimo
anche col rumore fra due semi della stessa rete.
"""
import sys
import time

import numpy as np

from asnm.config import CACHE
from asnm.higgs.adozione import caratteristiche, modelli, valuta
from asnm.higgs.dati import costruisci

SEME_DIVISIONE = 2024


def divisione(n_pop: int, y, seme: int = SEME_DIVISIONE):
    """70 / 15 / 15 sui nodi della popolazione, stratificata sull'etichetta."""
    rng = np.random.default_rng(seme)
    tr, va, te = [], [], []
    for classe in (0, 1):
        idx = np.flatnonzero(y == classe)
        rng.shuffle(idx)
        a, b = int(0.70 * len(idx)), int(0.85 * len(idx))
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    return (np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def esegui(repliche: int = 2000) -> dict:
    t0 = time.time()
    scala = caratteristiche.costruisci_scala()
    liv, _ = costruisci()
    pop, y = scala.popolazione, scala.y
    tr, va, te = divisione(len(pop), y)
    print(f"[adozione] divisione: {len(tr):,} addestramento, {len(va):,} "
          f"validazione, {len(te):,} test; base rate {y.mean():.3f}")

    previsioni = {}
    print("\nLA SCALA A PIOLI")
    for famiglia in caratteristiche.FAMIGLIE:
        x, nomi = scala.fino_a(famiglia)
        xp = x[pop]
        print(f"  gradino '{famiglia}': {len(nomi)} attributi")
        previsioni[famiglia] = modelli.gradient_boosting(
            xp[tr], y[tr], xp[te])

    # «propagati»: gli stessi alberi sugli attributi propagati (isola
    # l'aggregazione, a parita' di modello); «sign»: l'MLP sugli stessi
    x4, _ = scala.fino_a(caratteristiche.FAMIGLIE[-1])
    xprop = modelli.propaga(liv, x4)[pop]
    print(f"  gradino 'propagati': {xprop.shape[1]} attributi "
          "(x, A x, A^2 x), stessi alberi")
    previsioni["propagati"] = modelli.gradient_boosting(
        xprop[tr], y[tr], xprop[te])
    print(f"  gradino 'sign': stessi {xprop.shape[1]} attributi, MLP")
    p_gnn, _ = modelli.sign(xprop, tr, y[tr], va, y[va], te, seme=11)
    previsioni["sign"] = p_gnn
    p_gnn_b, _ = modelli.sign(xprop, tr, y[tr], va, y[va], te, seme=707)
    previsioni["sign (seme b)"] = p_gnn_b

    print(f"\n{'gradino':22s} {'AUC':>8s} {'AP':>8s}")
    for nome, p in previsioni.items():
        auc, ap = valuta.metriche(y[te], p)
        print(f"{nome:22s} {auc:>8.4f} {ap:>8.4f}")

    nomi = list(caratteristiche.FAMIGLIE) + ["propagati", "sign"]
    coppie = list(zip(nomi[:-1], nomi[1:]))
    print(f"\nBOOTSTRAP APPAIATO A GRAPPOLI SULLE COMUNITA' ({repliche:,} "
          f"repliche, Bonferroni su {len(coppie)} confronti)")
    print(valuta.Confronto.intestazione())
    esiti = valuta.bootstrap(y[te], previsioni, scala.comunita[pop[te]],
                             coppie, repliche=repliche)
    for e in esiti:
        print(e.riga())

    rumore = valuta.bootstrap(
        y[te], {"a": p_gnn, "b": p_gnn_b}, scala.comunita[pop[te]],
        [("a", "b")], repliche=repliche)[0]
    print("\nRUMORE DI RIFERIMENTO: SIGN contro se' stessa con un altro seme")
    print(f"  differenza {rumore.differenza:+.4f}  "
          f"IC [{rumore.basso:+.4f}, {rumore.alto:+.4f}]  p={rumore.p:.3f}")
    ultimo = esiti[-1]
    if abs(ultimo.differenza) > max(abs(rumore.basso), abs(rumore.alto)):
        print(f"  L'effetto propagati -> sign ({ultimo.differenza:+.4f}) supera "
              "il rumore fra semi.")
    else:
        print(f"  L'effetto propagati -> sign ({ultimo.differenza:+.4f}) sta "
              "dentro il rumore fra semi: il MLP non aggiunge agli alberi.")

    np.savez(CACHE / "higgs_adozione_previsioni.npz",
             test=pop[te], y=y[te], **{k.replace(" ", "_"): v
                                       for k, v in previsioni.items()})
    print(f"\n[adozione] fatto in {(time.time()-t0)/60:.0f} min")
    return {"previsioni": previsioni, "esiti": esiti, "rumore": rumore,
            "test": pop[te], "y": y[te]}


if __name__ == "__main__":
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(repliche=int(opzioni.get("rep", 2000)))
