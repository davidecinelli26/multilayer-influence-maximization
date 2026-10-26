"""Il settimo gradino, dall'inizio alla fine.

    python -m asnm.higgs.adozione.esegui_relazionale [--rep=2000]

Rilegge le previsioni dei gradini precedenti dalla cache (stessa divisione,
stesso test) e aggiunge gli alberi sugli attributi relazionali propagati e
R-SIGN, con i confronti appaiati e il rumore fra semi.
"""
import sys
import time

import numpy as np

from asnm.config import CACHE
from asnm.higgs.adozione import caratteristiche, modelli, relazionale, valuta
from asnm.higgs.adozione.esegui import divisione
from asnm.higgs.dati import costruisci


def esegui(repliche: int = 2000) -> dict:
    t0 = time.time()
    scala = caratteristiche.costruisci_scala()
    liv, _ = costruisci()
    pop, y = scala.popolazione, scala.y
    tr, va, te = divisione(len(pop), y)

    prima = np.load(CACHE / "higgs_adozione_previsioni.npz")
    assert np.array_equal(prima["test"], pop[te]), \
        "la divisione non coincide con quella della prima esecuzione"
    previsioni = {"propagati": prima["propagati"], "sign": prima["sign"]}

    x4, _ = scala.fino_a(caratteristiche.FAMIGLIE[-1])
    xrel, blocchi = relazionale.propaga(liv, x4)
    xrel = xrel[pop]
    print(f"[relazionale] blocchi: {', '.join(blocchi)}")

    print(f"\n  gradino 'propagati relazionali': {xrel.shape[1]} attributi, "
          "stessi alberi")
    previsioni["propagati relazionali"] = modelli.gradient_boosting(
        xrel[tr], y[tr], xrel[te])
    print(f"  gradino 'r-sign': stessi {xrel.shape[1]} attributi, "
          "un blocco di pesi per relazione")
    p_a, _, rete = relazionale.r_sign(xrel, len(blocchi), tr, y[tr], va, y[va],
                                      te, seme=11)
    previsioni["r-sign"] = p_a
    p_b, _, _ = relazionale.r_sign(xrel, len(blocchi), tr, y[tr], va, y[va],
                                   te, seme=707)
    previsioni["r-sign (seme b)"] = p_b

    print(f"\n{'gradino':26s} {'AUC':>8s} {'AP':>8s}")
    for nome, p in previsioni.items():
        auc, ap = valuta.metriche(y[te], p)
        print(f"{nome:26s} {auc:>8.4f} {ap:>8.4f}")

    coppie = [("propagati", "propagati relazionali"), ("sign", "r-sign")]
    print(f"\nBOOTSTRAP APPAIATO A GRAPPOLI SULLE COMUNITA' ({repliche:,} "
          f"repliche, Bonferroni su {len(coppie)} confronti)")
    print(valuta.Confronto.intestazione())
    esiti = valuta.bootstrap(y[te], previsioni, scala.comunita[pop[te]],
                             coppie, repliche=repliche)
    for e in esiti:
        print(e.riga())

    rumore = valuta.bootstrap(
        y[te], {"a": p_a, "b": p_b}, scala.comunita[pop[te]], [("a", "b")],
        repliche=repliche)[0]
    print("\nRUMORE DI RIFERIMENTO: R-SIGN contro se' stessa con un altro seme")
    print(f"  differenza {rumore.differenza:+.4f}  "
          f"IC [{rumore.basso:+.4f}, {rumore.alto:+.4f}]  p={rumore.p:.3f}")
    ultimo = esiti[-1]
    soglia = max(abs(rumore.basso), abs(rumore.alto))
    if abs(ultimo.differenza) > soglia:
        print(f"  L'effetto sign -> r-sign ({ultimo.differenza:+.4f}) supera il "
              "rumore fra semi.")
    else:
        print(f"  L'effetto sign -> r-sign ({ultimo.differenza:+.4f}) sta dentro "
              "il rumore fra semi: la struttura relazionale non aggiunge.")

    # norma dei pesi di ciascun blocco d'ingresso: un indizio su quale
    # relazione pesa, non una prova
    print("\nPESO DEI BLOCCHI IN R-SIGN (norma di Frobenius di W_r)")
    for nome, proiezione in zip(blocchi, rete.proiezioni):
        norma = float(proiezione.weight.detach().norm())
        print(f"  {nome:8s} {norma:>7.3f}  " + "#" * int(norma * 4))

    np.savez(CACHE / "higgs_adozione_previsioni_rel.npz", test=pop[te],
             y=y[te], **{k.replace(" ", "_").replace("-", "_"): v
                         for k, v in previsioni.items()})
    print(f"\n[relazionale] fatto in {(time.time()-t0)/60:.0f} min")
    return {"previsioni": previsioni, "esiti": esiti, "rumore": rumore}


if __name__ == "__main__":
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(repliche=int(opzioni.get("rep", 2000)))
