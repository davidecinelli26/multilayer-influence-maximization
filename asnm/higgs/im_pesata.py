"""Influence maximization pesata dalla previsione.

    python -m asnm.higgs.im_pesata [--k=20]

Le radici degli insiemi RR si estraggono in proporzione a p_hat(v), la
probabilita' prevista che v adotti (zero per chi e' gia' attivo): il greedy
massimizza allora gli adottanti probabili invece del numero di nodi.
Le p_hat sono prodotte out-of-fold su cinque fold, cosi' nessun nodo e'
pesato da un modello che ha visto la sua etichetta. Il confronto che conta
e' quello diretto col grado uscente sul criterio della validazione.
"""
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from asnm.config import CACHE
from asnm.evaluation import bootstrap, montecarlo
from asnm.higgs import fase3
from asnm.higgs.adozione import caratteristiche, modelli
from asnm.higgs.dati import costruisci
from asnm.higgs.fase2 import carica as carica_probabilita
from asnm.higgs.validazione import FATTORE_REALE, punteggi_per_ora
from asnm.seeds import greedy, rr

FOLD = 5


def probabilita_out_of_fold(liv, scala, seme: int = 0) -> np.ndarray:
    """p_hat per ogni nodo: zero per i gia' attivi, out-of-fold per gli altri."""
    percorso = CACHE / "higgs_adozione_phat_oof.npy"
    if percorso.exists():
        return np.load(percorso)
    t0 = time.time()
    x4, _ = scala.fino_a(caratteristiche.FAMIGLIE[-1])
    xprop = modelli.propaga(liv, x4)[scala.popolazione]
    y = scala.y
    phat_pop = np.zeros(len(y), np.float64)
    for k, (tr, te) in enumerate(StratifiedKFold(
            FOLD, shuffle=True, random_state=seme).split(xprop, y)):
        phat_pop[te] = modelli.gradient_boosting(xprop[tr], y[tr], xprop[te],
                                                 seme=k)
    from sklearn.metrics import roc_auc_score
    print(f"[im-pesata] p_hat out-of-fold: AUC {roc_auc_score(y, phat_pop):.4f} "
          f"su {len(y):,} nodi  [{time.time()-t0:.0f}s]")
    phat = np.zeros(liv.n, np.float64)
    phat[scala.popolazione] = phat_pop
    np.save(percorso, phat)
    return phat


def esegui(k: int = 20, quanti: int = 120_000, simulazioni: int = 2000,
           fattore_reale: float = FATTORE_REALE, simulazioni_reale: int = 150):
    liv, fin = costruisci()
    st, scala_p, varianti = carica_probabilita()
    sc = caratteristiche.costruisci_scala()
    phat = probabilita_out_of_fold(liv, sc)
    cid = sc.comunita
    dimensioni = np.bincount(cid)

    p = st.matrice(p_poveri=varianti["gnn uniforme"], scala=scala_p)
    insiemi = fase3._insiemi(p, quanti, fase3.SEME_A, "gnn-uniforme", scala_p)

    percorso = CACHE / f"higgs_rr_pesati_{quanti}_{fase3.SEME_A}.npz"
    if percorso.exists():
        import scipy.sparse as sp
        d = np.load(percorso)
        pesati = sp.csr_matrix((d["dati"], d["indici"], d["ptr"]),
                               shape=tuple(d["forma"])).tocsc()
    else:
        pesati, _ = rr.genera(p, quanti, fase3.SEME_A, pesi_radici=phat)
        c = pesati.tocsr()
        np.savez_compressed(percorso, dati=c.data, indici=c.indices,
                            ptr=c.indptr, forma=np.array(c.shape))

    bersagli = fin.bersagli
    ora_b = np.where(bersagli, (fin.adozione - fin.taglio) // 3600, -1).astype(int)
    n_ore = int(ora_b.max()) + 1

    strategie = {
        "TIM+ (alpha=1)": greedy.seleziona(insiemi, k, alpha=1.0).semi,
        "TIM pesato da p_hat": greedy.seleziona(pesati, k, alpha=1.0).semi,
        "TIM pesato + diversita' 0.80": greedy.seleziona(
            pesati, k, alpha=0.80, cid=cid, dimensioni=dimensioni,
            minimo=fase3.MINIMO).semi,
        "grado uscente": greedy.per_grado(p, k),
    }
    print(f"\n[im-pesata] k={k}: {simulazioni} simulazioni per strategia")
    punteggi = {}
    for nome, semi in strategie.items():
        pi = montecarlo.attivazione(p, semi, simulazioni)
        punteggi[nome] = punteggi_per_ora(pi, ora_b, n_ore)
        com = len({int(cid[s]) for s in semi})
        print(f"  {nome:30s} punteggio {punteggi[nome].sum():>8.1f}  "
              f"comunita' {com:>3}  gia' attivi fra i seed "
              f"{int(fin.gia_attivi[semi].sum()):>2}/{k}")

    rif = "TIM+ (alpha=1)"
    print("\nBOOTSTRAP APPAIATO A GRAPPOLI SULL'ORA")
    print(bootstrap.Confronto.intestazione())
    esiti = bootstrap.confronta(punteggi, rif)
    for e in esiti:
        print(e.riga())

    # il confronto diretto col grado
    print("\nCONTRO IL GRADO, direttamente")
    print(bootstrap.Confronto.intestazione())
    for e in bootstrap.confronta(punteggi, "grado uscente"):
        print(e.riga())
    np.savez(CACHE / f"higgs_im_pesata_k{k}.npz",
             **{n.replace(" ", "_").replace("'", ""): v
                for n, v in punteggi.items()})

    if fattore_reale:
        print(f"\nCONTROLLO alla trasmissivita' misurata (x{fattore_reale:.2f})")
        p_reale = st.matrice(p_poveri=varianti["gnn uniforme"],
                             scala=scala_p * fattore_reale)
        pr = {n: punteggi_per_ora(montecarlo.attivazione(p_reale, s, simulazioni_reale),
                                  ora_b, n_ore) for n, s in strategie.items()}
        print(bootstrap.Confronto.intestazione())
        for e in bootstrap.confronta(pr, rif):
            print(e.riga())
    return {"strategie": strategie, "punteggi": punteggi, "esiti": esiti}


if __name__ == "__main__":
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(k=int(opzioni.get("k", 20)),
           simulazioni=int(opzioni.get("sim", 2000)),
           fattore_reale=float(opzioni.get("reale", FATTORE_REALE)))
