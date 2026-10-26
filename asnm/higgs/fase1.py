"""Fase 1 su Higgs: quale partizione predice dove passa la diffusione.

Criterio decisionale Q_tras: modularita' della partizione sulla rete di
trasmissione realizzata dopo il taglio, cioe' gli archi di follow u -> v
tali che v ha adottato dopo il taglio e u era gia' attivo. E' fuori campione
per costruzione. Criterio di supporto: AMI fra la partizione e l'ora di
adozione dei bersagli. Il bootstrap e' appaiato e a grappoli sull'ora.
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import normalized_mutual_info_score as nmi

from asnm.communities import bootstrap
from asnm.config import CACHE
from asnm.communities.evaluate import modularita, statistiche_intra
from asnm.higgs import comunita
from asnm.higgs.dati import TAGLIO_ORE, costruisci


def trasmissioni(liv, fin):
    """Gli archi di follow u -> v con v adottante dopo il taglio e u gia'
    attivo. Restituisce (u, v, ora dell'adozione di v)."""
    u, v, _ = liv.archi["F"]
    a = fin.adozione
    bersaglio = fin.bersagli
    m = bersaglio[v] & (a[u] < a[v])
    ora = ((a[v[m]] - fin.taglio) // 3600).astype(np.int32)
    return u[m], v[m], ora


def rete_trasmissione(u, v, n) -> sp.csr_matrix:
    a = sp.csr_matrix((np.ones(len(u), np.float64), (u, v)), shape=(n, n))
    return (a + a.T).tocsr()


def valuta(nome, cid, rete, ore_nodo, maschera):
    """Q_tras, AMI e NMI sull'ora di adozione, quota intra e rapporto col caso."""
    _, etichette = np.unique(cid, return_inverse=True)
    quota, caso = statistiche_intra(rete, etichette)
    return {
        "nome": nome,
        "comunita": int(etichette.max()) + 1,
        "q": modularita(rete, etichette),
        "ami": float(ami(ore_nodo[maschera], etichette[maschera])),
        "nmi": float(nmi(ore_nodo[maschera], etichette[maschera])),
        "quota": quota,
        "rapporto": quota / caso if caso else float("inf"),
    }


def esegui(repliche: int = 2000) -> dict:
    liv, fin = costruisci()

    # le partizioni per prime, una alla volta e su disco
    partizioni = comunita.candidate(liv, cache=CACHE)
    partizioni["(controllo) singoletti"] = np.arange(liv.n, dtype=np.int32)

    t0 = time.time()
    u, v, ora = trasmissioni(liv, fin)
    n_ore = int(ora.max()) + 1
    rete = rete_trasmissione(u, v, liv.n)
    print(f"[fase1] rete di trasmissione: {len(u):,} archi realizzati su "
          f"{n_ore} ore di verifica  [{time.time()-t0:.0f}s]")

    ore_nodo = np.full(liv.n, -1, np.int32)
    ore_nodo[v] = ora
    bersagli = fin.bersagli
    ore_nodo[bersagli & (ore_nodo < 0)] = n_ore     # adottanti senza esposizione
    print(f"[fase1] bersagli con almeno un'esposizione: "
          f"{len(np.unique(v)):,} su {int(bersagli.sum()):,}")

    righe = [valuta(n, c, rete, ore_nodo, bersagli)
             for n, c in partizioni.items()]

    print(f"\n{'partizione':22s} {'com.':>8s} {'Q_tras':>9s} {'AMI ora':>9s} "
          f"{'NMI ora':>9s} {'intra':>8s} {'rapporto':>9s}")
    print("-" * 82)
    for r in sorted(righe, key=lambda x: -x["q"]):
        print(f"{r['nome']:22s} {r['comunita']:>8,} {r['q']:>9.4f} "
              f"{r['ami']:>9.4f} {r['nmi']:>9.4f} {r['quota']:>8.4f} "
              f"{r['rapporto']:>8.2f}x")

    migliore = max(righe, key=lambda x: x["q"])["nome"]
    print(f"\nBOOTSTRAP APPAIATO A GRAPPOLI SULL'ORA ({repliche:,} repliche, "
          f"unita' = l'ora intera)")
    print(bootstrap.Esito.intestazione().replace("Q_esp", "Q_tras"))
    for e in bootstrap.confronta(partizioni, u, v, ora, n_ore, migliore,
                                 repliche=repliche):
        print(e.riga())

    print(f"\nMigliore per Q_tras: {migliore}.")
    return {"righe": righe, "migliore": migliore, "partizioni": partizioni}


if __name__ == "__main__":
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    print(f"[fase1] substrato higgs, taglio all'ora {TAGLIO_ORE}")
    esegui(repliche=int(opzioni.get("rep", 2000)))
