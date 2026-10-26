"""Le famiglie di attributi della scala a pioli.

Quattro famiglie, tutte calcolabili prima dell'ora 72: grado, esposizione,
comunita' (come numeri: dimensione, quota accesa, esposizione interna),
spettro. I gradini successivi (propagati, SIGN, R-SIGN) usano gli stessi
attributi propagati sul grafo. La verita' di riferimento si legge in un
posto solo, `etichette()`.
"""
import time
from dataclasses import dataclass

import numpy as np

from asnm.config import CACHE
from asnm.higgs import comunita
from asnm.higgs.dati import costruisci

PARTIZIONE = "fusione calibrata"
"""La partizione scelta dalla Fase 1."""

FAMIGLIE = ("grado", "esposizione", "comunita", "spettro")
"""Le famiglie tabellari in ordine di impilamento. Non c'e' una famiglia
«strati di interazione»: sulla popolazione (i non attivi al taglio) i gradi
su RT, MT e RE sono identicamente zero, perche' comparire in un'interazione
e' cio' che definisce l'essere attivi."""


@dataclass
class Scala:
    nomi: dict          # famiglia -> lista dei nomi delle colonne
    blocchi: dict       # famiglia -> matrice (n, d) float32
    popolazione: np.ndarray   # indici dei nodi non attivi al taglio
    y: np.ndarray             # etichetta per ogni nodo della popolazione
    comunita: np.ndarray      # id di comunita' per ogni nodo (tutti i nodi)

    def fino_a(self, famiglia: str) -> tuple[np.ndarray, list]:
        """La matrice degli attributi impilati fino alla famiglia data."""
        quali = FAMIGLIE[: FAMIGLIE.index(famiglia) + 1]
        x = np.hstack([self.blocchi[f] for f in quali])
        nomi = [n for f in quali for n in self.nomi[f]]
        return x, nomi


# ------------------------------------------------------------------ etichette
def etichette(fin):
    """La popolazione (non attivi al taglio) e l'etichetta (adotta dopo).
    L'unico posto che legge dopo il taglio."""
    popolazione = np.flatnonzero(~fin.gia_attivi)
    y = fin.bersagli[popolazione].astype(np.int8)
    return popolazione, y


# ------------------------------------------------------------------ famiglie
def _archi_follow(liv):
    """(influenzatore, influenzato): v segue u, quindi u -> v."""
    u, v, _ = liv.archi["F"]
    return u, v


def grado(liv, fin):
    u, v = _archi_follow(liv)
    seguaci = np.bincount(u, minlength=liv.n).astype(np.float32)   # out-degree
    seguiti = np.bincount(v, minlength=liv.n).astype(np.float32)   # in-degree
    x = np.stack([np.log1p(seguaci), np.log1p(seguiti)], 1)
    return x, ["log_seguaci", "log_seguiti"]


def esposizione(liv, fin):
    """Quanti fra i miei seguiti erano gia' attivi al taglio, e quando."""
    u, v = _archi_follow(liv)
    attivo = fin.gia_attivi
    ora_att = np.where(attivo, (fin.adozione - fin.t0) / 3600.0, np.nan)

    m = attivo[u]
    seguiti_attivi = np.bincount(v[m], minlength=liv.n).astype(np.float32)
    seguiti = np.bincount(v, minlength=liv.n).astype(np.float32)
    frazione = np.where(seguiti > 0, seguiti_attivi / np.maximum(seguiti, 1), 0)

    # anche il verso opposto: quanti dei miei seguaci sono gia' attivi
    # (omofilia osservabile, non canale)
    seguaci_attivi = np.bincount(u[attivo[v]], minlength=liv.n).astype(np.float32)

    # l'ora della prima esposizione: il minimo fra le ore di adozione dei
    # seguiti attivi. 72 per chi non e' esposto.
    prima = np.full(liv.n, 72.0, np.float32)
    np.minimum.at(prima, v[m], ora_att[u[m]].astype(np.float32))

    x = np.stack([np.log1p(seguiti_attivi), frazione,
                  np.log1p(seguaci_attivi), prima / 72.0], 1).astype(np.float32)
    return x, ["log_seguiti_attivi", "quota_seguiti_attivi",
               "log_seguaci_attivi", "ora_prima_esposizione"]


def comunita_(liv, fin, cid):
    """Quanto e' grande la mia comunita', quanto e' gia' accesa, e quanta
    della mia esposizione viene da dentro."""
    dim = np.bincount(cid).astype(np.float32)
    attivi_per_com = np.bincount(cid, weights=fin.gia_attivi).astype(np.float32)
    quota_accesa = attivi_per_com / np.maximum(dim, 1)

    u, v = _archi_follow(liv)
    m = fin.gia_attivi[u]
    dentro = m & (cid[u] == cid[v])
    esp_intra = np.bincount(v[dentro], minlength=liv.n).astype(np.float32)
    esp_tot = np.bincount(v[m], minlength=liv.n).astype(np.float32)
    quota_intra = np.where(esp_tot > 0, esp_intra / np.maximum(esp_tot, 1), 0)

    x = np.stack([np.log1p(dim[cid]), quota_accesa[cid],
                  np.log1p(esp_intra), quota_intra], 1).astype(np.float32)
    return x, ["log_dim_comunita", "quota_comunita_accesa",
               "log_esposizione_intra", "quota_esposizione_intra"]


def spettro(liv, fin):
    """Le 32 dimensioni spettrali del grafo follower, le stesse della Fase 2."""
    from asnm.higgs.probabilita import attributi as attributi_fase2

    percorso = CACHE / "higgs_spettro32.npy"
    if percorso.exists():
        x = np.load(percorso)
    else:
        # attributi() restituisce 10 strutturali + 32 spettrali
        x = attributi_fase2(liv, fin)[:, -32:].astype(np.float32)
        np.save(percorso, x)
    return x, [f"spettro_{i:02d}" for i in range(32)]


# ------------------------------------------------------------------ driver
def costruisci_scala() -> Scala:
    t0 = time.time()
    liv, fin = costruisci()
    cid = np.unique(comunita.partizione(liv, PARTIZIONE, CACHE),
                    return_inverse=True)[1]
    popolazione, y = etichette(fin)

    blocchi, nomi = {}, {}
    for nome, f in (("grado", grado), ("esposizione", esposizione),
                    ("comunita", lambda l, f_: comunita_(l, f_, cid)),
                    ("spettro", spettro)):
        x, n = f(liv, fin)
        blocchi[nome], nomi[nome] = x, n
        print(f"[caratteristiche] {nome:12s} {x.shape[1]:>3d} attributi  "
              f"[{time.time()-t0:.0f}s]")

    print(f"[caratteristiche] popolazione {len(popolazione):,} nodi non "
          f"attivi al taglio, di cui {int(y.sum()):,} adottano "
          f"({100*y.mean():.1f}%)")
    return Scala(nomi, blocchi, popolazione, y, cid)


if __name__ == "__main__":
    s = costruisci_scala()
    x, nomi = s.fino_a("spettro")
    print(f"\nmatrice completa: {x.shape}, {len(nomi)} nomi")
    sub = x[s.popolazione]
    print("\nmedia per attributo, adottanti contro non adottanti:")
    for i, n in enumerate(nomi[:14]):
        a, b = sub[s.y == 1, i].mean(), sub[s.y == 0, i].mean()
        print(f"  {n:26s} {a:>8.3f} {b:>8.3f}")
