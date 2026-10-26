"""La rete multistrato: la classe `Livelli` e la sua costruzione da StackExchange.

Un'istanza di `Livelli` tiene i nodi e, per ogni livello, gli archi orientati
con i pesi; espone le viste che gli algoritmi usano (matrice orientata,
simmetrizzata, grafo networkx, fusione normalizzata). I nomi dei livelli sono
un attributo dell'istanza, cosi' la stessa classe regge i due livelli di
StackExchange (L1 risposta, L2 commento) e i quattro di Higgs.

Su StackExchange l'arco u -> v di L1 significa «u ha risposto a una domanda
di v», quello di L2 «u ha commentato un post di v»; entrano solo gli eventi
precedenti al confine temporale.
"""
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from asnm.adoption import leggi_post, mese
from asnm.config import CACHE, percorso_xml
from asnm.splits import CUTOFF, Divisione

NOMI = ("L1", "L2")
ETICHETTE = {"L1": "risposta", "L2": "commento"}
"""I livelli di StackExchange. Gli algoritmi leggono `liv.nomi` e
`liv.etichette`, non queste costanti."""


@dataclass
class Livelli:
    sito: str
    nodi: np.ndarray              # id utente, ordinati; la posizione e' l'indice
    archi: dict                   # nome -> (u, v, peso), array paralleli
    etichette: dict = field(default_factory=lambda: dict(ETICHETTE))

    def __post_init__(self):
        self.indice = {u: i for i, u in enumerate(self.nodi)}

    @property
    def n(self) -> int:
        return len(self.nodi)

    @property
    def nomi(self) -> tuple:
        """I livelli disponibili, nell'ordine di inserimento."""
        return tuple(self.archi)

    # ------------------------------------------------------------ viste
    def csr(self, nome: str) -> sp.csr_matrix:
        """Matrice orientata dei pesi, n x n."""
        u, v, w = self.archi[nome]
        return sp.csr_matrix((w, (u, v)), shape=(self.n, self.n))

    def csr_simmetrica(self, nome: str) -> sp.csr_matrix:
        """Vista non orientata: i pesi delle due direzioni si sommano."""
        a = self.csr(nome)
        return (a + a.T).tocsr()

    def grafo(self, nome: str, pesato: bool = True):
        """networkx.Graph non orientato, con tutti i nodi anche isolati (cosi'
        l'insieme dei nodi e' lo stesso su ogni livello)."""
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        a = sp.triu(self.csr_simmetrica(nome), k=1).tocoo()
        pesi = a.data if pesato else np.ones_like(a.data)
        g.add_weighted_edges_from(zip(a.row.tolist(), a.col.tolist(), pesi.tolist()))
        return g

    def fusione(self, nomi=None):
        """Fusione pesata, ogni livello normalizzato per il proprio peso totale."""
        import networkx as nx

        nomi = self.nomi if nomi is None else nomi
        tot = sp.csr_matrix((self.n, self.n))
        for nome in nomi:
            a = self.csr_simmetrica(nome).astype(float)
            somma = a.sum()
            if somma:
                tot = tot + a / somma
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        a = sp.triu(tot, k=1).tocoo()
        g.add_weighted_edges_from(zip(a.row.tolist(), a.col.tolist(), a.data.tolist()))
        return g

    def vicinato(self) -> list:
        """Vicini di ciascun nodo nell'unione non orientata dei livelli."""
        unione = sum((self.csr_simmetrica(n) for n in self.nomi),
                     sp.csr_matrix((self.n, self.n))).tocsr()
        return [unione.indices[unione.indptr[i]:unione.indptr[i + 1]]
                for i in range(self.n)]

    def riassunto(self) -> str:
        righe = [f"sito {self.sito}: {self.n:,} nodi (unione dei livelli)"]
        for nome in self.nomi:
            u, _, w = self.archi[nome]
            righe.append(f"  {nome} {self.etichette.get(nome, ''):9s} {len(u):>8,} archi orientati, "
                         f"peso totale {int(w.sum()):>9,}")
        return "\n".join(righe)


# ------------------------------------------------------------- costruzione
def _conta_archi(sito: str) -> tuple[Counter, Counter]:
    """Conteggi orientati dei due livelli, solo eventi pre-cutoff."""
    post = leggi_post(sito)
    t0 = time.time()

    L1 = Counter()
    for id_post, tipo in post.tipo.items():
        if tipo != "2" or post.quando[id_post] >= CUTOFF:
            continue
        autore = post.proprietario.get(id_post)
        domanda = post.genitore.get(id_post)
        if not autore or domanda is None:
            continue
        destinatario = post.proprietario.get(domanda)
        if destinatario and destinatario != autore:
            L1[(autore, destinatario)] += 1

    L2 = Counter()
    for a in _righe_commenti(sito):
        if "UserId" not in a or mese(a["CreationDate"]) >= CUTOFF:
            continue
        destinatario = post.proprietario.get(a["PostId"])
        if destinatario and destinatario != a["UserId"]:
            L2[(a["UserId"], destinatario)] += 1

    print(f"[layers] {sito}: L1 {len(L1):,} archi, L2 {len(L2):,} archi "
          f"in {time.time()-t0:.0f}s")
    return L1, L2


def _righe_commenti(sito: str):
    for _, el in ET.iterparse(percorso_xml(sito, "Comments.xml"), events=("end",)):
        if el.tag == "row":
            yield el.attrib
            el.clear()


def _percorso_cache(sito: str):
    return CACHE / f"livelli_{sito}.npz"


def costruisci(sito: str, usa_cache: bool = True) -> Livelli:
    percorso = _percorso_cache(sito)
    if usa_cache and percorso.exists():
        d = np.load(percorso, allow_pickle=False)
        archi = {n: (d[f"{n}_u"], d[f"{n}_v"], d[f"{n}_w"]) for n in NOMI}
        print(f"[layers] {sito}: livelli letti dalla cache")
        return Livelli(sito, d["nodi"], archi)

    conteggi = dict(zip(NOMI, _conta_archi(sito)))
    nodi = np.array(sorted({x for c in conteggi.values() for e in c for x in e}))
    indice = {u: i for i, u in enumerate(nodi)}

    archi, da_salvare = {}, {"nodi": nodi}
    for nome, c in conteggi.items():
        u = np.fromiter((indice[a] for a, _ in c), np.int32, len(c))
        v = np.fromiter((indice[b] for _, b in c), np.int32, len(c))
        w = np.fromiter(c.values(), np.float32, len(c))
        archi[nome] = (u, v, w)
        da_salvare |= {f"{nome}_u": u, f"{nome}_v": v, f"{nome}_w": w}

    np.savez_compressed(percorso, **da_salvare)
    print(f"[layers] {sito}: cache scritta in {percorso.name} "
          f"({percorso.stat().st_size/1e6:.1f} MB)")
    return Livelli(sito, nodi, archi)


if __name__ == "__main__":
    import sys

    from asnm.config import SITO_PRINCIPALE

    sito = sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE
    liv = costruisci(sito)
    print()
    print(liv.riassunto())

    # coerenza con il confine temporale
    d = Divisione.e_train(CUTOFF - 1), Divisione.e_train(CUTOFF)
    assert d == (True, False), "il filtro pre-cutoff non e' allineato a splits"

    vic = liv.vicinato()
    gradi = np.fromiter((len(v) for v in vic), int, liv.n)
    print(f"\nunione non orientata: grado mediano {np.median(gradi):.0f}, "
          f"massimo {gradi.max():,}, isolati {(gradi == 0).sum():,}")
    for nome in liv.nomi:
        g = liv.grafo(nome)
        print(f"  grafo {nome}: {g.number_of_nodes():,} nodi, "
              f"{g.number_of_edges():,} archi non orientati")
    f = liv.fusione()
    print(f"  fusione : {f.number_of_nodes():,} nodi, {f.number_of_edges():,} archi")
