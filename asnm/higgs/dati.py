"""Il substrato Higgs Twitter: quattro livelli e la finestra temporale.

Livelli (archi memorizzati nel verso in cui scorre l'influenza):
  F   follower  se A segue B l'arco e' B -> A. E' il canale, l'unico livello
                che esiste prima dell'evento.
  RT  retweet   se a ritwitta b l'arco e' b -> a
  MT  mention   a menziona b
  RE  reply     a risponde a b (contenuto in MT)
I tre livelli di interazione usano solo gli eventi precedenti al taglio.

L'adozione di un utente e' il primo istante in cui compare nel log delle
attivita'. Il taglio e' all'ora 72 (mezzanotte fra il 3 e il 4 luglio 2012,
poche ore prima dell'annuncio del CERN): tutto cio' che i modelli vedono sta
prima, la verifica sta dopo.
"""
import datetime as dt
import gzip
import time
from dataclasses import dataclass

import numpy as np

from asnm.config import CACHE, GREZZI
from asnm.layers import Livelli

CARTELLA = GREZZI / "higgs"
SITO = "higgs"

FILE = {
    "attivita": "higgs-activity_time.txt.gz",
    "follower": "higgs-social_network.edgelist.gz",
}
URL = "https://snap.stanford.edu/data/"

TAGLIO_ORE = 72
"""Il confine. Tutto cio' che il modello puo' vedere sta prima."""

FINE_ORE = 168
"""Sette giorni pieni: il dump si ferma li'."""

ETICHETTE = {"F": "follower", "RT": "retweet", "MT": "mention", "RE": "reply"}


# ------------------------------------------------------------------ lettura
def _controlla_file():
    mancanti = [f for f in FILE.values() if not (CARTELLA / f).exists()]
    if mancanti:
        raise FileNotFoundError(
            f"mancano {mancanti} in {CARTELLA}. Si scaricano da {URL}"
        )


def attivita():
    """(a, b, timestamp, azione) come array paralleli. a agisce su b."""
    _controlla_file()
    codice = {"RT": 0, "MT": 1, "RE": 2}
    a, b, t, z = [], [], [], []
    with gzip.open(CARTELLA / FILE["attivita"], "rt") as f:
        for riga in f:
            x, y, ts, az = riga.split()
            a.append(int(x))
            b.append(int(y))
            t.append(int(ts))
            z.append(codice[az])
    return (np.array(a, np.int32), np.array(b, np.int32),
            np.array(t, np.int64), np.array(z, np.int8))


def follower():
    """(segue, seguito). A segue B: i tweet di B raggiungono A."""
    _controlla_file()
    d = np.loadtxt(CARTELLA / FILE["follower"], dtype=np.int32)
    return d[:, 0], d[:, 1]


# ------------------------------------------------------------- divisione
@dataclass
class Finestra:
    """Il confine temporale e chi c'e' da prevedere (l'analogo di
    `asnm.splits.Divisione`: qui il confine e' solo temporale)."""

    t0: int                      # primo istante osservato
    adozione: np.ndarray         # per indice di nodo, il primo istante (o MAX)
    n: int

    MAX = np.iinfo(np.int64).max

    @property
    def taglio(self) -> int:
        return self.t0 + TAGLIO_ORE * 3600

    @property
    def fine(self) -> int:
        return self.t0 + FINE_ORE * 3600

    def e_train(self, ts) -> bool:
        return ts < self.taglio

    def ora(self, ts):
        """Ore dall'inizio dell'osservazione: l'unita' di stratificazione."""
        return (np.asarray(ts) - self.t0) // 3600

    # ------------------------------------------------------- insiemi di nodi
    @property
    def gia_attivi(self) -> np.ndarray:
        """Chi ha gia' adottato prima del taglio: non e' piu' da conquistare."""
        return self.adozione < self.taglio

    @property
    def bersagli(self) -> np.ndarray:
        """La verita' di riferimento: chi adotta dopo il taglio."""
        return (self.adozione >= self.taglio) & (self.adozione < self.MAX)

    def riassunto(self) -> str:
        q = dt.datetime.utcfromtimestamp(self.taglio)
        i = dt.datetime.utcfromtimestamp(self.t0)
        return (
            "substrato higgs\n"
            f"  osservazione         : {FINE_ORE} ore dal {i:%d %b %Y %H:%M} UTC\n"
            f"  taglio               : ora {TAGLIO_ORE} ({q:%d %b %H:%M} UTC), "
            "poche ore prima dell'annuncio\n"
            f"  nodi                 : {self.n:,}\n"
            f"  gia' attivi al taglio: {int(self.gia_attivi.sum()):,}\n"
            f"  da prevedere         : {int(self.bersagli.sum()):,} "
            f"({100*self.bersagli.mean():.1f}% della rete)\n"
            f"  mai attivi           : {int((self.adozione == self.MAX).sum()):,}"
        )


# ------------------------------------------------------------ costruzione
def _percorso_cache():
    return CACHE / f"higgs_livelli_{TAGLIO_ORE}h.npz"


def costruisci(usa_cache: bool = True):
    """I quattro livelli piu' la finestra temporale. I nodi sono tutti quelli
    del grafo dei follower, compresi quelli che non compaiono mai nel log."""
    percorso = _percorso_cache()
    if usa_cache and percorso.exists():
        d = np.load(percorso, allow_pickle=False)
        nomi = [str(x) for x in d["nomi"]]
        archi = {n: (d[f"{n}_u"], d[f"{n}_v"], d[f"{n}_w"]) for n in nomi}
        liv = Livelli(SITO, d["nodi"], archi, dict(ETICHETTE))
        fin = Finestra(int(d["t0"]), d["adozione"], len(d["nodi"]))
        print(f"[higgs] livelli letti dalla cache ({percorso.name})")
        return liv, fin

    avvio = time.time()
    seg, sgt = follower()
    a, b, ts, az = attivita()
    n = int(max(seg.max(), sgt.max(), a.max(), b.max())) + 1
    nodi = np.arange(n, dtype=np.int32)

    t0 = int(ts.min())
    adozione = np.full(n, Finestra.MAX, np.int64)
    np.minimum.at(adozione, a, ts)
    np.minimum.at(adozione, b, ts)
    fin = Finestra(t0, adozione, n)

    archi = {}
    # F: A segue B  ->  l'influenza va da B ad A
    archi["F"] = (sgt.astype(np.int32), seg.astype(np.int32),
                  np.ones(len(seg), np.float32))
    # gli altri: a agisce su b  ->  l'influenza va da b ad a
    prima = ts < fin.taglio
    for nome, codice in (("RT", 0), ("MT", 1), ("RE", 2)):
        m = prima & (az == codice) & (a != b)
        chiavi = b[m].astype(np.int64) * n + a[m]
        unici, conteggi = np.unique(chiavi, return_counts=True)
        archi[nome] = ((unici // n).astype(np.int32),
                       (unici % n).astype(np.int32),
                       conteggi.astype(np.float32))

    liv = Livelli(SITO, nodi, archi, dict(ETICHETTE))
    da_salvare = {"nodi": nodi, "t0": t0, "adozione": adozione,
                  "nomi": np.array(list(archi))}
    for nome, (u, v, w) in archi.items():
        da_salvare |= {f"{nome}_u": u, f"{nome}_v": v, f"{nome}_w": w}
    np.savez(percorso, **da_salvare)
    print(f"[higgs] costruito in {time.time()-avvio:.0f}s, cache "
          f"{percorso.stat().st_size/1e6:.0f} MB")
    return liv, fin


if __name__ == "__main__":
    liv, fin = costruisci()
    print()
    print(fin.riassunto())
    print()
    print(liv.riassunto())

    # RE e' contenuto in MT?
    def insieme(nome):
        u, v, _ = liv.archi[nome]
        return set(zip(u.tolist(), v.tolist()))

    re_, mt = insieme("RE"), insieme("MT")
    print(f"\n  RE dentro MT: {len(re_ & mt):,} su {len(re_):,} "
          f"({100*len(re_ & mt)/max(len(re_), 1):.0f}%)")
