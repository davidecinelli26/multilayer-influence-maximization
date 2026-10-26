"""Tutto cio' che la schermata deve mostrare, precalcolato una volta.

Il vincolo architetturale del progetto sta qui: **gli insiemi RR si generano
una volta sola e restano in cache**. Muovere lo slider deve rieseguire
soltanto il greedy, che costa un paio di secondi; se ogni movimento
rigenerasse gli insiemi RR l'interfaccia sarebbe inutilizzabile.

Anche le simulazioni Monte Carlo sono precalcolate sulla griglia di alpha,
perche' costano decine di secondi ciascuna. I numeri mostrati sono quindi
quelli veri, non stime al volo: lo slider scorre su una griglia discreta.
"""
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import networkx as nx

from asnm.communities import fase1
from asnm.config import CACHE
from asnm.evaluation import montecarlo
from asnm.evaluation.temporale import adottanti_per_tag
from asnm.layers import costruisci as costruisci_livelli
from asnm.probabilities.fase2 import carica as carica_probabilita
from asnm.seeds import fase3, greedy
from asnm.splits import dividi

GRIGLIA = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)
SIMULAZIONI = 2000


@dataclass
class Punto:
    alpha: float
    semi: list
    comunita_semi: int
    diffusione: float
    comunita_raggiunte: float
    adottanti: float
    motivi: list = field(default_factory=list)   # (nodo, guadagno, comunita, apre)


@dataclass
class Dossier:
    sito: str
    k: int
    n_nodi: int
    n_comunita: int
    punti: dict                 # alpha -> Punto
    baseline: dict              # nome -> Punto
    posizioni: np.ndarray       # coordinate delle comunita' (NaN se non disegnata)
    visibili: np.ndarray        # indici delle comunita' significative
    archi: np.ndarray           # archi fra comunita' (m x 2), i piu' pesanti
    dimensioni: np.ndarray      # nodi per comunita'
    cid: np.ndarray
    adottanti_per_comunita: np.ndarray   # quanti adottanti reali in ciascuna
    adottanti_noti: int
    tetto: float
    tempi: dict


def _motivi(sel: greedy.Selezione) -> list:
    viste, fuori = set(), []
    for nodo, guadagno, com in zip(sel.semi, sel.guadagni, sel.comunita):
        apre = com not in viste
        viste.add(com)
        fuori.append((int(nodo), int(guadagno), int(com), bool(apre)))
    return fuori


def _layout_comunita(liv, cid, n_com, dimensioni) -> tuple:
    """Grafo delle comunita' contratto, e sue coordinate.

    Disegnare 46.000 nodi produce un gomitolo illeggibile. Il grafo
    contratto per comunita' mostra invece esattamente la grandezza che lo
    slider controlla: quante e quali comunita' i seed toccano.

    Si disegnano solo le comunita' significative. Le altre -- su `security`
    sono 158 gruppetti da due a quattro persone -- formerebbero un anello di
    polvere intorno al grafico senza aggiungere niente.
    """
    from asnm.seeds.greedy import MINIMO
    visibili = np.flatnonzero(dimensioni >= MINIMO)
    ammesse = set(visibili.tolist())
    fusione = liv.fusione()
    contratto = nx.Graph()
    contratto.add_nodes_from(visibili.tolist())
    for a, b, d in fusione.edges(data=True):
        ca, cb = int(cid[a]), int(cid[b])
        if ca != cb and ca in ammesse and cb in ammesse:
            peso = contratto.get_edge_data(ca, cb, {"weight": 0})["weight"]
            contratto.add_edge(ca, cb, weight=peso + d.get("weight", 1.0))
    disposizione = nx.spring_layout(contratto, seed=1, iterations=200,
                                    weight="weight")
    posizioni = np.full((n_com, 2), np.nan)
    for i, xy in disposizione.items():
        posizioni[i] = xy
    archi = np.array([[a, b] for a, b in contratto.edges()], dtype=int)
    return posizioni, archi, visibili


def _a_dizionario(d: Dossier) -> dict:
    """Si serializzano dizionari, non classi.

    Costruendo il dossier con `python -m app.dossier` la classe verrebbe
    registrata nel pickle come `__main__.Dossier`, e riletta da un altro
    modulo -- per esempio dall'app -- fallirebbe. Salvare strutture semplici
    rende il file indipendente da come e' stato prodotto.
    """
    fuori = dict(d.__dict__)
    fuori["punti"] = {a: dict(p.__dict__) for a, p in d.punti.items()}
    fuori["baseline"] = {n: dict(p.__dict__) for n, p in d.baseline.items()}
    return fuori


def _da_dizionario(grezzo: dict) -> Dossier:
    dati = dict(grezzo)
    dati["punti"] = {a: Punto(**v) for a, v in grezzo["punti"].items()}
    dati["baseline"] = {n: Punto(**v) for n, v in grezzo["baseline"].items()}
    return Dossier(**dati)


def _percorso(sito, k):
    return CACHE / f"dossier_{sito}_{k}.pkl"


def costruisci(sito: str, k: int = 50, quanti: int = 120_000,
               usa_cache: bool = True) -> Dossier:
    percorso = _percorso(sito, k)
    if usa_cache and percorso.exists():
        with open(percorso, "rb") as f:
            grezzo = pickle.load(f)
        return _da_dizionario(grezzo)

    tempi = {}
    t = time.time()
    liv = costruisci_livelli(sito)
    div = dividi(sito)
    st, scala, varianti = carica_probabilita(sito)
    partizioni = fase1.candidate(liv)
    cid = np.unique(partizioni["L1+L2 fusione"], return_inverse=True)[1]
    n_com = int(cid.max()) + 1
    p = st.matrice(p_poveri=varianti["gnn uniforme"], scala=scala)
    tempi["caricamento"] = time.time() - t

    t = time.time()
    insiemi = fase3._insiemi(p, quanti, fase3.SEME_A, "gnn-uniforme", sito)
    tempi["insiemi RR"] = time.time() - t

    per_tag, eventi, noti = adottanti_per_tag(liv, div)
    adottanti = np.unique(np.concatenate(list(per_tag.values())))

    def valuta(semi, sel=None):
        diff, com_ragg, pi = montecarlo.analizza(p, semi, cid, SIMULAZIONI)
        return Punto(0.0, list(semi), len({int(cid[s]) for s in semi}),
                     diff, com_ragg, float(pi[adottanti].sum()),
                     _motivi(sel) if sel else [])

    punti, t_greedy, t_mc = {}, 0.0, 0.0
    for a in GRIGLIA:
        t = time.time()
        sel = greedy.seleziona(insiemi, k, alpha=a, cid=cid,
                               dimensioni=np.bincount(cid, minlength=n_com))
        t_greedy += time.time() - t
        t = time.time()
        punto = valuta(sel.semi, sel)
        t_mc += time.time() - t
        punto.alpha = a
        punti[a] = punto
        print(f"[dossier] alpha={a:.2f}: {punto.comunita_semi} comunita', "
              f"diffusione {punto.diffusione:.1f}, "
              f"adottanti {punto.adottanti:.1f}")
    tempi["selezione greedy"] = t_greedy / len(GRIGLIA)
    tempi["Monte Carlo"] = t_mc / len(GRIGLIA)

    baseline = {"grado massimo": valuta(greedy.per_grado(p, k)),
                "casuale": valuta(greedy.a_caso(liv.n, k, seme=0))}

    t = time.time()
    dimensioni_com = np.bincount(cid, minlength=n_com)
    posizioni, archi_com, visibili = _layout_comunita(
        liv, cid, n_com, dimensioni_com)
    tempi["disegno"] = time.time() - t

    dossier = Dossier(sito, k, liv.n, n_com, punti, baseline, posizioni,
                      visibili, archi_com, dimensioni_com, cid,
                      np.bincount(cid[adottanti], minlength=n_com),
                      len(adottanti), noti / eventi, tempi)
    with open(percorso, "wb") as f:
        pickle.dump(_a_dizionario(dossier), f)
    print(f"[dossier] scritto in {percorso.name}")
    return dossier


if __name__ == "__main__":
    import sys

    from asnm.config import SITO_PRINCIPALE

    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    d = costruisci(argomenti[0] if argomenti else SITO_PRINCIPALE,
                   k=int(opzioni.get("k", 50)),
                   quanti=int(opzioni.get("rr", 120_000)),
                   usa_cache="--rifai" not in sys.argv)
    print(f"\n{d.sito}: {d.n_nodi:,} nodi, {d.n_comunita} comunita', "
          f"k={d.k}, {len(d.punti)} valori di alpha")
    for nome, secondi in d.tempi.items():
        print(f"  {nome:18s} {secondi:6.2f}s")
