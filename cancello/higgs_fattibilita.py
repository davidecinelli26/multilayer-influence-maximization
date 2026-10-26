"""Fattibilita' delle tre fasi su Higgs: sovrapposizione fra gli strati di
interazione, adozioni rimaste dopo vari tagli temporali, e quota di eventi
con il bersaglio gia' adottante (materiale per la credit distribution).

    python -m cancello.higgs_fattibilita
"""

import gzip

import numpy as np

from asnm.config import GREZZI

B = GREZZI / "higgs"
P = lambda *a: print(*a, flush=True)

strati = {}
for nome in ("retweet", "reply", "mention"):
    E = set()
    with gzip.open(f"{B}/higgs-{nome}_network.edgelist.gz", "rt") as f:
        for r in f:
            a, b, _ = r.split()
            a, b = int(a), int(b)
            if a != b:
                E.add((a, b) if a < b else (b, a))
    strati[nome] = E
    P(
        f"  strato {nome:8s}: {len(E):>8,} archi non orientati, "
        f"{len({x for e in E for x in e}):>7,} nodi"
    )

P("\n  sovrapposizione fra strati (Jaccard):")
n = list(strati)
for i in range(3):
    for j in range(i + 1, 3):
        A, C = strati[n[i]], strati[n[j]]
        P(
            f"    {n[i]:8s} vs {n[j]:8s}  {len(A&C)/len(A|C):.4f}  "
            f"({len(A&C):,} in comune)"
        )

att = []
with gzip.open(f"{B}/higgs-activity_time.txt.gz", "rt") as f:
    for r in f:
        a, b, ts, az = r.split()
        att.append((int(a), int(b), int(ts), az))
ts = np.array([x[2] for x in att])
t0, t1 = ts.min(), ts.max()
P(f"\n  attivita': {len(att):,} eventi su {(t1-t0)/86400:.1f} giorni")

primo = {}
for a, b, t, _ in att:
    for u in (a, b):
        if u not in primo or t < primo[u]:
            primo[u] = t
pa = np.array(sorted(primo.values()))
P(f"  adottanti distinti: {len(pa):,}")
P("\n  taglio temporale: quante adozioni restano dopo?")
for ore in (24, 48, 72, 96, 120):
    taglio = t0 + ore * 3600
    prima, dopo = int((pa < taglio).sum()), int((pa >= taglio).sum())
    P(
        f"    dopo {ore:>3d}h  addestramento {prima:>7,}  verifica {dopo:>7,} "
        f"({100*dopo/len(pa):>4.1f}%)"
    )

# archi con evidenza di trasmissione: u agisce su v dopo che v ha adottato
ric = sum(1 for a, b, t, _ in att if primo.get(b, 1 << 62) < t)
P(
    f"\n  eventi con l'obiettivo gia' adottante prima: {ric:,} "
    f"({100*ric/len(att):.1f}%) -> materiale per la credit distribution"
)
