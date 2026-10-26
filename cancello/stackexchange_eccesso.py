"""Il cancello su StackExchange: eccesso strutturale della rete di risposte
e commenti sulle adozioni dei tag.

    python -m cancello.stackexchange_eccesso

Stessa misura di `higgs_eccesso.py`, mese per mese e su 70 tag estratti a
caso fra quelli con 40-1500 adottanti. Il modello nullo e' il doppio scambio
di archi a gradi preservati, che conserva anche il mese di ogni arco. Si
riporta il lift aggregato e quello stratificato per (mese, tag) e per mese.
Gli XML del sito devono essere gia' estratti in data/xml/security.
"""

import collections
import random
import time
import xml.etree.ElementTree as ET

import numpy as np

from asnm.config import XML as CARTELLA_XML


def rows(f):
    for _, el in ET.iterparse(f, events=("end",)):
        if el.tag == "row":
            yield el.attrib
            el.clear()


def mese(d):
    return int(d[:4]) * 12 + int(d[5:7])


XML = str(CARTELLA_XML / "security")
t0 = time.time()
owner, ptype, parent, qtags, pdate = {}, {}, {}, {}, {}
for a in rows(XML + "/Posts.xml"):
    i = a["Id"]
    ptype[i] = a["PostTypeId"]
    pdate[i] = a["CreationDate"]
    if "OwnerUserId" in a:
        owner[i] = a["OwnerUserId"]
    if "ParentId" in a:
        parent[i] = a["ParentId"]
    if "Tags" in a:
        qtags[i] = [
            t for t in a["Tags"].replace("|", "<").replace(">", "<").split("<") if t
        ]

primo = {}
attivo = collections.defaultdict(set)
for pid, t in ptype.items():
    u = owner.get(pid)
    if not u:
        continue
    m = mese(pdate[pid])
    attivo[m].add(u)
    tags = qtags.get(pid) if t == "1" else qtags.get(parent.get(pid, ""), None)
    if not tags:
        continue
    for tg in tags:
        k = (u, tg)
        if k not in primo or m < primo[k]:
            primo[k] = m

first_edge = {}


def add(u, v, m):
    if u == v:
        return
    k = (u, v) if u < v else (v, u)
    if k not in first_edge or m < first_edge[k]:
        first_edge[k] = m


for pid, t in ptype.items():
    if t == "2" and pid in owner and pid in parent:
        q = parent[pid]
        if q in owner:
            add(owner[pid], owner[q], mese(pdate[pid]))
for a in rows(XML + "/Comments.xml"):
    if "UserId" in a and a["PostId"] in owner:
        add(a["UserId"], owner[a["PostId"]], mese(a["CreationDate"]))

adopters = collections.Counter(tg for (_, tg) in primo)
tags_ok = [tg for tg, n in adopters.items() if 40 <= n <= 1500]
random.seed(0)
random.shuffle(tags_ok)
tags_ok = tags_ok[:70]
tagset = set(tags_ok)
mesi = [
    m
    for m in sorted(set(first_edge.values()) | set(primo.values()))
    if 2011 * 12 <= m <= 2023 * 12
]
ado_by_m = collections.defaultdict(list)
for (u, tg), m in primo.items():
    if tg in tagset:
        ado_by_m[m].append((u, tg))
print(
    f"[{time.time()-t0:.0f}s] dati pronti: {len(first_edge):,} archi, {len(tags_ok)} tag"
)


def dose_curve(edge_list):
    """edge_list: lista di (u,v,mese). Adozioni sempre quelle reali."""
    ebm = collections.defaultdict(list)
    for u, v, m in edge_list:
        ebm[m].append((u, v))
    vicini = collections.defaultdict(set)
    adottanti = collections.defaultdict(set)
    dose = collections.defaultdict(lambda: [0, 0])
    strati = collections.defaultdict(lambda: [0, 0, 0, 0])  # (mese,tag) -> n0,a0,n1,a1
    strati_m = collections.defaultdict(lambda: [0, 0, 0, 0])  # mese -> idem
    for m in mesi:
        att = attivo.get(m, set())
        ado_ora = collections.defaultdict(set)
        for u, tg in ado_by_m.get(m, []):
            ado_ora[tg].add(u)
        for tg in tags_ok:
            A = adottanti[tg]
            if not A:
                continue
            adesso = ado_ora.get(tg, set())
            for u in att:
                if u in A:
                    continue
                nb = vicini.get(u)
                if not nb:
                    continue
                k = len(nb & A)
                b = dose[min(k, 6)]
                b[0] += 1
                a = 1 if u in adesso else 0
                b[1] += a
                off = 0 if k == 0 else 2
                st = strati[(m, tg)]
                st[off] += 1
                st[off + 1] += a
                sm = strati_m[m]
                sm[off] += 1
                sm[off + 1] += a
        for u, v in ebm.get(m, []):
            vicini[u].add(v)
            vicini[v].add(u)
        for u, tg in ado_by_m.get(m, []):
            adottanti[tg].add(u)
    return dose, strati, strati_m


def mh(strati):
    """Mantel-Haenszel: rapporto entro strato, poi media pesata.

    Lo strato e' (mese, tag): sia il momento sia la popolarita' del tag
    spostano il rischio di base, e senza fissarli il rapporto aggregato
    scambia per effetto di rete cio' che e' composizione del campione.
    """
    num = den = 0.0
    for n0, a0, n1, a1 in strati.values():
        N = n0 + n1
        if N == 0:
            continue
        num += a1 * n0 / N
        den += a0 * n1 / N
    return num / den if den else float("nan")


def lift(d):
    n0, a0 = d[0]
    n1 = sum(d[k][0] for k in range(1, 7))
    a1 = sum(d[k][1] for k in range(1, 7))
    return a0 / n0, a1 / n1, (a1 / n1) / (a0 / n0)


reale = [(u, v, m) for (u, v), m in first_edge.items()]
print("--- rete reale ---")
d_real, st_real, stm_real = dose_curve(reale)
p0, p1, L = lift(d_real)
MH = mh(st_real)
MHm = mh(stm_real)
print(f"  P(k=0)={p0:.5f}  P(k>=1)={p1:.5f}  lift={L:.2f}x   [{time.time()-t0:.0f}s]")
print(f"  stratificato (mese,tag): {MH:.2f}x    stratificato (mese): {MHm:.2f}x")


def rewire(edges, rnd, passes=8):
    """double-edge swap: preserva la sequenza dei gradi e i mesi (legati allo slot)."""
    E = [[u, v, m] for u, v, m in edges]
    n = len(E)
    ex = {(u, v) if u < v else (v, u) for u, v, _ in E}
    ok = 0
    target = passes * n
    for _ in range(target):
        i, j = rnd.randrange(n), rnd.randrange(n)
        if i == j:
            continue
        u1, v1, m1 = E[i]
        u2, v2, m2 = E[j]
        if len({u1, v1, u2, v2}) < 4:
            continue
        a = (u1, v2) if u1 < v2 else (v2, u1)
        b = (u2, v1) if u2 < v1 else (v1, u2)
        if a in ex or b in ex:
            continue
        ex.discard((u1, v1) if u1 < v1 else (v1, u1))
        ex.discard((u2, v2) if u2 < v2 else (v2, u2))
        ex.add(a)
        ex.add(b)
        E[i] = [u1, v2, m1]
        E[j] = [u2, v1, m2]
        ok += 1
    return [(u, v, m) for u, v, m in E], ok


print("--- reti ricablate (3 repliche, grado preservato) ---")
Ls = []
MHs = []
MHms = []
for s in (1, 2, 3):
    rnd = random.Random(s)
    rw, ok = rewire(reale, rnd)
    d, stq, stmq = dose_curve(rw)
    q0, q1, Lq = lift(d)
    Ls.append(Lq)
    MHs.append(mh(stq))
    MHms.append(mh(stmq))
    print(
        f"  replica {s}: swap riusciti {ok:,}  P(k=0)={q0:.5f}  P(k>=1)={q1:.5f}  lift={Lq:.2f}x   [{time.time()-t0:.0f}s]"
    )

mn, sd = float(np.mean(Ls)), float(np.std(Ls))
print("\n" + "=" * 70)
print(f"lift rete REALE      : {L:.2f}x")
print(f"lift reti RICABLATE  : {mn:.2f}x   (dev.st {sd:.3f})")
print(f"ECCESSO STRUTTURALE  : {L/mn:.2f}x")
if sd > 0:
    print(f"z rispetto al null   : {(L-mn)/sd:.1f}")
m1 = float(np.mean(MHs))
m2 = float(np.mean(MHms))
print(
    f"STRATIFICATO (mese,tag): reale {MH:.2f}x  nullo {m1:.2f}x  ->  {100*(MH/m1-1):+.0f}%"
)
print(
    f"STRATIFICATO (mese)    : reale {MHm:.2f}x  nullo {m2:.2f}x  ->  {100*(MHm/m2-1):+.0f}%"
)
print("=" * 70)
print("\nCurva dose-risposta sulla rete reale:")
for k in range(7):
    n, a = d_real[k]
    if n:
        print(
            f"  k={'6+' if k==6 else k}  a rischio {n:>9,}  P={a/n:.5f}  lift={(a/n)/p0:.2f}x"
        )
