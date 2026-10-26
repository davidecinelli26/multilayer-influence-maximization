"""Il cancello su Higgs Twitter: eccesso strutturale del grafo dei follower.

    python -m cancello.higgs_eccesso [retweet | tutte]

Per ogni ora e per ogni nodo non ancora adottante si conta k, il numero di
seguiti gia' adottanti, e si registra se il nodo adotta in quell'ora. Il
lift e' P(adotta | k >= 1) / P(adotta | k = 0); l'eccesso strutturale e' il
rapporto fra il lift della rete reale e quello di reti ricablate a gradi
preservati (permutazione della colonna dei seguiti, che conserva grado
entrante e uscente di ogni nodo), meno uno. Si riporta sia aggregato sia
stratificato per ora (Mantel-Haenszel).

Il substrato e' il grafo dei follower, che esiste prima dell'evento; la
rete di retweet non va bene perche' li' l'arco e' l'evento di diffusione
stesso e nessun nodo puo' essere esposto prima di adottare.
"""

import gzip
import sys
import time

import numpy as np
import scipy.sparse as sp

from asnm.config import GREZZI

BASE = GREZZI / "higgs"
ORA = 3600
SOGLIA = 6
P = lambda *a: print(*a, flush=True)
t0 = time.time()


def carica_follower():
    """A B  =  A segue B, quindi i tweet di B raggiungono A."""
    d = np.loadtxt(f"{BASE}/higgs-social_network.edgelist.gz", dtype=np.int32)
    return d[:, 0], d[:, 1]


def carica_adozioni(n, azioni=None):
    """Primo istante in cui ciascun utente compare nel log."""
    t = np.full(n, np.iinfo(np.int64).max, np.int64)
    with gzip.open(f"{BASE}/higgs-activity_time.txt.gz", "rt") as f:
        for riga in f:
            a, b, ts, az = riga.split()
            if azioni and az not in azioni:
                continue
            ts = int(ts)
            for u in (int(a), int(b)):
                if ts < t[u]:
                    t[u] = ts
    return t


def curva(seg, sgt, n, bin_ado, nbin):
    """dose -> [a rischio, adottanti]. Il substrato e' statico: una CSR sola."""
    ok = seg != sgt
    A = sp.csr_matrix(
        (np.ones(int(ok.sum()), np.int8), (seg[ok], sgt[ok])), shape=(n, n)
    )
    # csr_matrix somma i duplicati: senza binarizzare, un multi-arco creato
    # dalla permutazione varrebbe due esposizioni invece di una e gonfierebbe
    # il nullo proprio dove si concentra il segnale.
    A.data[:] = 1
    dose = np.zeros((SOGLIA + 1, 2), np.int64)
    strati = []  # (n0, a0, n1, a1) ora per ora
    adottato = np.zeros(n, np.int8)
    for b in range(nbin):
        if adottato.any():
            k = np.minimum(A.dot(adottato), SOGLIA)
            rischio = adottato == 0
            nuovi = rischio & (bin_ado == b)
            for s in range(SOGLIA + 1):
                m = rischio & (k == s)
                dose[s, 0] += int(m.sum())
                dose[s, 1] += int((m & nuovi).sum())
            z = rischio & (k == 0)
            e = rischio & (k > 0)
            strati.append(
                (
                    int(z.sum()),
                    int((z & nuovi).sum()),
                    int(e.sum()),
                    int((e & nuovi).sum()),
                )
            )
        adottato |= (bin_ado == b).astype(np.int8)
    return dose, np.array(strati, np.float64)


def lift_stratificato(strati):
    """Mantel-Haenszel: il rapporto entro ogni ora, poi la media pesata.

    Senza questo il confronto e' esposto a un'obiezione seria: nelle ore di
    picco quasi tutti sono esposti E quasi tutti adottano, e il rapporto
    aggregato prenderebbe per effetto di rete cio' che e' solo il momento.
    Stratificando per ora quel canale e' chiuso per costruzione.
    """
    n0, a0, n1, a1 = strati.T
    N = n0 + n1
    v = N > 0
    num = (a1[v] * n0[v] / N[v]).sum()
    den = (a0[v] * n1[v] / N[v]).sum()
    return num / den if den else float("nan")


def lift(d):
    n0, a0 = d[0]
    n1, a1 = d[1:].sum(axis=0)
    if not n0 or not n1:
        return float("nan"), float("nan"), float("nan")
    return a0 / n0, a1 / n1, (a1 / n1) / (a0 / n0)


def esegui(nome, azioni):
    seg, sgt = carica_follower()
    n = int(max(seg.max(), sgt.max())) + 1
    t_ado = carica_adozioni(n, azioni)
    ha = t_ado < np.iinfo(np.int64).max
    inizio, fine = int(t_ado[ha].min()), int(t_ado[ha].max())
    bins = np.arange(inizio, fine + ORA, ORA)
    bin_ado = np.where(ha, np.searchsorted(bins, t_ado, "right") - 1, -1)

    P(f"\n{'='*72}\n{nome}\n{'='*72}")
    P(
        f"  nodi {n:,}  archi follower {len(seg):,}  adottanti {int(ha.sum()):,} "
        f"({100*ha.mean():.1f}%)  ore {len(bins)}   [{time.time()-t0:.0f}s]"
    )

    d, str_r = curva(seg, sgt, n, bin_ado, len(bins))
    p0, p1, L = lift(d)
    Ls = lift_stratificato(str_r)
    P(
        f"  rete REALE      P(k=0)={p0:.6f}  P(k>=1)={p1:.6f}  lift={L:.2f}x"
        f"   [{time.time()-t0:.0f}s]"
    )
    P(f"                  stratificato per ora: lift={Ls:.2f}x")

    lifts, lifts_s = [], []
    for s in (1, 2, 3):
        perm = np.random.default_rng(s).permutation(len(sgt))
        dq, str_q = curva(seg, sgt[perm], n, bin_ado, len(bins))
        q0, q1, Lq = lift(dq)
        lifts.append(Lq)
        lifts_s.append(lift_stratificato(str_q))
        P(
            f"  permutata {s}     P(k=0)={q0:.6f}  P(k>=1)={q1:.6f}  "
            f"lift={Lq:.2f}x   [{time.time()-t0:.0f}s]"
        )

    mn, sd = float(np.mean(lifts)), float(np.std(lifts))
    ecc = L / mn
    P(f"\n  lift reale {L:.2f}x   lift nullo {mn:.2f}x (dev.st {sd:.4f})")
    P(f"  ECCESSO STRUTTURALE: {ecc:.2f}x  ->  {100*(ecc-1):+.0f}%")
    if sd > 0:
        P(f"  z rispetto al nullo: {(L-mn)/sd:.0f}")
    ms = float(np.mean(lifts_s))
    P(
        f"  stratificato per ora: reale {Ls:.2f}x  nullo {ms:.2f}x  ->  "
        f"{100*(Ls/ms-1):+.0f}%"
    )
    P("\n  curva dose-risposta (rete reale):")
    for s in range(SOGLIA + 1):
        nn, aa = d[s]
        if nn:
            P(
                f"    k={'6+' if s == SOGLIA else s}  a rischio {nn:>11,}  "
                f"P={aa/nn:.6f}  lift={(aa/nn)/p0:>7.2f}x"
            )
    return ecc


if __name__ == "__main__":
    r = {}
    if "tutte" not in sys.argv[1:]:
        r["solo retweet"] = esegui("ADOZIONE = PRIMO RETWEET", {"RT"})
    if "retweet" not in sys.argv[1:]:
        r["qualunque azione"] = esegui("ADOZIONE = PRIMA AZIONE (RT/RE/MT)", None)
    P(f"\n{'='*72}\nCANCELLO: soglia +40%")
    for k, e in r.items():
        P(f"  {k:20s} {100*(e-1):+7.0f}%   {'PASSA' if e >= 1.40 else 'NON PASSA'}")
