"""Fattore di non stazionarieta': quanto la trasmissivita' cambia dopo il taglio.

    tasso(h) = archi di follow che trasmettono nell'ora h
               / archi di follow a rischio nell'ora h (u attivo, v non ancora)

Il rapporto fra il tasso medio dopo il taglio e quello prima misura dal dato
la stessa quantita' che la taratura del regime chiama scala.
"""
import numpy as np

from asnm.higgs.dati import FINE_ORE, TAGLIO_ORE, costruisci


def tassi_orari(liv, fin) -> np.ndarray:
    """Per ogni ora, (archi che trasmettono, archi che potevano trasmettere)."""
    u, v, _ = liv.archi["F"]
    a = fin.adozione
    ora = np.where(a < fin.MAX, (a - fin.t0) // 3600, np.int64(1 << 40))

    fuori = np.zeros((FINE_ORE, 2), np.float64)
    for h in range(FINE_ORE):
        prima_u = ora[u] < h
        if not prima_u.any():
            continue
        a_rischio = prima_u & (ora[v] >= h)
        fuori[h, 1] = a_rischio.sum()
        fuori[h, 0] = (a_rischio & (ora[v] == h)).sum()
    return fuori


def esegui() -> dict:
    liv, fin = costruisci()
    t = tassi_orari(liv, fin)

    pre, post = t[:TAGLIO_ORE], t[TAGLIO_ORE:]
    tasso_pre = pre[:, 0].sum() / max(pre[:, 1].sum(), 1)
    tasso_post = post[:, 0].sum() / max(post[:, 1].sum(), 1)
    fattore = tasso_post / tasso_pre if tasso_pre else float("nan")

    print(f"[stazionarieta] archi-ora a rischio: "
          f"{pre[:, 1].sum():,.0f} prima del taglio, "
          f"{post[:, 1].sum():,.0f} dopo")
    print(f"  tasso di trasmissione per arco-ora, ore 0-{TAGLIO_ORE-1}: "
          f"{tasso_pre:.6f}")
    print(f"  tasso di trasmissione per arco-ora, ore {TAGLIO_ORE}-"
          f"{FINE_ORE-1}: {tasso_post:.6f}")
    print(f"\n  FATTORE DI NON STAZIONARIETA': {fattore:.2f}x")
    print(f"  La p stimata nel periodo calmo sottostima di {fattore:.2f} volte "
          f"quella del periodo da prevedere.")

    print(f"\n  {'ora':>5s} {'a rischio':>13s} {'trasmettono':>12s} {'tasso':>10s}")
    for h in list(range(0, TAGLIO_ORE, 12)) + list(range(TAGLIO_ORE, FINE_ORE, 12)):
        n, e = t[h, 1], t[h, 0]
        segno = "  <- taglio" if h == TAGLIO_ORE else ""
        if n:
            print(f"  {h:>5d} {n:>13,.0f} {e:>12,.0f} {e/n:>10.6f}{segno}")
    return {"tassi": t, "fattore": float(fattore),
            "pre": float(tasso_pre), "post": float(tasso_post)}


if __name__ == "__main__":
    esegui()
