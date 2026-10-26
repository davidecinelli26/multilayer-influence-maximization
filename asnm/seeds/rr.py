"""Insiemi RR: campionamento a raggiungibilita' inversa.

Si estrae una radice e si visita il grafo all'indietro tenendo ogni arco
entrante con la sua probabilita'; l'insieme raggiunto e' chi avrebbe potuto
attivare la radice. La diffusione attesa di un insieme di seed e'
proporzionale alla frazione di insiemi RR che interseca. Con `pesi_radici`
le radici si estraggono in proporzione ai pesi (influence maximization
pesata sui nodi).
"""
import time

import numpy as np
import scipy.sparse as sp


def genera(p: sp.csr_matrix, quanti: int, seme: int = 1,
           silenzioso: bool = False, pesi_radici=None):
    """Matrice sparsa (quanti x n): riga i = l'insieme RR i-esimo.

    `p` e' orientata con p[u, v] = probabilita' che u attivi v; la visita va
    all'indietro, quindi si lavora sulla trasposta. `pesi_radici`, se dato,
    e' un vettore non negativo lungo n da cui estrarre le radici in
    proporzione.
    """
    n = p.shape[0]
    indietro = p.T.tocsr()
    ind, ptr, dati = indietro.indices, indietro.indptr, indietro.data
    rng = np.random.default_rng(seme)
    t0 = time.time()

    if pesi_radici is not None:
        w = np.asarray(pesi_radici, np.float64)
        radici = rng.choice(n, size=quanti, p=w / w.sum())
    else:
        radici = rng.integers(0, n, size=quanti)

    righe, colonne = [], []
    for i in range(quanti):
        radice = int(radici[i])
        visti = {radice}
        pila = [radice]
        while pila:
            x = pila.pop()
            a, b = ptr[x], ptr[x + 1]
            if a == b:
                continue
            for y in ind[a:b][rng.random(b - a) < dati[a:b]]:
                y = int(y)
                if y not in visti:
                    visti.add(y)
                    pila.append(y)
        righe.extend([i] * len(visti))
        colonne.extend(visti)

    insiemi = sp.csr_matrix(
        (np.ones(len(righe), np.int8), (righe, colonne)), shape=(quanti, n)
    ).tocsc()
    dimensione = len(righe) / quanti
    if not silenzioso:
        print(f"[rr] {quanti:,} insiemi RR, dimensione media {dimensione:.2f}, "
              f"{time.time()-t0:.1f}s")
    return insiemi, dimensione


def dimensione_media(p: sp.csr_matrix, campione: int = 600, seme: int = 5) -> float:
    """Solo la dimensione media, per la taratura del regime."""
    return genera(p, campione, seme, silenzioso=True)[1]
