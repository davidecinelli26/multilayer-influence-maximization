"""GraphSAGE sugli archi di Higgs: estende p(u,v) agli archi senza trasmissioni.

La rete e' quella di `asnm/probabilities/gnn.py`. Tre scelte di contorno:
  * si addestra su un solo verso per arco (il verso opposto non e' un canale);
  * si campionano NEGATIVI_PER_RICCO archi senza trasmissione, con bersaglio
    zero, per ogni arco ricco, altrimenti la rete sovrastima sui poveri;
  * divisione 70/15/15 degli archi campionati, arresto anticipato sulla
    validazione, MSE riportato sul test.
Il prodotto sparso del primo strato, A x, non cambia durante l'addestramento
e si calcola una volta (`SAGEveloce`). La predizione sui 14,8 milioni di
archi poveri va a blocchi.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch

from asnm.probabilities.gnn import GraphSAGE

EPOCHE = 300
LR = 3e-3
SEME = 11
NEGATIVI_PER_RICCO = 4
"""Quanti archi senza trasmissione si campionano per ogni arco ricco."""
BLOCCO = 1_000_000
PAZIENZA = 25        # epoche senza miglioramento prima dell'arresto
CONTROLLO = 5        # ogni quante epoche si guarda la validazione


class SAGEveloce(GraphSAGE):
    """La stessa rete, con il prodotto sparso del primo strato (A x, costante
    durante l'addestramento) calcolato una volta sola. E' una riscrittura
    esatta: la rete calcola la stessa funzione."""

    def immersione_veloce(self, x, ax, adiacenza):
        h = torch.relu(self.a1(x) + self.b1(ax))
        return torch.relu(self.a2(h) + self.b2(torch.sparse.mm(adiacenza, h)))

    def avanti(self, x, ax, adiacenza, sorgente, destinazione, attributi):
        h = self.immersione_veloce(x, ax, adiacenza)
        a, b = h[sorgente], h[destinazione]
        return self.testa(
            torch.cat([a * b, a + b, (a - b).abs(), attributi], 1)).squeeze(1)


def _adiacenza(st) -> torch.Tensor:
    """Adiacenza simmetrica normalizzata per grado (il canale di propagazione
    degli attributi)."""
    a = sp.coo_matrix((np.ones(len(st.src), np.float32), (st.src, st.dst)),
                      shape=(st.n, st.n)).tocsr()
    a = a + a.T
    gradi = np.asarray(a.sum(1)).ravel()
    gradi[gradi == 0] = 1
    norm = (sp.diags(1 / gradi) @ a).tocoo()
    return torch.sparse_coo_tensor(
        np.vstack((norm.row, norm.col)),
        torch.tensor(norm.data.astype(np.float32)), (st.n, st.n)).coalesce()


def _attributi_arco(st, s, d) -> np.ndarray:
    """Il logaritmo delle attivita' dei due estremi (il denominatore della
    credit distribution)."""
    return np.log1p(np.stack([st.attivita[s], st.attivita[d]], 1)).astype(np.float32)


def _campioni(st, rng):
    ricchi = np.flatnonzero(st.ricco)
    poveri = np.flatnonzero(~st.ricco)
    quanti = min(len(poveri), NEGATIVI_PER_RICCO * len(ricchi))
    negativi = rng.choice(poveri, quanti, replace=False)
    indici = np.concatenate([ricchi, negativi])
    bersaglio = np.concatenate([st.p[ricchi], np.zeros(quanti)])
    return indici, bersaglio.astype(np.float32)


def addestra(st, x: np.ndarray, pesata: bool = False, epoche: int = EPOCHE,
             seme: int = SEME):
    """Addestra e predice su tutti gli archi poveri. (p sui poveri, diagnostica)."""
    torch.manual_seed(seme)
    rng = np.random.default_rng(seme)
    t0 = time.time()

    adiacenza = _adiacenza(st)
    xt = torch.tensor(x)

    indici, y_np = _campioni(st, rng)
    s_np, d_np = st.src[indici], st.dst[indici]
    e_np = _attributi_arco(st, s_np, d_np)

    s = torch.tensor(s_np.astype(np.int64))
    d = torch.tensor(d_np.astype(np.int64))
    e = torch.tensor(e_np)
    y = torch.tensor(y_np)

    gradi = np.bincount(st.dst, minlength=st.n).astype(np.float32)
    peso = torch.tensor(gradi[d_np] / max(gradi[d_np].mean(), 1e-9))

    # tre insiemi: l'arresto anticipato guarda la validazione, l'MSE il test
    generatore = torch.Generator().manual_seed(seme)
    ordine = torch.randperm(len(y), generator=generatore)
    a, b = int(0.70 * len(y)), int(0.85 * len(y))
    tr, va, te = ordine[:a], ordine[a:b], ordine[b:]

    rete = SAGEveloce(x.shape[1], e_np.shape[1])
    with torch.no_grad():
        ax = torch.sparse.mm(adiacenza, xt)   # costante: si calcola una volta
    media = float(y[tr].mean().clamp(1e-6, 1 - 1e-6))
    with torch.no_grad():
        rete.testa[-1].bias.fill_(float(np.log(media / (1 - media))))
        rete.testa[-1].weight.mul_(0.1)

    def perdita_su(indici):
        uscita = torch.sigmoid(rete.avanti(xt, ax, adiacenza, s[indici],
                                           d[indici], e[indici]))
        errore = (uscita - y[indici]) ** 2
        return (errore * peso[indici]).mean() if pesata else errore.mean()

    ott = torch.optim.Adam(rete.parameters(), lr=LR)
    partenza, migliore, senza_progresso, fermata = None, float("inf"), 0, epoche
    for epoca in range(epoche):
        rete.train()
        ott.zero_grad()
        perdita = perdita_su(tr)
        if epoca == 0:
            partenza = float(perdita.detach())
        perdita.backward()
        ott.step()

        # arresto anticipato sulla validazione
        if (epoca + 1) % CONTROLLO == 0:
            rete.eval()
            with torch.no_grad():
                v = float(perdita_su(va))
            if v < migliore - 1e-9:
                migliore, senza_progresso = v, 0
            else:
                senza_progresso += CONTROLLO
                if senza_progresso >= PAZIENZA:
                    fermata = epoca + 1
                    break

    rete.eval()
    with torch.no_grad():
        o = torch.sigmoid(rete.avanti(xt, ax, adiacenza, s[te], d[te], e[te]))
        costante = torch.full_like(o, float(y[tr].mean()))
        diagnostica = {
            "perdita_iniziale": partenza,
            "perdita_finale": float(perdita.detach()),
            "mse": float(((o - y[te]) ** 2).mean()),
            "mse_hub": float(((o - y[te]) ** 2 * peso[te]).sum() / peso[te].sum()),
            "mse_costante": float(((costante - y[te]) ** 2).mean()),
            "mse_hub_costante": float(
                ((costante - y[te]) ** 2 * peso[te]).sum() / peso[te].sum()),
        }
        # l'immersione dei nodi si calcola una volta; la testa va a blocchi
        h = rete.immersione_veloce(xt, ax, adiacenza)
        poveri = np.flatnonzero(~st.ricco)
        p_poveri = np.empty(len(poveri), np.float32)
        for i in range(0, len(poveri), BLOCCO):
            blocco = poveri[i:i + BLOCCO]
            sb = torch.tensor(st.src[blocco].astype(np.int64))
            db = torch.tensor(st.dst[blocco].astype(np.int64))
            eb = torch.tensor(_attributi_arco(st, st.src[blocco], st.dst[blocco]))
            a, b = h[sb], h[db]
            uscita = rete.testa(
                torch.cat([a * b, a + b, (a - b).abs(), eb], 1)).squeeze(1)
            p_poveri[i:i + len(blocco)] = torch.sigmoid(uscita).numpy()

    diagnostica["secondi"] = time.time() - t0
    diagnostica["epoche"] = fermata
    etichetta = "pesata " if pesata else "uniforme"
    calo = 100 * (1 - diagnostica["perdita_finale"] / diagnostica["perdita_iniziale"])
    print(f"[gnn-higgs] loss {etichetta}: MSE {diagnostica['mse']:.6f} "
          f"(costante {diagnostica['mse_costante']:.6f}), "
          f"MSE hub {diagnostica['mse_hub']:.6f} "
          f"(costante {diagnostica['mse_hub_costante']:.6f}), "
          f"perdita in addestramento {calo:+.1f}%, "
          f"{fermata} epoche, {diagnostica['secondi']:.0f}s")
    return p_poveri, diagnostica
