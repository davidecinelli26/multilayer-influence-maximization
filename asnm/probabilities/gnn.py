"""GraphSAGE per estendere p(u,v) agli archi senza eventi.

Due strati di aggregazione (media sul vicinato) e una testa sulla coppia di
rappresentazioni:

    h1 = relu( W1 x + U1 A_norm x )
    h2 = relu( W2 h1 + U2 A_norm h1 )
    p  = sigmoid( MLP([ a*b, a+b, |a-b|, attributi dell'arco ]) )

Si addestra sugli archi ricchi con l'MSE, uniforme oppure pesato per il
grado del nodo di arrivo. Il bias d'uscita parte dal logit della media dei
bersagli.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from asnm.probabilities.credit import Stima

EPOCHE = 260
NASCOSTI = 64
USCITA = 32
LR = 3e-3
SEME = 11


class GraphSAGE(nn.Module):
    def __init__(self, entrata: int, arco: int, nascosti=NASCOSTI, uscita=USCITA):
        super().__init__()
        self.a1, self.b1 = nn.Linear(entrata, nascosti), nn.Linear(entrata, nascosti)
        self.a2, self.b2 = nn.Linear(nascosti, uscita), nn.Linear(nascosti, uscita)
        self.testa = nn.Sequential(
            nn.Linear(3 * uscita + arco, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def immersione(self, x, adiacenza):
        h = torch.relu(self.a1(x) + self.b1(torch.sparse.mm(adiacenza, x)))
        return torch.relu(self.a2(h) + self.b2(torch.sparse.mm(adiacenza, h)))

    def forward(self, x, adiacenza, sorgente, destinazione, attributi_arco):
        h = self.immersione(x, adiacenza)
        a, b = h[sorgente], h[destinazione]
        return self.testa(
            torch.cat([a * b, a + b, (a - b).abs(), attributi_arco], 1)
        ).squeeze(1)


def _adiacenza_normalizzata(st: Stima) -> torch.Tensor:
    n = st.n
    a = sp.coo_matrix(
        (np.ones(2 * len(st.u), np.float32),
         (np.r_[st.u, st.v], np.r_[st.v, st.u])), shape=(n, n)
    ).tocsr()
    gradi = np.asarray(a.sum(1)).ravel()
    gradi[gradi == 0] = 1
    norm = (sp.diags(1 / gradi) @ a).tocoo()
    return torch.sparse_coo_tensor(
        np.vstack((norm.row, norm.col)), torch.tensor(norm.data.astype(np.float32)),
        (n, n)
    ).coalesce()


def _campioni(st: Stima):
    """Archi orientati con bersaglio (i ricchi), in entrambe le direzioni."""
    r = st.ricco
    sorgente = np.r_[st.u[r], st.v[r]]
    destinazione = np.r_[st.v[r], st.u[r]]
    bersaglio = np.r_[st.p_uv[r], st.p_vu[r]]
    return sorgente, destinazione, bersaglio


def _attributi_arco(st: Stima, indici_archi) -> np.ndarray:
    """Il logaritmo del numero di adozioni dei due estremi (il denominatore
    della credit distribution)."""
    return np.log1p(np.stack([st.adozioni[indici_archi[0]],
                              st.adozioni[indici_archi[1]]], 1)).astype(np.float32)


def addestra(st: Stima, x: np.ndarray, pesata: bool = False,
             epoche: int = EPOCHE, seme: int = SEME):
    """Addestra sugli archi ricchi e predice su quelli poveri.

    Restituisce (p sui poveri, diagnostica).
    """
    torch.manual_seed(seme)
    t0 = time.time()
    adiacenza = _adiacenza_normalizzata(st)
    xt = torch.tensor(x)

    sorgente, destinazione, bersaglio = _campioni(st)
    attributi = _attributi_arco(st, (sorgente, destinazione))
    s = torch.tensor(sorgente.astype(np.int64))
    d = torch.tensor(destinazione.astype(np.int64))
    e = torch.tensor(attributi)
    y = torch.tensor(bersaglio.astype(np.float32))

    # peso per il grado del nodo di arrivo, normalizzato a media uno
    gradi = np.asarray(sp.csr_matrix(
        (np.ones(2 * len(st.u)), (np.r_[st.u, st.v], np.r_[st.v, st.u])),
        shape=(st.n, st.n)).sum(1)).ravel()
    peso = torch.tensor((gradi[destinazione] / gradi[destinazione].mean()
                         ).astype(np.float32))

    generatore = torch.Generator().manual_seed(seme)
    ordine = torch.randperm(len(y), generator=generatore)
    taglio = int(0.8 * len(y))
    tr, te = ordine[:taglio], ordine[taglio:]

    rete = GraphSAGE(x.shape[1], attributi.shape[1])

    # il bias di uscita parte dal logit della media dei bersagli: da zero
    # la rete partirebbe da 0,5 contro bersagli medi di 0,06 e non
    # arriverebbe alla costante ottima nel numero di epoche disponibili
    media = float(y[tr].mean().clamp(1e-6, 1 - 1e-6))
    with torch.no_grad():
        rete.testa[-1].bias.fill_(float(np.log(media / (1 - media))))
        rete.testa[-1].weight.mul_(0.1)

    ott = torch.optim.Adam(rete.parameters(), lr=LR)
    partenza = None
    for epoca in range(epoche):
        rete.train()
        ott.zero_grad()
        uscita = torch.sigmoid(rete(xt, adiacenza, s[tr], d[tr], e[tr]))
        errore = (uscita - y[tr]) ** 2
        perdita = (errore * peso[tr]).mean() if pesata else errore.mean()
        if epoca == 0:
            partenza = float(perdita)
        perdita.backward()
        ott.step()

    rete.eval()
    with torch.no_grad():
        o = torch.sigmoid(rete(xt, adiacenza, s[te], d[te], e[te]))
        costante = torch.full_like(o, float(y[tr].mean()))
        diagnostica = {
            "perdita_iniziale": partenza,
            "perdita_finale": float(perdita),
            "mse": float(((o - y[te]) ** 2).mean()),
            "mse_hub": float(((o - y[te]) ** 2 * peso[te]).sum() / peso[te].sum()),
            "mse_costante": float(((costante - y[te]) ** 2).mean()),
            "mse_hub_costante": float(
                ((costante - y[te]) ** 2 * peso[te]).sum() / peso[te].sum()),
        }
        poveri = ~st.ricco
        sp_ = torch.tensor(np.r_[st.u[poveri], st.v[poveri]].astype(np.int64))
        dp_ = torch.tensor(np.r_[st.v[poveri], st.u[poveri]].astype(np.int64))
        ep_ = torch.tensor(_attributi_arco(st, (sp_.numpy(), dp_.numpy())))
        p_poveri = torch.sigmoid(rete(xt, adiacenza, sp_, dp_, ep_)).numpy()

    diagnostica["secondi"] = time.time() - t0
    etichetta = "pesata " if pesata else "uniforme"
    calo = 100 * (1 - diagnostica["perdita_finale"] / diagnostica["perdita_iniziale"])
    print(f"[gnn] loss {etichetta}: MSE {diagnostica['mse']:.5f} "
          f"(costante {diagnostica['mse_costante']:.5f}), "
          f"MSE hub {diagnostica['mse_hub']:.5f} "
          f"(costante {diagnostica['mse_hub_costante']:.5f}), "
          f"perdita in addestramento {calo:+.1f}%, {diagnostica['secondi']:.0f}s")
    return p_poveri, diagnostica
