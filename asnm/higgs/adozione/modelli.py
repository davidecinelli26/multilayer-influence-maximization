"""I modelli della scala a pioli.

Gradient boosting a iperparametri fissi per i gradini sugli attributi
tabellari; `propaga` precalcola [x, A x, A^2 x] sul grafo dei follower;
`sign` e' l'MLP su quegli attributi propagati (SIGN). `graphsage` e' il
classificatore a nodi end-to-end, tenuto come confronto: sulla scala di
questo grafo non converge in tempi utili.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

from asnm.higgs.gnn import SAGEveloce

# gradient boosting
ITERAZIONI = 300
FOGLIE = 31
PASSO = 0.08

# GraphSAGE a nodi
EPOCHE = 200
LR = 3e-3
PAZIENZA = 20
CONTROLLO = 4


def gradient_boosting(x_tr, y_tr, x_te, seme: int = 0):
    """Probabilita' sul test, con iperparametri fissi."""
    t0 = time.time()
    m = HistGradientBoostingClassifier(
        max_iter=ITERAZIONI, max_leaf_nodes=FOGLIE, learning_rate=PASSO,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=seme)
    m.fit(x_tr, y_tr)
    p = m.predict_proba(x_te)[:, 1]
    print(f"    gradient boosting: {m.n_iter_} iterazioni, "
          f"{time.time()-t0:.0f}s")
    return p


class SAGEnodi(nn.Module):
    def __init__(self, entrata: int):
        super().__init__()
        self.corpo = SAGEveloce(entrata, arco=0)
        self.testa = nn.Sequential(
            nn.Linear(self.corpo.a2.out_features, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, ax, adiacenza, indici):
        h = self.corpo.immersione_veloce(x, ax, adiacenza)
        return self.testa(h[indici]).squeeze(1)


def _adiacenza(liv) -> torch.Tensor:
    u, v, _ = liv.archi["F"]
    a = sp.coo_matrix((np.ones(len(u), np.float32), (u, v)),
                      shape=(liv.n, liv.n)).tocsr()
    a = a + a.T
    gradi = np.asarray(a.sum(1)).ravel()
    gradi[gradi == 0] = 1
    norm = (sp.diags(1 / gradi) @ a).tocoo()
    return torch.sparse_coo_tensor(
        np.vstack((norm.row, norm.col)),
        torch.tensor(norm.data.astype(np.float32)), (liv.n, liv.n)).coalesce()


def propaga(liv, x, passi: int = 2) -> np.ndarray:
    """[x, A x, A^2 x]: gli attributi del nodo, la media dei vicini, la media
    dei vicini dei vicini, con A l'adiacenza simmetrizzata normalizzata per
    riga. E' la propagazione precalcolata di SGC e SIGN."""
    u, v, _ = liv.archi["F"]
    a = sp.coo_matrix((np.ones(len(u), np.float32), (u, v)),
                      shape=(liv.n, liv.n)).tocsr()
    a = a + a.T
    gradi = np.asarray(a.sum(1)).ravel()
    gradi[gradi == 0] = 1
    norm = (sp.diags((1 / gradi).astype(np.float32)) @ a).tocsr()
    pezzi, h = [x], x
    for _ in range(passi):
        h = (norm @ h).astype(np.float32)
        pezzi.append(h)
    return np.hstack(pezzi)


class MLP(nn.Module):
    def __init__(self, entrata: int, nascosti: int = 128):
        super().__init__()
        self.rete = nn.Sequential(
            nn.Linear(entrata, nascosti), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(nascosti, nascosti // 2), nn.ReLU(),
            nn.Linear(nascosti // 2, 1))

    def forward(self, x):
        return self.rete(x).squeeze(1)


def sign(x_prop, idx_tr, y_tr, idx_va, y_va, idx_te, seme: int = 11,
         epoche: int = 400, lotto: int = 8192):
    """SIGN: MLP sugli attributi propagati, a mini-batch."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seme)
    t0 = time.time()
    mu = x_prop[idx_tr].mean(0, keepdims=True)
    sd = x_prop[idx_tr].std(0, keepdims=True) + 1e-6
    z = ((x_prop - mu) / sd).astype(np.float32)
    xtr, xva, xte = (torch.tensor(z[i]) for i in (idx_tr, idx_va, idx_te))
    ytr = torch.tensor(y_tr.astype(np.float32))

    rete = MLP(z.shape[1])
    base = float(np.clip(y_tr.mean(), 1e-6, 1 - 1e-6))
    with torch.no_grad():
        rete.rete[-1].bias.fill_(float(np.log(base / (1 - base))))
    perdita_fn = nn.BCEWithLogitsLoss()
    ott = torch.optim.Adam(rete.parameters(), lr=1e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seme)
    migliore, senza, fermata, stato = -1.0, 0, epoche, None
    for epoca in range(epoche):
        rete.train()
        for i in torch.randperm(len(ytr), generator=gen).split(lotto):
            ott.zero_grad()
            perdita_fn(rete(xtr[i]), ytr[i]).backward()
            ott.step()
        rete.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_va, torch.sigmoid(rete(xva)).numpy())
        if auc > migliore + 1e-5:
            migliore, senza = auc, 0
            stato = {k: w.clone() for k, w in rete.state_dict().items()}
        else:
            senza += 1
            if senza >= 15:
                fermata = epoca + 1
                break
    rete.load_state_dict(stato)
    rete.eval()
    with torch.no_grad():
        p = torch.sigmoid(rete(xte)).numpy()
    print(f"    SIGN seme {seme}: fermata all'epoca {fermata}, AUC di "
          f"validazione {migliore:.4f}, {time.time()-t0:.0f}s")
    return p, fermata


def graphsage(liv, x_tutti, idx_tr, y_tr, idx_va, y_va, idx_te,
              seme: int = 11, epoche: int = EPOCHE):
    """GraphSAGE a nodi, transduttiva: vede gli attributi di tutti i nodi e
    le etichette dei soli nodi di addestramento."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seme)
    t0 = time.time()

    # standardizzazione sui soli nodi di addestramento
    mu = x_tutti[idx_tr].mean(0, keepdims=True)
    sd = x_tutti[idx_tr].std(0, keepdims=True) + 1e-6
    xt = torch.tensor(((x_tutti - mu) / sd).astype(np.float32))

    adiacenza = _adiacenza(liv)
    with torch.no_grad():
        ax = torch.sparse.mm(adiacenza, xt)

    tr = torch.tensor(idx_tr.astype(np.int64))
    va = torch.tensor(idx_va.astype(np.int64))
    te = torch.tensor(idx_te.astype(np.int64))
    ytr = torch.tensor(y_tr.astype(np.float32))

    rete = SAGEnodi(xt.shape[1])
    base = float(np.clip(y_tr.mean(), 1e-6, 1 - 1e-6))
    with torch.no_grad():
        rete.testa[-1].bias.fill_(float(np.log(base / (1 - base))))
        rete.testa[-1].weight.mul_(0.1)

    perdita_fn = nn.BCEWithLogitsLoss()
    ott = torch.optim.Adam(rete.parameters(), lr=LR)
    migliore, senza, fermata, stato = -1.0, 0, epoche, None
    for epoca in range(epoche):
        rete.train()
        ott.zero_grad()
        perdita = perdita_fn(rete(xt, ax, adiacenza, tr), ytr)
        perdita.backward()
        ott.step()
        if (epoca + 1) % CONTROLLO == 0:
            rete.eval()
            with torch.no_grad():
                auc = roc_auc_score(y_va, torch.sigmoid(
                    rete(xt, ax, adiacenza, va)).numpy())
            if auc > migliore + 1e-5:
                migliore, senza = auc, 0
                stato = {k: v.clone() for k, v in rete.state_dict().items()}
            else:
                senza += CONTROLLO
                if senza >= PAZIENZA:
                    fermata = epoca + 1
                    break

    if stato is not None:
        rete.load_state_dict(stato)
    rete.eval()
    with torch.no_grad():
        p = torch.sigmoid(rete(xt, ax, adiacenza, te)).numpy()
    print(f"    GraphSAGE seme {seme}: fermata all'epoca {fermata}, "
          f"AUC di validazione {migliore:.4f}, {time.time()-t0:.0f}s")
    return p, fermata
