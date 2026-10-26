"""Il settimo gradino: aggregazione per tipo di relazione e per direzione (R-SIGN).

Dieci operatori di propagazione, ciascuno con un proprio blocco di pesi:
x, A_in x, A_out x, le quattro combinazioni a due passi, e A_in A_l x per
l in {RT, MT, RE}. Gli strati di interazione entrano solo a due passi perche'
un nodo non ancora attivo non ha archi RT/MT/RE propri. Il modello e' il
primo strato di un R-GCN con propagazione precalcolata, nello spirito di SIGN.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def _normalizza_righe(a: sp.csr_matrix) -> sp.csr_matrix:
    gradi = np.asarray(a.sum(1)).ravel()
    gradi[gradi == 0] = 1
    return (sp.diags((1 / gradi).astype(np.float32)) @ a).tocsr()


def operatori(liv) -> dict:
    """Le matrici di propagazione, una per relazione, normalizzate per riga.

    Gli archi dei livelli sono memorizzati nel verso dell'influenza
    (u -> v: v segue u, oppure v ha ritwittato u). `A_in[v, u] = 1` significa
    quindi «u influenza v»: la riga v aggrega sui nodi che v vede.
    """
    n = liv.n
    fuori = {}
    u, v, _ = liv.archi["F"]
    uno = np.ones(len(u), np.float32)
    a_in = sp.csr_matrix((uno, (v, u)), shape=(n, n))      # riga = chi segue
    fuori["in"] = _normalizza_righe(a_in)
    fuori["out"] = _normalizza_righe(a_in.T.tocsr())        # riga = chi e' seguito
    for nome in ("RT", "MT", "RE"):
        uu, vv, _ = liv.archi[nome]
        m = sp.csr_matrix((np.ones(len(uu), np.float32), (vv, uu)), shape=(n, n))
        fuori[nome] = _normalizza_righe((m + m.T).tocsr())
    return fuori


def propaga(liv, x: np.ndarray) -> tuple[np.ndarray, list]:
    """[x, A_in x, A_out x, A_in A_in x, ..., A_in A_RE x] e i nomi dei blocchi."""
    t0 = time.time()
    op = operatori(liv)
    x = x.astype(np.float32)
    h_in, h_out = op["in"] @ x, op["out"] @ x
    blocchi = [("x", x), ("in", h_in), ("out", h_out),
               ("in.in", op["in"] @ h_in), ("in.out", op["in"] @ h_out),
               ("out.in", op["out"] @ h_in), ("out.out", op["out"] @ h_out)]
    for nome in ("RT", "MT", "RE"):
        blocchi.append((f"in.{nome}", op["in"] @ (op[nome] @ x)))
    nomi = [n for n, _ in blocchi]
    fuori = np.hstack([b.astype(np.float32) for _, b in blocchi])
    print(f"[relazionale] {len(blocchi)} blocchi x {x.shape[1]} attributi = "
          f"{fuori.shape[1]} colonne  [{time.time()-t0:.0f}s]")
    return fuori, nomi


class RSIGN(nn.Module):
    """Un blocco lineare per relazione (le W_r del primo strato di un R-GCN),
    poi lo stesso MLP di SIGN sulla concatenazione."""

    def __init__(self, blocchi: int, per_blocco: int, dim: int = 32,
                 nascosti: int = 128):
        super().__init__()
        self.blocchi, self.per_blocco = blocchi, per_blocco
        self.proiezioni = nn.ModuleList(
            [nn.Linear(per_blocco, dim) for _ in range(blocchi)])
        self.testa = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(blocchi * dim, nascosti), nn.ReLU(),
            nn.Linear(nascosti, nascosti // 2), nn.ReLU(),
            nn.Linear(nascosti // 2, 1))

    def forward(self, x):
        pezzi = [p(x[:, i * self.per_blocco:(i + 1) * self.per_blocco])
                 for i, p in enumerate(self.proiezioni)]
        return self.testa(torch.cat(pezzi, 1)).squeeze(1)


def r_sign(x_prop, blocchi: int, idx_tr, y_tr, idx_va, y_va, idx_te,
           seme: int = 11, epoche: int = 400, lotto: int = 8192):
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seme)
    t0 = time.time()
    mu = x_prop[idx_tr].mean(0, keepdims=True)
    sd = x_prop[idx_tr].std(0, keepdims=True) + 1e-6
    z = ((x_prop - mu) / sd).astype(np.float32)
    xtr, xva, xte = (torch.tensor(z[i]) for i in (idx_tr, idx_va, idx_te))
    ytr = torch.tensor(y_tr.astype(np.float32))

    rete = RSIGN(blocchi, z.shape[1] // blocchi)
    base = float(np.clip(y_tr.mean(), 1e-6, 1 - 1e-6))
    with torch.no_grad():
        rete.testa[-1].bias.fill_(float(np.log(base / (1 - base))))
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
    print(f"    R-SIGN seme {seme}: fermata all'epoca {fermata}, AUC di "
          f"validazione {migliore:.4f}, {time.time()-t0:.0f}s")
    return p, fermata, rete
