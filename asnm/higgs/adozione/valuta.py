"""AUC e average precision sul test, con bootstrap appaiato a grappoli sulle
comunita' e correzione di Bonferroni sui confronti fra gradini.

Il ricampionamento e' per pesi: estrarre le comunita' con reinserimento
equivale a pesare ogni nodo per quante volte la sua comunita' e' uscita, e
`roc_auc_score` accetta il peso come `sample_weight`.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class Confronto:
    da: str
    a: str
    auc_da: float
    auc_a: float
    differenza: float
    basso: float
    alto: float
    p: float
    confronti: int

    @property
    def soglia(self) -> float:
        return 0.05 / max(self.confronti, 1)

    def riga(self) -> str:
        if self.p >= 0.05:
            esito = "equivalenti"
        elif self.p < self.soglia:
            esito = "MIGLIORE"
        else:
            esito = f"al limite (cade a {self.soglia:.4f})"
        return (f"{self.da:>12s} -> {self.a:<12s} {self.auc_da:>7.4f} "
                f"{self.auc_a:>7.4f} {self.differenza:>+8.4f}  "
                f"[{self.basso:>+7.4f}, {self.alto:>+7.4f}]  {self.p:>6.3f}  "
                f"{esito}")

    @staticmethod
    def intestazione() -> str:
        return (f"{'gradino':>12s}    {'':12s} {'AUC da':>7s} {'AUC a':>7s} "
                f"{'diff.':>8s}  {'IC 95%':^18s}  {'p':>6s}  esito")


def metriche(y, p) -> tuple[float, float]:
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def bootstrap(y, previsioni: dict, comunita_test, coppie: list,
              repliche: int = 2000, seme: int = 3) -> list:
    """`previsioni`: nome -> p sul test. `coppie`: [(da, a), ...]."""
    _, grappolo = np.unique(comunita_test, return_inverse=True)
    n_gr = grappolo.max() + 1
    rng = np.random.default_rng(seme)
    nomi = list(previsioni)
    campioni = {n: np.empty(repliche) for n in nomi}
    for b in range(repliche):
        estratte = rng.integers(0, n_gr, n_gr)
        peso = np.bincount(estratte, minlength=n_gr)[grappolo].astype(float)
        if peso[y == 1].sum() == 0 or peso[y == 0].sum() == 0:
            estratte = np.arange(n_gr)
            peso = np.ones(len(y))
        for n in nomi:
            campioni[n][b] = roc_auc_score(y, previsioni[n], sample_weight=peso)

    esiti = []
    for da, a in coppie:
        d = campioni[a] - campioni[da]
        basso, alto = np.percentile(d, [2.5, 97.5])
        p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
        esiti.append(Confronto(
            da, a, float(roc_auc_score(y, previsioni[da])),
            float(roc_auc_score(y, previsioni[a])),
            float(roc_auc_score(y, previsioni[a]) - roc_auc_score(y, previsioni[da])),
            float(basso), float(alto), min(p, 1.0), len(coppie)))
    return esiti
