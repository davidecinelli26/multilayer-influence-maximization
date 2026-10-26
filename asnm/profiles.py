"""Profili tematici degli utenti StackExchange: il vettore dei tag usati
prima del confine. Servono come verita' di riferimento nella Fase 1 (senza i
tag bersaglio) e come attributi dei nodi nella Fase 2 (politica in `splits`).
"""
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from asnm.adoption import leggi_post, tag_del_post
from asnm.layers import Livelli
from asnm.splits import Divisione, dividi


@dataclass
class Profili:
    matrice: sp.csr_matrix    # n_nodi x n_tag, conteggi d'uso
    tag: np.ndarray           # colonna -> nome del tag

    def dominante(self, min_usi: int = 5):
        """(indici dei nodi valutabili, etichetta intera del tag dominante).

        Sotto `min_usi` il tag dominante e' rumore: un utente con due post
        non ha un profilo tematico, ha due post.
        """
        m = self.matrice
        totali = np.asarray(m.sum(axis=1)).ravel()
        nodi = np.flatnonzero(totali >= min_usi)
        if len(nodi) == 0:
            return nodi, np.array([], dtype=int)
        sotto = m[nodi]
        # argmax per riga su matrice sparsa
        dominanti = np.asarray(sotto.argmax(axis=1)).ravel()
        _, etichette = np.unique(dominanti, return_inverse=True)
        return nodi, etichette


def costruisci(liv: Livelli, per_verita: bool = True,
               div: Divisione | None = None) -> Profili:
    """Profili sui nodi di `liv`, con la politica di mascheramento richiesta."""
    d = div or dividi(liv.sito)
    ammesso = d.ammesso_come_verita if per_verita else d.ammesso_per_stima

    post = leggi_post(liv.sito)
    t0 = time.time()
    righe, colonne, colonna_di = [], [], {}
    for id_post, quando in post.quando.items():
        if not Divisione.e_train(quando):
            continue
        utente = post.proprietario.get(id_post)
        i = liv.indice.get(utente) if utente else None
        if i is None:
            continue
        tg = tag_del_post(post, id_post)
        if not tg:
            continue
        for t in tg:
            if not ammesso(t):
                continue
            righe.append(i)
            colonne.append(colonna_di.setdefault(t, len(colonna_di)))

    m = sp.csr_matrix(
        (np.ones(len(righe), np.float32), (righe, colonne)),
        shape=(liv.n, len(colonna_di)),
    )
    tag = np.empty(len(colonna_di), dtype=object)
    for t, j in colonna_di.items():
        tag[j] = t
    politica = "verita' di riferimento" if per_verita else "attributi"
    print(f"[profiles] {liv.sito}: {m.shape[0]:,} nodi x {m.shape[1]:,} tag "
          f"({politica}) in {time.time()-t0:.0f}s")
    return Profili(m, tag)


if __name__ == "__main__":
    import sys

    from asnm.config import SITO_PRINCIPALE
    from asnm.layers import costruisci as costruisci_livelli

    sito = sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE
    liv = costruisci_livelli(sito)
    p = costruisci(liv)
    nodi, etichette = p.dominante()
    print(f"\nnodi con almeno 5 usi di tag : {len(nodi):,}")
    print(f"tag dominanti distinti       : {etichette.max()+1:,}")

    colonne = np.asarray(p.matrice[nodi].argmax(axis=1)).ravel()
    conta = np.bincount(colonne, minlength=p.matrice.shape[1])
    print("\n  tag dominanti piu' frequenti:")
    for j in np.argsort(-conta)[:6]:
        print(f"    {p.tag[j]:28s} {conta[j]:>6,} utenti")

    d = dividi(sito)
    intrusi = [t for t in p.tag if t in d.bersagli]
    print(f"\n  tag bersaglio nel profilo: {len(intrusi)} "
          f"({'corretto' if not intrusi else 'ERRORE: leakage'})")
