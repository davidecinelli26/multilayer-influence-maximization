"""Influence maximization competitiva su StackExchange, con NetMax.

Due agenti scelgono i seed a turni alternati sulla stessa rete; chi arriva
per primo su un nodo se lo prende. Lo sfidante usa il vincolo di diversita'
(alpha < 1), l'avversario TIM+ esatto; la metrica e' la quota di nodi
attivati dallo sfidante. Con `EVITA_AVVERSARI` le comunita' occupate
dall'avversario contano come gia' toccate.

    python -m asnm.seeds.competitivo [sito] [--k=5] [--sim=400] [--rip=5]
"""
import sys
import time

import networkx as nx
import numpy as np
from netmax.algorithms.sketch_based.timp import TIMp
from netmax.influence_maximization import InfluenceMaximization

from asnm.communities import fase1
from asnm.config import SITO_PRINCIPALE
from asnm.layers import costruisci as costruisci_livelli
from asnm.probabilities.fase2 import carica as carica_probabilita
from asnm.seeds import greedy


class DiversityTIM(TIMp):
    """TIM+ con il vincolo di DTIM, applicato per agente. Lo stato e' di
    classe perche' NetMax istanzia l'algoritmo con una firma fissata."""

    name = "diversity_tim"

    ALPHA_PER_AGENTE: dict = {}      # id agente -> alpha
    COMUNITA: dict = {}              # nodo (etichetta NetMax) -> comunita'
    DIMENSIONI: dict = {}            # comunita' -> quanti nodi
    MINIMO: int = greedy.MINIMO
    EVITA_AVVERSARI: bool = False

    def __node_selection__(self, agents):
        alpha = self.ALPHA_PER_AGENTE.get(self.curr_agent_id, 1.0)
        if alpha >= 1.0 or not self.COMUNITA:
            return super().__node_selection__(agents)

        seed = agents[self.curr_agent_id].seed
        occupate = list(seed)
        if self.EVITA_AVVERSARI:
            occupate = [s for a in agents for s in a.seed]
        coperte = {self.COMUNITA.get(s) for s in occupate}

        massimo = max((len(r) for r in self.occurrences.values()), default=0) or 1

        def punteggio(voce):
            nodo, insiemi = voce
            com = self.COMUNITA.get(nodo)
            vale = com is not None and self.DIMENSIONI.get(com, 0) >= self.MINIMO
            nuova = 1.0 if (vale and com not in coperte) else 0.0
            return alpha * (len(insiemi) / massimo) + (1 - alpha) * nuova

        top = max(self.occurrences.items(), key=punteggio)[0]
        seed.append(top)
        self.rr_sets = [r for i, r in enumerate(self.rr_sets)
                        if i not in self.occurrences[top]]
        self.occurrences = {
            v: self.occurrences[v].difference(self.occurrences[top])
            for v in self.graph.nodes if not self.__in_some_seed_set__(v, agents)
        }


def grafo_diretto(st, varianti, scala) -> nx.DiGraph:
    """Il grafo che NetMax si aspetta: orientato, con l'attributo `p`."""
    p = st.matrice(p_poveri=varianti["gnn uniforme"], scala=scala).tocoo()
    g = nx.DiGraph()
    g.add_weighted_edges_from(zip(p.row.tolist(), p.col.tolist(),
                                  p.data.tolist()), weight="p")
    return g


def prepara(sito: str, nome_partizione: str = "L1+L2 fusione"):
    """Grafo e partizione, caricati una volta sola per tutte le partite."""
    liv = costruisci_livelli(sito)
    st, scala, varianti = carica_probabilita(sito)
    cid = np.unique(fase1.candidate(liv)[nome_partizione], return_inverse=True)[1]
    return grafo_diretto(st, varianti, scala), cid, np.bincount(cid)


def gioca(g, cid, dimensioni, k: int, alpha_sfidante: float,
          evita: bool = False, simulazioni: int = 100,
          sfidante_primo: bool = False):
    """Una partita fra TIM+ esatto e il vincolo di diversita'.

    `sfidante_primo` scambia l'ordine dei turni: NetMax fa muovere gli agenti
    nell'ordine in cui compaiono, e la prima mossa vale molto.
    """
    nomi = ["diversita'", "TIM+"] if sfidante_primo else ["TIM+", "diversita'"]
    im = InfluenceMaximization(
        input_graph=g.copy(), agents={nomi[0]: k, nomi[1]: k},
        alg=DiversityTIM, diff_model="ic", inf_prob=None,
        endorsement_policy="random", r=simulazioni, verbose=False,
    )
    # NetMax rietichetta i nodi togliendo gli isolati: la partizione va
    # tradotta nelle sue etichette, non nelle nostre
    DiversityTIM.COMUNITA = {nuovo: int(cid[vecchio])
                             for vecchio, nuovo in im.mapping.items()}
    DiversityTIM.DIMENSIONI = {c: int(n) for c, n in enumerate(dimensioni)}
    id_sfidante = nomi.index("diversita'")
    DiversityTIM.ALPHA_PER_AGENTE = {1 - id_sfidante: 1.0,
                                     id_sfidante: alpha_sfidante}
    DiversityTIM.EVITA_AVVERSARI = evita

    t0 = time.time()
    semi, quote, _ = im.run()
    return semi, quote, time.time() - t0


def esegui(sito: str, k: int = 5, simulazioni: int = 400,
           ripetizioni: int = 5):
    """Piu' partite per configurazione, con media e dispersione delle quote."""
    g, cid, dimensioni = prepara(sito)
    print(f"[competitivo] {sito}: due agenti con budget {k} ciascuno, turni "
          f"alternati.\n  {ripetizioni} partite per configurazione, "
          f"{simulazioni} simulazioni per stimare le quote.")
    print(f"\n{'alpha':>6s} {'variante':>17s} {'com. sfid.':>10s} "
          f"{'scarto ± err':>15s} {'quota sfid.':>14s} {'disp.':>12s}  esito")

    def partite(alpha, evita, quante):
        """Meta' partite con lo sfidante che muove per secondo, meta' per primo:
        il vantaggio della prima mossa va bilanciato."""
        quote, comunita = [], []
        for i in range(quante):
            sfidante_primo = i % 2 == 1
            semi, punteggi, _ = gioca(g, cid, dimensioni, k, alpha, evita,
                                      simulazioni, sfidante_primo)
            a, b = punteggi["TIM+"], punteggi["diversita'"]
            quote.append(b / (a + b) if a + b else 0.5)
            comunita.append(len({int(cid[s]) for s in semi["diversita'"]}))
        return np.array(quote), float(np.mean(comunita))

    # riferimento: entrambi gli agenti giocano la stessa strategia; il metro
    # e' l'errore standard della media
    base, com_base = partite(1.00, False, 2 * ripetizioni)
    err_base = base.std(ddof=1) / np.sqrt(len(base))
    print(f"{1.00:>6.2f} {'riferimento':>17s} {com_base:>10.1f} "
          f"{100*base.mean():>8.1f}% ± {100*err_base:<4.1f} "
          f"{'':>13s} {100*base.std(ddof=1):>11.1f}%  stessa strategia")
    print()

    esiti = {(False, 1.00): base}
    for evita in (False, True):
        for alpha in (0.90, 0.80, 0.70, 0.50):
            q, com = partite(alpha, evita, ripetizioni)
            esiti[(evita, alpha)] = q
            scarto = q.mean() - base.mean()
            errore = np.sqrt(q.var(ddof=1) / len(q) + err_base ** 2)
            verdetto = ("MEGLIO" if scarto > 2 * errore else
                        "PEGGIO" if scarto < -2 * errore else "dentro il rumore")
            etichetta = "evita avversario" if evita else "solo se stesso"
            print(f"{alpha:>6.2f} {etichetta:>17s} {com:>10.1f} "
                  f"{100*scarto:>+8.1f}% ± {100*errore:<4.1f} "
                  f"{100*q.mean():>13.1f}% {100*q.std(ddof=1):>11.1f}%  {verdetto}")
        print()

    print(f"Riferimento {100*base.mean():.1f}%: la quota dello sfidante quando "
          f"gioca la stessa\nstrategia dell'avversario. Uno scarto conta solo "
          f"se supera due volte il proprio\nerrore standard, riportato accanto "
          f"a ogni riga.")
    return esiti


if __name__ == "__main__":
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(argomenti[0] if argomenti else SITO_PRINCIPALE,
           k=int(opzioni.get("k", 5)),
           simulazioni=int(opzioni.get("sim", 400)),
           ripetizioni=int(opzioni.get("rip", 5)))
