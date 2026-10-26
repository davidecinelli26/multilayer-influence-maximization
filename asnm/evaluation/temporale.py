"""Validazione su StackExchange: i seed contro gli adottanti veri.

    python -m asnm.evaluation.temporale [sito] [--rr=120000] [--k=50]

I seed si scelgono con dati fino al confine temporale; il punteggio di una
strategia su un tag e' la somma, sugli adottanti reali di quel tag nella
finestra di verifica, della probabilita' pi(v) di essere attivati partendo
dai seed. Bootstrap appaiato a grappoli sui tag.
"""
import sys
from collections import defaultdict

import numpy as np

from asnm.communities import fase1
from asnm.config import SITO_PRINCIPALE
from asnm.evaluation import attivita, bootstrap, montecarlo
from asnm.layers import costruisci as costruisci_livelli
from asnm.probabilities.fase2 import carica as carica_probabilita
from asnm.seeds import fase3, greedy
from asnm.splits import dividi

ALPHA = (1.00, 0.95, 0.90, 0.80, 0.70, 0.50)
SIMULAZIONI = 4000


def adottanti_per_tag(liv, div):
    """tag -> indici dei nodi che l'hanno adottato nella finestra, piu' il tetto."""
    per_tag = defaultdict(set)
    eventi = noti = 0
    for (utente, tag), _ in div.test.items():
        if tag not in div.bersagli:
            continue
        eventi += 1
        i = liv.indice.get(utente)
        if i is not None:
            noti += 1
            per_tag[tag].add(i)
    per_tag = {t: np.fromiter(s, int, len(s)) for t, s in per_tag.items() if len(s) >= 5}
    return per_tag, eventi, noti


def esegui(sito: str, quanti: int = 120_000, k: int = 50) -> dict:
    liv = costruisci_livelli(sito)
    div = dividi(sito)
    st, scala, varianti = carica_probabilita(sito)
    partizioni = fase1.candidate(liv)
    cid = np.unique(partizioni["L1+L2 fusione"], return_inverse=True)[1]

    p = st.matrice(p_poveri=varianti["gnn uniforme"], scala=scala)
    insiemi = fase3._insiemi(p, quanti, fase3.SEME_A, "gnn-uniforme", sito)

    per_tag, eventi, noti = adottanti_per_tag(liv, div)
    tag = sorted(per_tag)
    print("\nIL TETTO, PRIMA DI TUTTO")
    print(f"  adozioni reali dei bersagli nella finestra : {eventi:,}")
    print(f"  su utenti gia' nella rete nel 2019         : {noti:,} "
          f"({100*noti/eventi:.1f}%)")
    print(f"  utenti iscritti dopo il confine            : {eventi-noti:,} "
          f"({100*(1-noti/eventi):.1f}%) -- irraggiungibili per costruzione")
    print(f"  tag valutabili (>=5 adottanti noti)        : {len(tag)}")

    strategie = {}
    for a in ALPHA:
        strategie[f"alpha={a:.2f}"] = greedy.seleziona(
            insiemi, k, alpha=a, cid=cid,
            dimensioni=np.bincount(cid)).semi
    strategie["grado massimo"] = greedy.per_grado(p, k)
    strategie["casuale"] = greedy.a_caso(liv.n, k, seme=0)

    print(f"\ncalcolo di pi con {SIMULAZIONI:,} simulazioni per strategia...")
    punteggi, riepilogo = {}, {}
    for nome, semi in strategie.items():
        pi = montecarlo.attivazione(p, semi, SIMULAZIONI)
        punteggi[nome] = np.array([pi[per_tag[t]].sum() for t in tag])
        comunita_semi = len({int(cid[s]) for s in semi})
        riepilogo[nome] = (comunita_semi, float(punteggi[nome].sum()))

    denominatore = sum(len(per_tag[t]) for t in tag)
    print(f"\n{'strategia':26s} {'com. seed':>10s} {'adottanti attesi':>17s} "
          f"{'quota':>7s}")
    for nome, (com, tot) in riepilogo.items():
        print(f"{nome:26s} {com:>10} {tot:>17.1f} {100*tot/denominatore:>6.2f}%")

    riferimento = "alpha=1.00"
    print(f"\nBOOTSTRAP APPAIATO A GRAPPOLI SUI {len(tag)} TAG (3.000 repliche)")
    print(bootstrap.Confronto.intestazione())
    print("-" * 96)
    esiti = bootstrap.confronta(punteggi, riferimento)
    for e in esiti:
        print(e.riga())

    # ------------------------------------------- il declino, e la normalizzazione
    conteggi = attivita.base_attiva(sito)
    anni = defaultdict(list)
    for m, n in conteggi.items():
        if div.e_test(m):
            anni[m // 12 if m % 12 else m // 12 - 1].append(n)
    print("\nIL DECLINO DEL SITO DENTRO LA FINESTRA DI VERIFICA")
    for anno in sorted(anni):
        print(f"  {anno}: {np.mean(anni[anno]):>6.0f} utenti attivi al mese "
              f"(media su {len(anni[anno])} mesi)")
    if len(anni) > 1:
        primo, ultimo = sorted(anni)[0], sorted(anni)[-1]
        calo = 100 * (np.mean(anni[ultimo]) / np.mean(anni[primo]) - 1)
        print(f"  variazione dal {primo} al {ultimo}: {calo:+.0f}%")

    dimensioni = np.array([len(per_tag[t]) for t in tag], float)
    ponderazioni = attivita.pesi(div, tag, conteggi, dimensioni)
    print("\nLA CONCLUSIONE REGGE AL CAMBIO DI PONDERAZIONE?")
    print(f"  {'':26s}" + "".join(f"{n:^20s}" for n in ponderazioni))
    print(f"  {'strategia':26s}" + "".join(f"{'d%':>11s}{'p':>9s}"
                                           for _ in ponderazioni))
    tabella = {}
    for nome_peso, w in ponderazioni.items():
        tabella[nome_peso] = {
            e.nome.split("  ")[0]: e
            for e in bootstrap.confronta({n: v * w for n, v in punteggi.items()},
                                         riferimento)
        }
    for nome in punteggi:
        if nome == riferimento:
            continue
        riga = f"  {nome:26s}"
        for nome_peso in ponderazioni:
            e = tabella[nome_peso][nome]
            rel = 100 * e.differenza / e.base if e.base else 0.0
            riga += f"{rel:>10.2f}%{e.p:>9.3f}"
        print(riga)
    print("  Un fattore di scala globale si cancellerebbe nel confronto appaiato:"
          "\n  la ponderazione conta perche' cambia il peso *relativo* dei tag.")

    # ------------------------------------------- il modello invecchia?
    epoche = attivita.epoca_dei_tag(div, tag)
    # split per rango, non per valore: i mesi sono discreti e sulla mediana
    # ci sono molti pareggi, che sbilancerebbero i due gruppi
    ordine = np.argsort(epoche, kind="stable")
    prima = np.zeros(len(tag), bool)
    prima[ordine[: len(tag) // 2]] = True
    print("\nIL MODELLO INVECCHIA DENTRO LA FINESTRA?")
    for etichetta, maschera in (("prima meta'", prima), ("seconda meta'", ~prima)):
        rif = punteggi[riferimento][maschera].sum()
        gr = punteggi["grado massimo"][maschera].sum()
        print(f"  {etichetta:14s} ({int(maschera.sum()):>3} tag): "
              f"TIM+ {rif:>7.1f} contro grado {gr:>7.1f}  "
              f"vantaggio {100*(rif/gr-1):>+6.2f}%")
    print("  Se il vantaggio crollasse nella seconda meta', i seed scelti sul "
          "2019\n  perderebbero validita' col passare del tempo, e andrebbe detto.")

    print(f"\n{riferimento} e' TIM+ esatto: e' il riferimento perche' la domanda "
          "e' se il vincolo\ndi diversita' costi qualcosa sugli adottanti veri.")
    return {"punteggi": punteggi, "esiti": esiti, "tag": tag,
            "tetto": noti / eventi, "ponderazioni": tabella}


if __name__ == "__main__":
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    opzioni = dict(o[2:].split("=") for o in sys.argv[1:] if "=" in o)
    esegui(argomenti[0] if argomenti else SITO_PRINCIPALE,
           quanti=int(opzioni.get("rr", 120_000)), k=int(opzioni.get("k", 50)))
