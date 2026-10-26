"""Fase 1 su StackExchange, dall'inizio alla fine.

    python -m asnm.communities.fase1 [sito] [--senza-emcd] [--repliche=N]

Costruisce le candidate, le valuta, e con il bootstrap appaiato a grappoli
sui tag dice se il margine fra le migliori e' reale.
"""
import sys

import numpy as np

from asnm.communities import bootstrap, evaluate, exposure, single
from asnm.config import CACHE, SITO_PRINCIPALE
from asnm.layers import costruisci as costruisci_livelli
from asnm.profiles import costruisci as costruisci_profili
from asnm.splits import dividi

THETA = (0.4, 0.5)
"""Due soglie per M-EMCD: una da' una partizione sensata, l'altra una quasi
degenere, tenuta come controllo sulle metriche."""


def _cache(sito):
    return CACHE / f"partizioni_{sito}.npz"


def candidate(liv, usa_emcd: bool = True, usa_cache: bool = True) -> dict:
    percorso = _cache(liv.sito)
    if usa_cache and percorso.exists():
        d = np.load(percorso, allow_pickle=False)
        print(f"[fase1] partizioni lette dalla cache ({len(d.files)} candidate)")
        return {k: d[k] for k in d.files}

    fuori = single.candidate(liv)
    if usa_emcd:
        from asnm.communities import emcd

        if not emcd.verifica_aucs():
            raise SystemExit("il wrapper M-EMCD non riproduce il riferimento")
        per_livello = {n: fuori[f"{n} {'risposta' if n == 'L1' else 'commento'}"]
                       for n in liv.nomi}
        for t in THETA:
            fuori[f"M-EMCD th={t}"] = emcd.consenso(liv, per_livello, t)

    np.savez_compressed(percorso, **fuori)
    return fuori


def esegui(sito: str, usa_emcd: bool = True, repliche: int = 2000) -> dict:
    liv = costruisci_livelli(sito)
    div = dividi(sito)
    profili = costruisci_profili(liv, per_verita=True, div=div)
    u, w, tag, nomi_tag, adozioni = exposure.eventi_per_tag(liv, div)
    rete, esposizioni = exposure.costruisci(liv, div)
    partizioni = candidate(liv, usa_emcd)

    # a pesi unitari la formula del bootstrap deve coincidere con la modularita'
    prova = next(iter(partizioni.values()))
    diretta = evaluate.modularita(rete, np.unique(prova, return_inverse=True)[1])
    vettoriale = bootstrap.q_da_pesi(
        bootstrap.prepara(prova, u, w, tag, len(nomi_tag)), np.ones(len(nomi_tag))
    )
    assert abs(diretta - vettoriale) < 1e-9, (
        f"formula del bootstrap incoerente: {diretta} vs {vettoriale}")

    print()
    print(evaluate.Valutazione.intestazione())
    print("-" * 74)
    risultati = {}
    for nome, cid in partizioni.items():
        v = evaluate.valuta(nome, cid, rete, profili)
        risultati[nome] = v
        print(v.riga())
    print(evaluate.controllo_singoletti(rete, profili, liv.n).riga())

    nodi, _ = profili.dominante()
    print(f"\ncriterio tematico su {len(nodi):,} nodi ({100*len(nodi)/liv.n:.1f}% "
          f"della rete); criterio di diffusione su {esposizioni:,} esposizioni "
          f"da {adozioni:,} adozioni di {len(nomi_tag)} tag bersaglio.")

    riferimento = max(risultati, key=lambda n: risultati[n].q_esp)
    print(f"\nBOOTSTRAP APPAIATO A GRAPPOLI SUI TAG ({repliche:,} repliche, "
          f"unita' = il tag intero)")
    print(bootstrap.Esito.intestazione())
    print("-" * 92)
    esiti = bootstrap.confronta(partizioni, u, w, tag, len(nomi_tag),
                                riferimento, repliche)
    for e in esiti:
        print(e.riga())

    equivalenti = [e.nome for e in esiti[1:] if not e.significativa]
    fragili = [e.nome for e in esiti[1:]
               if e.significativa and not e.regge_alla_correzione]
    print(f"\nMigliore per Q_esp: {riferimento}.")
    if equivalenti:
        print("Indistinguibili da essa: " + ", ".join(equivalenti) +
              ". Per parsimonia si sceglie la piu' semplice fra le equivalenti.")
    if fragili:
        print(f"Al limite, e non reggono la correzione di Bonferroni "
              f"(soglia {esiti[1].soglia_corretta:.4f} per "
              f"{esiti[1].confronti} confronti): " + ", ".join(fragili) +
              ".\nVanno riportate come differenze marginali, non come vittorie.")
    if not equivalenti and not fragili:
        print("Tutte le altre sono significativamente peggiori anche dopo "
              "la correzione: il verdetto e' netto.")
    return {"livelli": liv, "rete": rete, "partizioni": partizioni,
            "risultati": risultati, "esiti": esiti}


if __name__ == "__main__":
    argomenti = [a for a in sys.argv[1:] if not a.startswith("--")]
    opzioni = [a for a in sys.argv[1:] if a.startswith("--")]
    repliche = 2000
    for o in opzioni:
        if o.startswith("--repliche="):
            repliche = int(o.split("=")[1])
    esegui(argomenti[0] if argomenti else SITO_PRINCIPALE,
           usa_emcd="--senza-emcd" not in opzioni, repliche=repliche)
