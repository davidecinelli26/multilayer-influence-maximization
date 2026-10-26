"""Confine temporale e tag bersaglio su StackExchange.

E' l'unico modulo in cui sono definiti dove finisce l'addestramento e
comincia la verifica, e quali tag si vogliono prevedere. I tag bersaglio
sono esclusi dalla verita' di riferimento tematica e, con MASCHERA_SEVERA,
anche dalla stima di p e dagli attributi dei nodi: altrimenti lo stesso
segnale costruirebbe la rete, definirebbe la verita' e sarebbe l'oggetto
della previsione.
"""
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from asnm.adoption import adozioni, etichetta_mese

# --------------------------------------------------------------- politica
CUTOFF = 2020 * 12 + 1
"""Gennaio 2020. Struttura, probabilita' e selezione dei seed usano solo
eventi con mese < CUTOFF."""

FINE_TEST = 2023 * 12 + 1
"""Gennaio 2023: la finestra di verifica e' 2020-2022. Non arriva al 2024
perche' l'attivita' del sito principale crolla e la coda non ha volume."""

MIN_ADOZIONI_POST = {"security": 100, "computergraphics": 30}
"""Quante adozioni dopo il cutoff servono perche' un tag sia un bersaglio.
La soglia e' piu' bassa sul sito piccolo, che ha un ordine di grandezza
in meno di attivita'."""

MASCHERA_SEVERA = True
"""Se True i tag bersaglio sono esclusi anche dalla stima di p e dagli
attributi dei nodi, non solo dalla verita' di riferimento tematica."""


@dataclass
class Divisione:
    sito: str
    bersagli: frozenset
    train: dict = field(repr=False)   # (utente, tag) -> mese, mese < CUTOFF
    test: dict = field(repr=False)    # (utente, tag) -> mese, CUTOFF <= mese < FINE_TEST

    # -------------------------------------------------- predicati temporali
    @staticmethod
    def e_train(mese: int) -> bool:
        return mese < CUTOFF

    @staticmethod
    def e_test(mese: int) -> bool:
        return CUTOFF <= mese < FINE_TEST

    # ------------------------------------------------- politica sui tag
    def ammesso_per_stima(self, tag: str) -> bool:
        """Un tag puo' entrare nella stima di p e negli attributi dei nodi?"""
        return not (MASCHERA_SEVERA and tag in self.bersagli)

    def ammesso_come_verita(self, tag: str) -> bool:
        """Un tag puo' entrare nel profilo tematico usato come verita' di
        riferimento? Mai un bersaglio, in nessuna configurazione."""
        return tag not in self.bersagli

    # ------------------------------------------------------------- comodita'
    def adottanti(self, tag: str) -> set:
        """Gli utenti che hanno adottato il tag nella finestra di verifica."""
        return {u for (u, t) in self.test if t == tag}

    def riassunto(self) -> str:
        n_ut = len({u for u, _ in self.train} | {u for u, _ in self.test})
        return (
            f"sito {self.sito}\n"
            f"  confine temporale   : {etichetta_mese(CUTOFF)} "
            f"(verifica fino a {etichetta_mese(FINE_TEST - 1)})\n"
            f"  adozioni train      : {len(self.train):,}\n"
            f"  adozioni test       : {len(self.test):,}\n"
            f"  utenti coinvolti    : {n_ut:,}\n"
            f"  tag bersaglio       : {len(self.bersagli)} "
            f"(soglia {MIN_ADOZIONI_POST.get(self.sito, 100)} adozioni post-cutoff)\n"
            f"  maschera severa     : {'attiva' if MASCHERA_SEVERA else 'disattivata'}"
        )


@lru_cache(maxsize=4)
def dividi(sito: str) -> Divisione:
    """Costruisce la divisione per un sito. Il risultato e' memoizzato."""
    t0 = time.time()
    tutte = adozioni(sito)
    train = {k: m for k, m in tutte.items() if Divisione.e_train(m)}
    test = {k: m for k, m in tutte.items() if Divisione.e_test(m)}

    conta_post = Counter(t for (_, t) in test)
    soglia = MIN_ADOZIONI_POST.get(sito, 100)
    bersagli = frozenset(t for t, n in conta_post.items() if n >= soglia)

    print(f"[splits] {sito}: divisione pronta in {time.time()-t0:.0f}s")
    return Divisione(sito, bersagli, train, test)


if __name__ == "__main__":
    import sys

    from asnm.config import SITO_PRINCIPALE

    d = dividi(sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE)
    print()
    print(d.riassunto())
    piu_diffusi = Counter(t for (_, t) in d.test if t in d.bersagli).most_common(8)
    print("\n  bersagli piu' adottati nel test:")
    for t, n in piu_diffusi:
        print(f"    {t:32s} {n:>5,}")
