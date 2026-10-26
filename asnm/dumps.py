"""Estrazione degli XML necessari (Posts, Comments) dagli archivi .7z di
StackExchange, una volta sola, su disco.
"""
import time

import py7zr

from asnm.config import GREZZI, SITI, XML, percorso_xml

NECESSARI = ("Posts.xml", "Comments.xml")


def estrai(sito: str, forza: bool = False) -> dict:
    """Garantisce che gli XML del sito siano su disco. Restituisce i percorsi."""
    if sito not in SITI:
        raise ValueError(f"sito sconosciuto: {sito!r}; disponibili {sorted(SITI)}")

    percorsi = {n: percorso_xml(sito, n) for n in NECESSARI}
    if not forza and all(p.exists() and p.stat().st_size > 0 for p in percorsi.values()):
        return percorsi

    archivio = GREZZI / SITI[sito]
    if not archivio.exists():
        raise FileNotFoundError(
            f"archivio mancante: {archivio}\n"
            f"Scaricalo dai dump pubblici StackExchange (Internet Archive) "
            f"e mettilo in {GREZZI}."
        )

    destinazione = XML / sito
    destinazione.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with py7zr.SevenZipFile(archivio, "r") as z:
        z.extract(path=destinazione, targets=list(NECESSARI))
    print(f"[dumps] {sito}: estratti {', '.join(NECESSARI)} in {time.time()-t0:.0f}s")
    return percorsi


if __name__ == "__main__":
    import sys

    for s in sys.argv[1:] or [*SITI]:
        for nome, p in estrai(s).items():
            print(f"  {nome:14s} {p.stat().st_size/1e6:8.1f} MB  {p}")
