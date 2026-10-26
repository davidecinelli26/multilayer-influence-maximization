"""Percorsi e costanti del progetto. Nessun import dal resto del pacchetto."""
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
GREZZI = RADICE / "data" / "raw"      # gli archivi .7z scaricati
XML = RADICE / "data" / "xml"         # gli XML estratti, per sito
CACHE = RADICE / "data" / "cache"     # artefatti intermedi (.npz, .pkl)
VENDOR = RADICE / "vendor"            # EMCD e altro codice di terzi

# I due siti StackExchange: il principale e una replica su scala ridotta.
SITI = {
    "security": "security.stackexchange.7z",
    "computergraphics": "computergraphics.stackexchange.7z",
}
SITO_PRINCIPALE = "security"

for _d in (XML, CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def percorso_xml(sito: str, nome: str) -> Path:
    """data/xml/<sito>/<nome>.xml"""
    return XML / sito / nome
