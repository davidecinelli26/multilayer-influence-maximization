"""Lettura dei post StackExchange e costruzione degli eventi di adozione.

Un'adozione e' la coppia (utente, tag) con il mese in cui quell'utente ha
usato quel tag per la prima volta; il tag di una risposta e' quello della
domanda. Gli XML si leggono in streaming.
"""
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from asnm.config import percorso_xml
from asnm.dumps import estrai


def mese(data_iso: str) -> int:
    """'2019-07-14T...' -> indice di mese assoluto. Ordinabile e sottraibile."""
    return int(data_iso[:4]) * 12 + int(data_iso[5:7])


def etichetta_mese(m: int) -> str:
    """Inversa di mese(). Dicembre e' il resto 0, quindi appartiene
    all'anno precedente: senza questa correzione dic-2022 si stampa 2023-12."""
    anno, mm = divmod(m, 12)
    if mm == 0:
        anno, mm = anno - 1, 12
    return f"{anno}-{mm:02d}"


def _righe(percorso):
    for _, el in ET.iterparse(percorso, events=("end",)):
        if el.tag == "row":
            yield el.attrib
            el.clear()


@dataclass
class Post:
    """Le sole colonne di Posts.xml che servono al progetto."""
    proprietario: dict   # id_post -> id_utente  (assente per i post anonimi)
    tipo: dict           # id_post -> '1' domanda, '2' risposta
    genitore: dict       # id_risposta -> id_domanda
    tag: dict            # id_domanda -> lista di tag
    quando: dict         # id_post -> indice di mese


def leggi_post(sito: str) -> Post:
    estrai(sito)
    t0 = time.time()
    p = Post({}, {}, {}, {}, {})
    for a in _righe(percorso_xml(sito, "Posts.xml")):
        i = a["Id"]
        p.tipo[i] = a["PostTypeId"]
        p.quando[i] = mese(a["CreationDate"])
        if "OwnerUserId" in a:
            p.proprietario[i] = a["OwnerUserId"]
        if "ParentId" in a:
            p.genitore[i] = a["ParentId"]
        if "Tags" in a:
            grezzi = a["Tags"].replace("|", "<").replace(">", "<").split("<")
            p.tag[i] = [t for t in grezzi if t]
    print(f"[adoption] {sito}: {len(p.tipo):,} post letti in {time.time()-t0:.0f}s")
    return p


def tag_del_post(p: Post, id_post: str):
    """I tag associati a un post: i suoi se e' una domanda, quelli della
    domanda genitore se e' una risposta."""
    if p.tipo.get(id_post) == "1":
        return p.tag.get(id_post)
    return p.tag.get(p.genitore.get(id_post, ""))


def adozioni(sito: str, post: Post | None = None) -> dict:
    """(utente, tag) -> mese della prima volta."""
    p = post or leggi_post(sito)
    t0 = time.time()
    primo: dict[tuple[str, str], int] = {}
    for id_post, tipo in p.tipo.items():
        utente = p.proprietario.get(id_post)
        if not utente:
            continue
        tg = tag_del_post(p, id_post)
        if not tg:
            continue
        m = p.quando[id_post]
        for t in tg:
            chiave = (utente, t)
            if chiave not in primo or m < primo[chiave]:
                primo[chiave] = m
    print(f"[adoption] {sito}: {len(primo):,} eventi di adozione "
          f"in {time.time()-t0:.0f}s")
    return primo


if __name__ == "__main__":
    import sys
    from collections import Counter

    sito = sys.argv[1] if len(sys.argv) > 1 else "security"
    p = leggi_post(sito)
    a = adozioni(sito, p)
    per_anno = Counter(m // 12 for m in p.quando.values())
    print(f"\nutenti distinti con adozioni : {len({u for u, _ in a}):,}")
    print(f"tag distinti                 : {len({t for _, t in a}):,}")
    print("post per anno:", " ".join(f"{y}:{n}" for y, n in sorted(per_anno.items())))
