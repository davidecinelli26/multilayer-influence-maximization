"""Base attiva mensile di un sito StackExchange: utenti distinti che hanno
scritto un post o un commento in ciascun mese. Serve a normalizzare il
punteggio di validazione per il declino dell'attivita' nella finestra di
verifica.
"""
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from asnm.adoption import leggi_post, mese
from asnm.config import percorso_xml


def base_attiva(sito: str) -> dict:
    """mese -> numero di utenti distinti attivi in quel mese."""
    t0 = time.time()
    post = leggi_post(sito)
    per_mese = defaultdict(set)
    for id_post, quando in post.quando.items():
        utente = post.proprietario.get(id_post)
        if utente:
            per_mese[quando].add(utente)
    for _, el in ET.iterparse(percorso_xml(sito, "Comments.xml"), events=("end",)):
        if el.tag == "row":
            a = el.attrib
            if "UserId" in a:
                per_mese[mese(a["CreationDate"])].add(a["UserId"])
            el.clear()
    conteggi = {m: len(u) for m, u in per_mese.items()}
    print(f"[attivita] {sito}: base attiva su {len(conteggi)} mesi "
          f"in {time.time()-t0:.0f}s")
    return conteggi


def epoca_dei_tag(div, tag_ordinati) -> np.ndarray:
    """Il mese mediano di adozione di ciascun tag nella finestra di verifica.

    E' l'epoca a cui attribuire il tag quando lo si normalizza per la base
    attiva: usare il mese della prima adozione sarebbe fuorviante, perche'
    un tag puo' comparire presto e diffondersi molto dopo.
    """
    per_tag = defaultdict(list)
    for (_, tag), m in div.test.items():
        if tag in div.bersagli:
            per_tag[tag].append(m)
    return np.array([np.median(per_tag[t]) for t in tag_ordinati])


def pesi(div, tag_ordinati, conteggi: dict, dimensioni: np.ndarray) -> dict:
    """Le tre ponderazioni messe a confronto.

    * `grezza`     -- ogni tag pesa quanto i suoi adottanti: e' il totale
                      che si e' usato finora;
    * `per tag`    -- ogni tag pesa uguale, cioe' si guarda la *quota* di
                      adottanti raggiunti invece del numero;
    * `base attiva`-- ogni tag e' riportato alla base attiva della sua epoca,
                      cosi' i tag del 2022 non valgono meno solo perche' il
                      sito si era svuotato.
    """
    epoche = epoca_dei_tag(div, tag_ordinati)
    base = np.array([conteggi.get(int(round(e)), np.nan) for e in epoche])
    if np.isnan(base).any():
        mediana = np.nanmedian(base)
        base = np.where(np.isnan(base), mediana, base)
    return {
        "grezza": np.ones(len(tag_ordinati)),
        "per tag": 1.0 / np.maximum(dimensioni, 1),
        "base attiva": float(np.mean(base)) / base,
    }
