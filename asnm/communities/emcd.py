"""Wrapper sul JAR di M-EMCD (consenso di ensemble multistrato), la quarta
candidata della Fase 1 su StackExchange.

Il JAR non e' redistribuibile e va scaricato a parte in `vendor/`:
http://people.dimes.unical.it/andreatagarelli/emcd/
(Tagarelli, Amelio, Gullo, Data Mining and Knowledge Discovery 31(5), 2017).

Ingresso e uscita passano da file: adj0..adj{l-1} (liste di adiacenza per
livello, indici a base 1), class0..class{l-1} (partizione per livello), e

    java -jar generateGreedy.jar #livelli #nodi theta <cartella>

La cartella di output va creata prima, e theta va passato come stringa.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from asnm.config import VENDOR
from asnm.layers import Livelli

CARTELLA = VENDOR / "EMCD"
JAR = CARTELLA / "generateGreedy.jar"

_CANDIDATI_JAVA = ("java",)


def trova_java() -> str:
    """Un eseguibile java utilizzabile. Anche quello incluso in Gephi va bene."""
    if percorso := os.environ.get("ASNM_JAVA"):
        return percorso
    for c in _CANDIDATI_JAVA:
        if shutil.which(c) or Path(c).exists():
            return c
    for radice in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
                   Path(r"C:\Program Files")):
        if radice.exists():
            for p in radice.glob("*/**/bin/java.exe"):
                return str(p)
    raise RuntimeError(
        "nessun java trovato. Installa un JRE 8+ oppure indica l'eseguibile "
        "nella variabile d'ambiente ASNM_JAVA."
    )


def _controlla_jar():
    if not JAR.exists():
        raise FileNotFoundError(
            f"{JAR} mancante.\nEMCD non e' redistribuibile e non sta nel "
            f"repository: scaricalo da people.dimes.unical.it/andreatagarelli/emcd/ "
            f"ed estrailo in {VENDOR}."
        )


def scrivi_ingresso(cartella: Path, adiacenze: list, partizioni: list):
    """Scrive adj*/class* nel formato del manuale. `adiacenze` e' una lista di
    liste di vicini per nodo, a base 0; il file li vuole a base 1."""
    cartella.mkdir(parents=True, exist_ok=True)
    for k, (adj, cid) in enumerate(zip(adiacenze, partizioni)):
        with open(cartella / f"adj{k}.txt", "w") as f:
            f.writelines(" ".join(str(int(v) + 1) for v in vicini) + "\n"
                         for vicini in adj)
        with open(cartella / f"class{k}.txt", "w") as f:
            f.writelines(f"{int(c)}\n" for c in cid)


def esegui(cartella: Path, n_livelli: int, n_nodi: int, theta: float,
           memoria: str = "8G", timeout: int = 3600) -> np.ndarray:
    """Lancia il JAR e restituisce il vettore di consenso."""
    _controlla_jar()
    testo_theta = str(theta)
    uscita = cartella / testo_theta
    uscita.mkdir(parents=True, exist_ok=True)   # il JAR non lo fa da solo

    t0 = time.time()
    esito = subprocess.run(
        [trova_java(), "-jar", f"-Xmx{memoria}", str(JAR),
         str(n_livelli), str(n_nodi), testo_theta, str(cartella)],
        capture_output=True, text=True, timeout=timeout,
    )
    consenso = uscita / "consensus.txt"
    if not consenso.exists():
        coda = (esito.stderr or esito.stdout or "")[-400:]
        raise RuntimeError(f"M-EMCD non ha prodotto consensus.txt.\n{coda}")

    etichette = np.array([int(r) for r in consenso.read_text().split()], dtype=np.int64)
    if len(etichette) != n_nodi:
        raise RuntimeError(
            f"consensus.txt ha {len(etichette)} righe invece di {n_nodi}"
        )
    modularita = float((uscita / "modularity.txt").read_text().strip())
    print(f"[emcd] theta={theta}: {len(set(etichette.tolist())):,} comunita', "
          f"modularita' multistrato {modularita:.6f}, {time.time()-t0:.1f}s")
    return etichette


def consenso(liv: Livelli, partizioni: dict, theta: float = 0.4,
             memoria: str = "8G") -> np.ndarray:
    """M-EMCD sui due livelli, date le partizioni per livello."""
    cartella = CARTELLA / f"rete_{liv.sito}"
    adiacenze, per_livello = [], []
    for nome in liv.nomi:
        a = liv.csr_simmetrica(nome)
        adiacenze.append([a.indices[a.indptr[i]:a.indptr[i + 1]] for i in range(liv.n)])
        per_livello.append(partizioni[nome])
    scrivi_ingresso(cartella, adiacenze, per_livello)
    return esegui(cartella, len(liv.nomi), liv.n, theta, memoria)


# ------------------------------------------------------------- autoverifica
def verifica_aucs() -> bool:
    """Il wrapper riproduce l'esempio di riferimento fornito col pacchetto?

    Il pacchetto contiene la rete AUCS (61 nodi, 5 livelli) gia' risolta a
    theta=0.4. Se il nostro consenso coincide riga per riga con il loro, la
    catena scrittura-esecuzione-lettura e' corretta, e lo si sa prima di
    lanciarla su decine di migliaia di nodi.
    """
    aucs = CARTELLA / "aucs"
    atteso = aucs / "0.4" / "consensus.txt"
    if not atteso.exists():
        print("[emcd] esempio AUCS non disponibile: verifica saltata")
        return False
    riferimento = np.array([int(r) for r in atteso.read_text().split()])

    prova = CARTELLA / "aucs_verifica"
    shutil.rmtree(prova, ignore_errors=True)
    prova.mkdir(parents=True)
    for k in range(5):
        shutil.copy(aucs / f"adj{k}.txt", prova / f"adj{k}.txt")
        shutil.copy(aucs / f"class{k}.txt", prova / f"class{k}.txt")

    ottenuto = esegui(prova, 5, 61, 0.4, memoria="2G")
    uguale = bool((ottenuto == riferimento).all())
    print(f"[emcd] verifica AUCS: {'identico al riferimento' if uguale else 'DIVERSO'}")
    shutil.rmtree(prova, ignore_errors=True)
    return uguale


if __name__ == "__main__":
    import sys

    from asnm.communities.single import louvain
    from asnm.config import SITO_PRINCIPALE
    from asnm.layers import costruisci as costruisci_livelli

    if not verifica_aucs():
        raise SystemExit("il wrapper non riproduce il riferimento: non proseguo")

    sito = sys.argv[1] if len(sys.argv) > 1 else SITO_PRINCIPALE
    liv = costruisci_livelli(sito)
    per_livello = {n: louvain(liv.grafo(n), liv.n) for n in liv.nomi}
    for theta in (float(t) for t in sys.argv[2:]) or [0.4]:
        consenso(liv, per_livello, theta)
