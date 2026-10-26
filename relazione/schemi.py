"""Gli schemi concettuali della relazione: disegni, non grafici da dati.

    python relazione/schemi.py

Capitolo 2: l'RR-set e la copertura greedy; il vincolo DTIM annotato.
Capitolo 5: il test dose-risposta con ricablaggio; la scala a pioli;
i dieci operatori di R-SIGN. Stesso stile di `figure.py`.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

QUI = Path(__file__).resolve().parent
FUORI = QUI / "figure"
BLU, ARANCIO, ACQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIGIO, GRIGIO_CHIARO, INCHIOSTRO, MUTO = "#898781", "#c3c2b7", "#0b0b0b", "#52514e"
GRIGLIA, ROSSO, BLU_CHIARO = "#e1e0d9", "#e34948", "#eaf1fb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9.5, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def tela(w, h, xmax=100, ymax=60):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def salva(fig, nome):
    for est in ("pdf", "png"):
        fig.savefig(FUORI / f"{nome}.{est}")
    plt.close(fig); print(f"  {nome}")


def nodo(ax, x, y, etichetta="", colore=GRIGIO, riemp="white", r=2.4, fs=8.5, peso="normal"):
    ax.add_patch(Circle((x, y), r, fc=riemp, ec=colore, lw=1.4, zorder=3))
    if etichetta:
        ax.text(x, y, etichetta, ha="center", va="center", fontsize=fs, color=INCHIOSTRO,
                weight=peso, zorder=4)


def arco(ax, a, b, colore=GRIGIO, stile="-", lw=1.2, r=2.4, freccia=True, curva=0.0):
    (x0, y0), (x1, y1) = a, b
    d = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    p0 = (x0 + ux * r, y0 + uy * r); p1 = (x1 - ux * r, y1 - uy * r)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if freccia else "-", mutation_scale=9,
                                 color=colore, lw=lw, ls=stile, zorder=2,
                                 connectionstyle=f"arc3,rad={curva}"))


def riquadro(ax, x, y, w, h, testo, colore=BLU, riemp="white", fs=8.5, peso="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1",
                                fc=riemp, ec=colore, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, testo, ha="center", va="center", fontsize=fs,
            color=INCHIOSTRO, weight=peso, linespacing=1.25, zorder=3)


# ============================================================ 1. RR-set
def schema_rr():
    fig, ax = tela(11, 4.6, 112, 46)
    ax.text(2, 44, "a) Un insieme RR: chi avrebbe potuto attivare v", fontsize=10, weight="bold", color=INCHIOSTRO)
    # grafo: v in basso a destra, visita all'indietro
    P = {"v": (30, 10), "a": (18, 22), "b": (34, 24), "c": (8, 32), "d": (24, 36),
         "e": (42, 36), "f": (46, 14)}
    vivi = [("a", "v", True), ("b", "v", True), ("c", "a", True), ("d", "a", False),
            ("d", "b", True), ("e", "b", False), ("f", "v", False)]
    for s, t, vivo in vivi:
        arco(ax, P[s], P[t], colore=BLU if vivo else GRIGIO_CHIARO,
             stile="-" if vivo else (0, (2, 2)), lw=1.6 if vivo else 1.1)
    raggiunti = {"v", "a", "b", "c", "d"}
    for n, (x, y) in P.items():
        nodo(ax, x, y, n, colore=BLU if n in raggiunti else GRIGIO_CHIARO,
             riemp=BLU_CHIARO if n in raggiunti else "white",
             peso="bold" if n == "v" else "normal")
    ax.text(36, 8, "radice v,\nestratta a caso", fontsize=8, color=MUTO, va="center")
    ax.text(2, 40, "ogni arco entrante è vivo con probabilità p(u,v):\n"
            "vivo (blu) si attraversa, morto (tratteggiato) no", fontsize=8, color=MUTO)
    ax.text(2, 4.5, "RR(v) = {v, a, b, c, d}", fontsize=9.5, color=BLU, weight="bold")
    ax.text(2, 1, "se un seed sta in RR(v), v viene attivato", fontsize=8.5, color=MUTO)

    # b) copertura greedy
    ax.text(68, 44, "b) Massima copertura", fontsize=10, weight="bold", color=INCHIOSTRO)
    nodi_ = ["a", "b", "c", "d", "e", "f"]
    insiemi = [{"a", "b", "c", "d"}, {"a", "c"}, {"b", "e"}, {"d", "f"}, {"a", "f"}, {"b", "d"}]
    x0, y0, cw, ch = 71, 8, 4.6, 4.4
    for j, n in enumerate(nodi_):
        ax.text(x0 + j * cw + cw / 2, y0 + 6 * ch + 1.2, n, ha="center", fontsize=8.5, color=INCHIOSTRO)
    for i, ins in enumerate(insiemi):
        yy = y0 + (5 - i) * ch
        ax.text(x0 - 1.5, yy + ch / 2, f"RR{i+1}", ha="right", va="center", fontsize=7.5, color=MUTO)
        for j, n in enumerate(nodi_):
            dentro = n in ins
            ax.add_patch(Rectangle((x0 + j * cw, yy), cw - 0.4, ch - 0.4,
                                   fc=BLU_CHIARO if dentro else "white", ec=GRIGLIA, lw=0.8))
            if dentro:
                ax.text(x0 + j * cw + cw / 2, yy + ch / 2, "•", ha="center", va="center",
                        fontsize=11, color=BLU)
    conteggi = [sum(n in ins for ins in insiemi) for n in nodi_]
    for j, c in enumerate(conteggi):
        ax.text(x0 + j * cw + cw / 2, y0 - 2.2, str(c), ha="center", fontsize=8.5,
                color=BLU if c == max(conteggi) else MUTO, weight="bold" if c == max(conteggi) else "normal")
    ax.text(x0 - 1.5, y0 - 2.2, "copre", ha="right", fontsize=7.5, color=MUTO)
    ax.add_patch(Rectangle((x0 - 0.3, y0 - 0.3), cw - 0.1, 6 * ch + 0.2, fc="none", ec=BLU, lw=1.6, zorder=5))
    ax.text(x0 + 6 * cw + 1.5, y0 + 3 * ch, "il greedy prende il nodo\nche copre più insiemi,\n"
            "toglie quelli coperti,\nripete k volte.\n\nGaranzia: ≥ (1−1/e)\ndell'ottimo.",
            fontsize=8, color=MUTO, va="center")
    salva(fig, "schema_rr")


# ============================================================ 2. DTIM
def schema_dtim():
    fig, ax = tela(10, 3.8, 100, 38)
    ax.text(50, 31, r"$s(v) \;=\; \alpha \cdot \dfrac{|R_v|}{\max_u |R_u|} \;+\; (1-\alpha)\cdot \mathbf{1}\,[\,c(v) \notin C\,]$",
            ha="center", va="center", fontsize=16, color=INCHIOSTRO)
    # annotazioni
    def nota(x, y, testo, xt, yt, colore=BLU):
        ax.annotate(testo, xy=(x, y), xytext=(xt, yt), fontsize=8.5, color=colore, ha="center",
                    arrowprops=dict(arrowstyle="-", color=colore, lw=0.9))
    nota(27, 27, "copertura: quanti insiemi RR\nancora scoperti contengono v", 22, 14)
    nota(47, 27, "normalizzata sul massimo — senza,\nil conteggio (centinaia) schiaccerebbe\nun indicatore che vale 0 o 1", 50, 12, MUTO)
    nota(74, 27, "1 se la comunità di v non è\nancora toccata da nessun seed\n(e ha almeno 200 nodi)", 78, 14, ACQUA)
    nota(14, 31, "α = 1: TIM+ esatto,\nstesso codice", 8, 20, INCHIOSTRO)
    # tasso di scambio
    ax.text(2, 5.5, "tasso di scambio (1−α)/α — quanta copertura in più deve avere un nodo ridondante per battere uno che apre una comunità:",
            fontsize=8.5, color=MUTO)
    for i, (a, r) in enumerate([(0.95, 0.053), (0.90, 0.111), (0.80, 0.25), (0.70, 0.43), (0.50, 1.0)]):
        x = 6 + i * 19
        ax.text(x, 1.5, f"α={a:.2f} → {r:.2f}" + ("  (≥1: irraggiungibile)" if r >= 1 else ""),
                fontsize=8.5, color=ROSSO if r >= 1 else INCHIOSTRO, weight="bold" if r >= 1 else "normal")
    salva(fig, "schema_dtim")


# ============================================================ 3. dose-risposta
def schema_dose():
    fig, ax = tela(10, 4.4, 100, 44)
    ax.text(2, 42, "a) rete reale", fontsize=10, weight="bold")
    ax.text(36, 42, "b) ricablata: stessi gradi, vicini a caso", fontsize=10, weight="bold")
    ax.text(70, 42, "c) curva dose-risposta", fontsize=10, weight="bold")

    def rete(ox, archi, attivi, centro):
        P = {"u": (ox + 14, 20), "1": (ox + 4, 30), "2": (ox + 14, 34), "3": (ox + 24, 30),
             "4": (ox + 4, 10), "5": (ox + 24, 10), "6": (ox + 28, 20)}
        for a, b in archi:
            arco(ax, P[a], P[b], colore=GRIGIO, freccia=False)
        for n, (x, y) in P.items():
            att = n in attivi
            nodo(ax, x, y, "", colore=BLU if att else GRIGIO, riemp=BLU if att else "white", r=2)
        nodo(ax, *P["u"], "u", colore=INCHIOSTRO, riemp="white", r=2.4, peso="bold")
        k = sum(1 for a, b in archi if (a == "u" and b in attivi) or (b == "u" and a in attivi))
        ax.text(ox + 14, 3.5, f"u ha k = {k} vicini già attivi", ha="center", fontsize=8.5, color=INCHIOSTRO)
        return P

    archi_reali = [("u", "1"), ("u", "2"), ("u", "3"), ("u", "4"), ("1", "2"), ("2", "3"), ("5", "6"), ("3", "6")]
    rete(0, archi_reali, {"1", "2", "3"}, "u")
    # ricablata: u mantiene grado 4, ma i vicini cambiano
    archi_ric = [("u", "1"), ("u", "4"), ("u", "5"), ("u", "6"), ("2", "3"), ("1", "2"), ("3", "6"), ("2", "5")]
    rete(34, archi_ric, {"1", "2", "3"}, "u")
    ax.text(50, 38.5, "gli attivi sono gli stessi, i mesi/ore pure:\ncambia solo chi è collegato a chi", ha="center", fontsize=8, color=MUTO)

    # c) mini-grafico
    x0, y0, w, h = 72, 8, 24, 26
    ax.add_patch(Rectangle((x0, y0), w, h, fc="white", ec=GRIGIO_CHIARO, lw=1))
    ks = np.array([0, 1, 2, 3, 4]); xs = x0 + 2 + ks * 5
    reale = np.array([2, 7, 11, 14, 17]); nullo = np.array([2, 4.5, 6.5, 8, 9.5])
    ax.plot(xs, y0 + reale, color=BLU, lw=1.8, marker="o", ms=3.5)
    ax.plot(xs, y0 + nullo, color=GRIGIO, lw=1.8, marker="o", ms=3.5)
    ax.text(x0 + 23, y0 + 18.5, "reale", color=BLU, fontsize=8.5, ha="right", weight="bold")
    ax.text(x0 + 23, y0 + 7.2, "ricablata", color=GRIGIO, fontsize=8.5, ha="right")
    for k, xx in zip(ks, xs):
        ax.text(xx, y0 - 2, str(k), ha="center", fontsize=8, color=MUTO)
    ax.text(x0 + w / 2, y0 - 5, "k, vicini già attivi", ha="center", fontsize=8.5, color=MUTO)
    ax.text(x0 - 1.5, y0 + h / 2, "P(adotta | k)", rotation=90, ha="right", va="center", fontsize=8.5, color=MUTO)
    ax.text(x0 + w / 2, 37.5, "eccesso = lift reale / lift nullo − 1\nstratificato per (mese, tag) o per ora",
            ha="center", fontsize=8.5, color=INCHIOSTRO)
    salva(fig, "schema_dose")


# ============================================================ 4. la scala
def schema_scala():
    fig, ax = tela(10, 5.0, 100, 51)
    gradini = [("grado", "seguaci, seguiti", "alberi"),
               ("esposizione", "quanti seguiti\ngià attivi, quando", "alberi"),
               ("comunità", "quanto è accesa,\nesposizione intra", "alberi"),
               ("spettro", "32 dimensioni\ndel follower", "alberi"),
               ("propagati", "A·x, A²·x\n(vicini, vicini dei vicini)", "alberi"),
               ("SIGN", "stessi attributi,\nMLP", "GNN"),
               ("R-SIGN", "direzione e tipo\ndi relazione separati", "GNN")]
    w, dy = 13.2, 5.2
    for i, (nome, cosa, modello) in enumerate(gradini):
        x = 1.5 + i * w; y = 4 + i * dy
        col = BLU if modello == "alberi" else ACQUA
        ax.add_patch(Rectangle((x, 4), w - 1, y - 4 + 5.5, fc=BLU_CHIARO if modello == "alberi" else "#e8f6f0", ec=col, lw=1.2))
        ax.text(x + (w - 1) / 2, y + 3.4, nome, ha="center", fontsize=9.5, weight="bold", color=INCHIOSTRO)
        ax.text(x + (w - 1) / 2, y + 0.9, cosa, ha="center", va="center", fontsize=7.4, color=MUTO)
        if i:
            ax.annotate("", xy=(x + 0.5, y + 6.3), xytext=(x - 1.5, y - dy + 6.3),
                        arrowprops=dict(arrowstyle="-|>", color=INCHIOSTRO, lw=1))
    ax.text(1.5, 49, "ogni gradino = il precedente + una famiglia di attributi, tutti calcolabili prima dell'ora 72",
            fontsize=9, color=INCHIOSTRO, weight="bold")
    ax.text(1.5, 46.3, "stesso modello per i primi cinque (gradient boosting, iperparametri fissi); gli ultimi due sono la GNN.",
            fontsize=8, color=MUTO)
    ax.text(1.5, 43.8, "ogni salto è giudicato con bootstrap appaiato a grappoli sulle comunità e Bonferroni; l'ultimo contro il rumore fra due semi.",
            fontsize=8, color=MUTO)
    ax.add_patch(Rectangle((1.5, 36.5), 3.2, 2.4, fc=BLU_CHIARO, ec=BLU)); ax.text(6, 37.7, "alberi", va="center", fontsize=8)
    ax.add_patch(Rectangle((14, 36.5), 3.2, 2.4, fc="#e8f6f0", ec=ACQUA)); ax.text(18.5, 37.7, "GNN", va="center", fontsize=8)
    salva(fig, "schema_scala")


# ============================================================ 5. R-SIGN
def schema_rsign():
    fig, ax = tela(10, 4.8, 100, 49)
    # a) il vicinato di v
    ax.text(2, 47, "a) che cosa vede v", fontsize=10, weight="bold")
    V = (18, 22)
    nodo(ax, *V, "v", colore=INCHIOSTRO, riemp="white", r=2.8, peso="bold")
    seguiti = [(6, 34), (18, 38), (30, 34)]
    seguaci = [(6, 10), (30, 10)]
    for s in seguiti:
        nodo(ax, *s, "", colore=BLU, riemp=BLU_CHIARO, r=2); arco(ax, s, V, colore=BLU, lw=1.5)
    for s in seguaci:
        nodo(ax, *s, "", colore=GRIGIO, riemp="white", r=2); arco(ax, V, s, colore=GRIGIO, lw=1.3)
    ax.text(18, 44.6, "A_in: i seguiti — i loro tweet mi raggiungono", ha="center", fontsize=8, color=BLU)
    ax.text(18, 4.5, "A_out: i seguaci — omofilia, non canale", ha="center", fontsize=8, color=GRIGIO)
    # secondi vicini con relazioni tipizzate
    for (x, y), col, tag in [((3.5, 39), ARANCIO, "RT"), ((25, 41), ACQUA, "MT"), ((36, 39), "#4a3aa7", "RE")]:
        nodo(ax, x, y, tag, colore=col, riemp="white", r=1.9, fs=6.5)
    arco(ax, (3.5, 39), seguiti[0], colore=ARANCIO, lw=1.2, r=2.0, freccia=False)
    arco(ax, (25, 41), seguiti[1], colore=ACQUA, lw=1.2, r=2.0, freccia=False)
    arco(ax, (36, 39), seguiti[2], colore="#4a3aa7", lw=1.2, r=2.0, freccia=False)
    ax.text(36, 24, "a due passi: con chi i miei\nseguiti hanno ritwittato (RT),\nmenzionato (MT), risposto (RE).\nUn nodo non attivo non ha\narchi di interazione propri.",
            fontsize=7.8, color=MUTO, va="center")

    # b) i blocchi
    ax.text(58, 47, "b) i dieci blocchi, un peso per relazione", fontsize=10, weight="bold")
    blocchi = ["x", "A_in x", "A_out x", "A_in A_in x", "A_in A_out x", "A_out A_in x",
               "A_out A_out x", "A_in A_RT x", "A_in A_MT x", "A_in A_RE x"]
    for i, b in enumerate(blocchi):
        y = 42 - i * 3.6
        ax.add_patch(Rectangle((58, y - 1.3), 14, 2.8, fc=BLU_CHIARO, ec=BLU, lw=0.9))
        ax.text(65, y + 0.1, b, ha="center", va="center", fontsize=7.6, color=INCHIOSTRO)
        ax.add_patch(Rectangle((75, y - 1.3), 6, 2.8, fc="white", ec=ACQUA, lw=0.9))
        ax.text(78, y + 0.1, f"W{i+1}", ha="center", va="center", fontsize=7.4, color=ACQUA)
        ax.annotate("", xy=(75, y + 0.1), xytext=(72, y + 0.1), arrowprops=dict(arrowstyle="-|>", color=GRIGIO, lw=0.8))
        ax.annotate("", xy=(85, 25), xytext=(81, y + 0.1), arrowprops=dict(arrowstyle="-", color=GRIGIO_CHIARO, lw=0.7))
    riquadro(ax, 85, 18, 12, 14, "concatena\n↓\nReLU\n↓\nMLP\n↓\np̂(v)", colore=ACQUA, riemp="#e8f6f0", fs=8)
    ax.text(58, 1.5, "propagazione precalcolata con scipy, una volta; il primo strato di un R-GCN\nsenza la non-linearità fra le aggregazioni. 420 colonne, addestra in 3 minuti.",
            fontsize=7.8, color=MUTO)
    salva(fig, "schema_rsign")


if __name__ == "__main__":
    print("schemi in", FUORI)
    for f in (schema_rr, schema_dtim, schema_dose, schema_scala, schema_rsign):
        f()
