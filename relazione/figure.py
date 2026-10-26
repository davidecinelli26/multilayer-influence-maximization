"""Le figure della relazione, tutte da uno script.

    python relazione/figure.py

Ogni numero ha accanto il file di `risultati/higgs/` da cui viene: le figure
sono una vista sulle uscite del codice, non una fonte a se'. Le due serie
orarie (adozioni, tasso di trasmissione) vengono da `higgs_serie_orarie.npz`,
prodotto una volta da `stazionarieta.tassi_orari`.

Convenzioni grafiche, uguali per tutte: un colore di enfasi (blu) per il
substrato di lavoro o la serie che conta, grigio per il contesto, arancio
solo dove servono due serie che sono entrambe il soggetto. Segni sottili,
etichette dirette, griglia recessiva, nessun doppio asse.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

QUI = Path(__file__).resolve().parent
FUORI = QUI / "figure"
CACHE = QUI.parent / "data" / "cache"
FUORI.mkdir(exist_ok=True)

BLU, ARANCIO, ACQUA, GIALLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIGIO, GRIGIO_CHIARO, INCHIOSTRO, MUTO = "#898781", "#c3c2b7", "#0b0b0b", "#52514e"
GRIGLIA, ROSSO = "#e1e0d9", "#e34948"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.edgecolor": GRIGIO_CHIARO, "axes.labelcolor": MUTO,
    "xtick.color": MUTO, "ytick.color": MUTO,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIGLIA, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def salva(fig, nome):
    for est in ("pdf", "png"):
        fig.savefig(FUORI / f"{nome}.{est}")
    plt.close(fig)
    print(f"  {nome}")


# ---------------------------------------------------------------- 1. schema
def fig_architettura():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, testo, colore=BLU, riemp="#ffffff", peso="normal", fs=9):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                           fc=riemp, ec=colore, lw=1.4)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, testo, ha="center", va="center",
                fontsize=fs, color=INCHIOSTRO, weight=peso, linespacing=1.25)

    def freccia(x0, y0, x1, y1, colore=GRIGIO, stile="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=stile,
                                     mutation_scale=11, color=colore, lw=1.1))

    def scalino(x0, y0, x1, y1, ym, colore=GRIGIO):
        # a gradini: verticale, orizzontale, verticale; la punta sull'ultimo tratto
        ax.plot([x0, x0, x1], [y0, ym, ym], color=colore, lw=1.1, solid_capstyle="round")
        freccia(x1, ym, x1, y1, colore)

    # riga dei dati
    box(2, 44, 20, 9, "StackExchange\nrisposte · commenti\n2 livelli", GRIGIO, "#f4f4f1")
    box(78, 44, 20, 9, "Higgs Twitter\nfollower · RT · MT · RE\n4 livelli", BLU, "#eaf1fb", "bold")
    # il cancello
    box(33, 44, 34, 9, "IL CANCELLO\neccesso strutturale stratificato\ncontro il nullo a gradi preservati",
        INCHIOSTRO, "#ffffff", "bold")
    freccia(22, 48.5, 33, 48.5); freccia(78, 48.5, 67, 48.5)
    ax.text(27.5, 41.6, "−0%  →  controllo negativo", ha="center", fontsize=8.5, color=MUTO)
    ax.text(72.5, 41.6, "+73%  →  substrato di lavoro", ha="center", fontsize=8.5, color=BLU, weight="bold")

    # le tre fasi
    y = 27
    box(4, y, 26, 10, "FASE 1 · comunità\nLeiden, accoppiamento tarato\nQ sulle trasmissioni realizzate", BLU)
    box(37, y, 26, 10, "FASE 2 · probabilità\ncredit distribution + GraphSAGE\nregime · soglia di percolazione", BLU)
    box(70, y, 26, 10, "FASE 3 · seed\nTIM (RR-sketch) + vincolo DTIM\nsuperficie k × α", BLU)
    freccia(50, 44, 50, 37.6, INCHIOSTRO)
    freccia(30, 32, 37, 32); freccia(63, 32, 70, 32)

    # validazione
    box(25, 14.5, 50, 7.5, "VALIDAZIONE su adozioni vere · bootstrap a grappoli\n"
        "il grado batte TIM+  →  TIM+ non sa CHI adotterà", ROSSO, "#fbecea", "bold")
    scalino(83, 27, 60, 22.3, 24.6); scalino(17, 27, 40, 22.3, 24.6); freccia(50, 27, 50, 22.3)

    # il giro
    box(4, 2, 42, 8.5, "CHI ADOTTERÀ · scala a sette gradini\ngrado → esposizione → comunità → spettro\n→ propagati → SIGN (→ R-SIGN)", ACQUA, "#e8f6f0", "bold")
    box(54, 2, 42, 8.5, "IM PESATA dalla previsione\nradici RR in proporzione a p̂ (out-of-fold)\nbatte il grado a k=10 e k=20", ACQUA, "#e8f6f0", "bold")
    freccia(28, 14.5, 28, 10.8, ROSSO); freccia(46, 6.2, 54, 6.2, ACQUA)
    ax.text(30.5, 12.6, "prevedere prima di ottimizzare", ha="left", va="center", fontsize=8.5,
            color=ROSSO, style="italic")
    salva(fig, "fig_architettura")


# ---------------------------------------------------- 2. adozioni per ora
def fig_adozioni_ora():
    d = np.load(CACHE / "higgs_serie_orarie.npz")
    h = d["adozioni"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.fill_between(np.arange(168), h, color=BLU, alpha=0.18, lw=0)
    ax.plot(np.arange(168), h, color=BLU, lw=1.8)
    ax.axvline(72, color=INCHIOSTRO, lw=1.2, ls="--")
    ax.text(71, h.max() * 0.93, "taglio · ora 72", ha="right", fontsize=8.5, color=INCHIOSTRO)
    ax.axvspan(0, 72, color=GRIGLIA, alpha=0.45, lw=0)
    ax.text(36, h.max() * 0.93, "addestramento\n35.610 già attivi", ha="center", fontsize=8.5, color=MUTO)
    ax.text(122, h.max() * 0.93, "verifica\n269.081 da prevedere", ha="center", fontsize=8.5, color=MUTO)
    ax.annotate("annuncio del CERN\n4 lug 2012, ~07 UTC", xy=(78, h[78]), xytext=(100, h[78] * 0.75),
                fontsize=8.5, color=MUTO, arrowprops=dict(arrowstyle="-", color=GRIGIO, lw=0.8))
    ax.set_xlabel("ore dal 1° luglio 2012, 00:00 UTC")
    ax.set_ylabel("nuovi adottanti per ora")
    ax.set_xlim(0, 168); ax.set_xticks(range(0, 169, 24))
    ax.set_title("Higgs Twitter: il taglio cade prima dell'esplosione", loc="left")
    salva(fig, "fig_adozioni_ora")


# ------------------------------------------------- 3. eccesso strutturale
def fig_eccesso():
    # istruttoria/cascata4_stratificata.py e istruttoria/higgs_eccesso.py
    etichette = ["StackExchange\naggregato", "StackExchange\nstratificato (mese, tag)",
                 "Higgs\naggregato", "Higgs\nstratificato (ora)"]
    valori = [10, 0, 42, 73]
    colori = [GRIGIO, GRIGIO, BLU, BLU]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    barre = ax.bar(etichette, valori, color=colori, width=0.62)
    ax.axhline(40, color=INCHIOSTRO, lw=1.1, ls="--")
    ax.text(3.45, 41.5, "soglia +40%, fissata prima", ha="right", fontsize=8.5, color=INCHIOSTRO)
    for b, v in zip(barre, valori):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:+d}%" if v else "0%", ha="center", fontsize=9.5,
                color=INCHIOSTRO, weight="bold")
    ax.set_ylabel("eccesso strutturale (lift reale / lift nullo − 1)")
    ax.set_ylim(0, 84); ax.set_title("Quanto la topologia spiega, al netto di grado e attività", loc="left")
    ax.grid(axis="x", visible=False)
    salva(fig, "fig_eccesso")


# ------------------------------------------------------- 4. Fase 1: Q_tras
def fig_fase1():
    # risultati/higgs/fase1.txt: Q_tras e IC della differenza vs fusione calibrata
    rif = 0.6211
    righe = [  # nome, Q, IC basso, IC alto (assoluti = rif + IC differenza)
        ("fusione calibrata (λ=0,003)", 0.6211, None, None),
        ("F Louvain", 0.6201, rif - 0.0025, rif + 0.0005),
        ("F follower, Leiden", 0.6196, rif - 0.0018, rif - 0.0011),
        ("fusione convenzionale (λ=1)", 0.4694, rif - 0.1638, rif - 0.1396),
    ]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    y = np.arange(len(righe))[::-1]
    for yi, (nome, q, lo, hi) in zip(y, righe):
        col = BLU if "calibrata" in nome else GRIGIO
        ax.barh(yi, q, color=col, height=0.55)
        if lo is not None:
            ax.plot([lo, hi], [yi, yi], color=INCHIOSTRO, lw=1.2)
        ax.text(q + 0.008, yi, f"{q:.4f}", va="center", fontsize=9, color=INCHIOSTRO)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in righe])
    ax.set_xlim(0, 0.72); ax.set_xlabel("Q sulle trasmissioni realizzate dopo il taglio (fuori campione)")
    ax.set_title("Fase 1: quale partizione predice dove passa la diffusione", loc="left")
    ax.grid(axis="y", visible=False)
    ax.text(0.005, -0.85, "singoletti (controllo): Q = −0,0001 · AMI 0,0000 · NMI 0,4729",
            fontsize=8.5, color=MUTO)
    ax.set_ylim(-1.3, len(righe) - 0.4)
    salva(fig, "fig_fase1_q")


def fig_nmi_ami():
    # controllo a singoletti: fase1.txt (Higgs) e risultati StackExchange
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(2); w = 0.3
    nmi = [0.7689, 0.4729]; ami = [0.0000, 0.0000]
    ax.bar(x - w / 2, nmi, w, color=GRIGIO, label="NMI (grezza)")
    ax.bar(x + w / 2, ami, w, color=BLU, label="AMI (corretta per il caso)")
    for i in range(2):
        ax.text(x[i] - w / 2, nmi[i] + 0.02, f"{nmi[i]:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + w / 2, 0.02, "0,00", ha="center", fontsize=9, color=BLU, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["StackExchange\n46.312 singoletti", "Higgs\n456.627 singoletti"])
    ax.set_ylim(0, 0.9); ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("La trappola: una partizione di soli singoletti", loc="left")
    ax.grid(axis="x", visible=False)
    salva(fig, "fig_nmi_ami")


def fig_accoppiamento():
    # risultati/higgs/accoppiamento.txt
    lam = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    q = [0.4600, 0.4609, 0.4587, 0.4379, 0.3685, 0.2820, 0.1870]
    com = [226, 264, 311, 425, 587, 704, 919]
    x = [1e-4] + lam[1:]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.plot(x, q, color=BLU, lw=1.8, marker="o", ms=5)
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels(["0", "0,003", "0,01", "0,03", "0,1", "0,3", "1"])
    for xi, qi, c in zip(x, q, com):
        ax.text(xi, qi + 0.012, f"{c}", ha="center", fontsize=8, color=MUTO)
    ax.annotate("λ scelto\n(in campione)", xy=(0.003, 0.4609), xytext=(0.012, 0.50),
                fontsize=8.5, color=BLU, arrowprops=dict(arrowstyle="-", color=BLU, lw=0.8))
    ax.text(0.85, 0.235, "normalizzazione\nconvenzionale", ha="right", fontsize=8.5, color=MUTO)
    ax.set_xlabel("λ, peso degli strati di interazione (i numeri sono le comunità)")
    ax.set_ylabel("Q sulle trasmissioni prima del taglio")
    ax.set_ylim(0.15, 0.53)
    ax.set_title("L'accoppiamento tarato dal dato, senza toccare la verifica", loc="left")
    salva(fig, "fig_accoppiamento")


# ------------------------------------------------- 5. Fase 2: regime, soglia
def fig_regime():
    # risultati/higgs/regime.txt (sulla p della GNN) + stazionarieta.txt
    scale = [0.55, 1.0, 1.5, 2.0, 3.0]
    rr = [1.3, 16.3, 83.0, 145.0, 336.5]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.axhspan(3, 30, color="#eaf1fb", lw=0)
    ax.text(2.1, 12, "finestra utile: il greedy ha\nalternative vere fra cui scegliere", fontsize=8.5, color=BLU)
    ax.axhspan(200, 1000, color="#fbecea", lw=0)
    ax.text(0.57, 400, "sopracritico: migliaia di nodi equivalenti,\nogni confronto misura rumore", fontsize=8.5, color=ROSSO)
    ax.plot(scale, rr, color=INCHIOSTRO, lw=1.6, marker="o", ms=5)
    ax.axvline(4.55, color=ARANCIO, lw=1.3, ls="--")
    ax.text(4.45, 1.25, "×4,55: trasmissività\nmisurata dopo l'annuncio", ha="right", va="bottom", fontsize=8.5, color=ARANCIO)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 1.5, 2, 3, 4.55]); ax.set_xticklabels(["0,5", "1", "1,5", "2", "3", "4,55"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(0.5, 6); ax.set_ylim(1, 1000)
    ax.set_xlabel("fattore di scala su p (1 = stima non modificata)")
    ax.set_ylabel("dimensione media degli insiemi RR")
    ax.set_title("La soglia di percolazione sta fra ×1,5 e ×3 — e l'annuncio la attraversa", loc="left")
    salva(fig, "fig_regime")


def fig_stazionarieta():
    d = np.load(CACHE / "higgs_serie_orarie.npz")
    tasso = d["trasmettono"] / np.maximum(d["a_rischio"], 1)
    ore = np.arange(168)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(ore[1:], tasso[1:], color=BLU, lw=1.6)
    ax.axvline(72, color=INCHIOSTRO, lw=1.1, ls="--")
    ax.hlines(0.003298, 1, 72, color=GRIGIO, lw=1.2)
    ax.hlines(0.014996, 72, 167, color=ARANCIO, lw=1.2)
    ax.text(36, 0.0075, "prima: 0,0033 per arco-ora", ha="center", fontsize=8.5, color=MUTO)
    ax.text(120, 0.034, "dopo: 0,0150 per arco-ora  (×4,55)", ha="center", fontsize=8.5, color=ARANCIO, weight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0, 168); ax.set_xticks(range(0, 169, 24)); ax.set_ylim(2e-4, 0.08)
    ax.set_xlabel("ore dal 1° luglio 2012"); ax.set_ylabel("trasmissioni / archi a rischio, per ora (log)")
    ax.set_title("La probabilità di trasmissione non è stazionaria", loc="left")
    salva(fig, "fig_stazionarieta")


# ------------------------------------------------- 6. Fase 3: superficie
def fig_superficie():
    # risultati/higgs/fase3.txt (partizione fusione calibrata)
    k = [5, 10, 20, 50]; alpha = [1.00, 0.95, 0.90, 0.80, 0.70, 0.50]
    costo = np.array([
        [0.00, 0.00, 0.00, 0.06, -0.23, -0.23],
        [0.00, 0.00, -0.15, 0.54, -0.21, -0.57],
        [0.00, -0.16, 0.01, 0.09, -0.63, -3.96],
        [0.00, 0.03, -0.05, 0.01, -0.27, -1.80]])
    com = np.array([[2, 2, 2, 3, 5, 5], [4, 4, 4, 7, 8, 10], [7, 8, 8, 9, 12, 20], [10, 12, 12, 14, 18, 23]])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    im = ax.imshow(costo, cmap="RdBu", vmin=-4, vmax=4, aspect="auto")
    for i in range(4):
        for j in range(6):
            c = "white" if abs(costo[i, j]) > 2.2 else INCHIOSTRO
            ax.text(j, i - 0.17, f"{costo[i, j]:+.2f}%", ha="center", va="center", fontsize=8.5, color=c)
            ax.text(j, i + 0.22, f"{com[i, j]} com.", ha="center", va="center", fontsize=7.5, color=c if c == "white" else MUTO)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"α={a:.2f}" for a in alpha])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"k={x}" for x in k])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("costo di diffusione vs α=1", fontsize=8.5)
    ax.set_title("Superficie k × α: quante comunità, a quale costo (rumore MC ≈ 0,2%)", loc="left")
    salva(fig, "fig_superficie")


# ------------------------------------------------- 7. validazione
def fig_validazione():
    # risultati/higgs/validazione-k{10,20,50}.txt — differenza relativa vs TIM+ con IC
    strat = ["grado uscente", "TIM+ con p costante", "diversità α=0,80"]
    dati = {  # k -> [(rel, lo, hi) per strategia]; IC relativi = IC assoluti / TIM+
        10: [(-1.45, -2.6 / 121.6, -0.9 / 121.6), (-1.91, -4.8 / 121.6, 0.8 / 121.6), (-0.39, -1.2 / 121.6, 0.2 / 121.6)],
        20: [(7.66, 6.1 / 129.1, 15.0 / 129.1), (6.67, 4.5 / 129.1, 13.6 / 129.1), (-0.02, -0.6 / 129.1, 0.5 / 129.1)],
        50: [(3.34, 1.5 / 178.3, 11.4 / 178.3), (-3.94, -13.5 / 178.3, -0.1 / 178.3), (-2.65, -6.4 / 178.3, -3.3 / 178.3)],
    }
    colori = [ARANCIO, GRIGIO, BLU]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x = np.arange(3); w = 0.26
    for j, (s, c) in enumerate(zip(strat, colori)):
        vals = [dati[k][j][0] for k in (10, 20, 50)]
        lo = [dati[k][j][1] * 100 for k in (10, 20, 50)]
        hi = [dati[k][j][2] * 100 for k in (10, 20, 50)]
        ax.bar(x + (j - 1) * w, vals, w, color=c, label=s)
        for xi, v, l, h in zip(x + (j - 1) * w, vals, lo, hi):
            ax.plot([xi, xi], [l, h], color=INCHIOSTRO, lw=1)
    ax.axhline(0, color=INCHIOSTRO, lw=1)
    ax.text(2.42, 0.4, "TIM+ (α=1)", fontsize=8.5, color=MUTO)
    ax.set_xticks(x); ax.set_xticklabels(["k = 10", "k = 20", "k = 50"])
    ax.set_ylabel("differenza relativa vs TIM+ (%)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax.set_title("Validazione sulle adozioni vere: il grado batte TIM+ da k=20", loc="left")
    ax.grid(axis="x", visible=False); ax.set_ylim(-9, 12)
    salva(fig, "fig_validazione")


# ------------------------------------------------- 8. la scala a pioli
def fig_scala():
    # risultati/higgs/adozione.txt e adozione-relazionale.txt
    nomi = ["grado", "+ esposizione", "+ comunità", "+ spettro", "+ propagati", "SIGN", "R-SIGN"]
    auc = [0.6047, 0.6265, 0.6498, 0.6648, 0.6735, 0.6782, 0.6795]
    delta = [None, "+0,0217\np<0,001", "+0,0234\np<0,001", "+0,0150\np<0,001",
             "+0,0088\np<0,001", "+0,0047\np=0,006", "+0,0013\np=0,218"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(nomi))
    colori = [BLU] * 6 + [GRIGIO]
    ax.plot(x[:6], auc[:6], color=BLU, lw=1.6, zorder=2)
    ax.plot(x[5:], auc[5:], color=GRIGIO, lw=1.6, ls="--", zorder=2)
    ax.scatter(x, auc, s=46, color=colori, zorder=3, edgecolor="white", lw=1)
    for xi, a in zip(x, auc):
        ax.text(xi, a + 0.004, f"{a:.4f}", ha="center", fontsize=8.5, color=INCHIOSTRO, weight="bold")
    ax.axhspan(0.6782 - 0.0023, 0.6782 + 0.0023, xmin=0.72, xmax=1, color=GRIGLIA, lw=0)
    ax.text(6.55, 0.6782 - 0.0042, "rumore\nfra semi", fontsize=7.5, color=MUTO, ha="center")
    # il salto e il suo p stanno sotto il nome del gradino, non fra i punti
    etichette = [n if d is None else n + "\n" + d for n, d in zip(nomi, delta)]
    ax.set_xticks(x); ax.set_xticklabels(etichette, fontsize=8)
    ax.set_ylabel("AUC sul test (63.154 nodi)"); ax.set_ylim(0.595, 0.69); ax.set_xlim(-0.5, 7.15)
    ax.set_title("Chi adotterà: ogni famiglia di attributi aggiunge — la GNN poco, la relazionale niente", loc="left")
    ax.grid(axis="x", visible=False)
    salva(fig, "fig_scala")


# ------------------------------------------------- 9. IM pesata
def fig_im_pesata():
    # risultati/higgs/im-pesata-k{10,20,50}.txt — differenza relativa vs TIM+ con IC
    strat = ["grado uscente", "TIM pesato da p̂", "TIM pesato + diversità 0,80"]
    dati = {
        10: [(-1.45, -2.6 / 121.6, -0.9 / 121.6), (10.23, 7.4 / 121.6, 18.2 / 121.6), (8.45, 5.7 / 121.6, 15.8 / 121.6)],
        20: [(7.66, 6.1 / 129.1, 15.0 / 129.1), (15.26, 12.3 / 129.1, 27.9 / 129.1), (9.33, 4.9 / 129.1, 20.6 / 129.1)],
        50: [(3.34, 1.5 / 178.3, 11.4 / 178.3), (2.30, -6.0 / 178.3, 14.9 / 178.3), (2.46, -5.3 / 178.3, 14.7 / 178.3)],
    }
    colori = [ARANCIO, BLU, ACQUA]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    x = np.arange(3); w = 0.26
    for j, (s, c) in enumerate(zip(strat, colori)):
        vals = [dati[k][j][0] for k in (10, 20, 50)]
        lo = [dati[k][j][1] * 100 for k in (10, 20, 50)]
        hi = [dati[k][j][2] * 100 for k in (10, 20, 50)]
        ax.bar(x + (j - 1) * w, vals, w, color=c, label=s)
        for xi, v, l, h in zip(x + (j - 1) * w, vals, lo, hi):
            ax.plot([xi, xi], [l, h], color=INCHIOSTRO, lw=1)
    ax.axhline(0, color=INCHIOSTRO, lw=1)
    ax.text(2.42, 0.6, "TIM+ (α=1)", fontsize=8.5, color=MUTO)
    ax.set_xticks(x); ax.set_xticklabels(["k = 10", "k = 20", "k = 50"])
    ax.set_ylabel("differenza relativa vs TIM+ (%)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    ax.set_title("IM pesata dalla previsione: batte il grado a k=10 e k=20", loc="left")
    ax.grid(axis="x", visible=False); ax.set_ylim(-4, 24)
    salva(fig, "fig_im_pesata")


if __name__ == "__main__":
    print("figure in", FUORI)
    for f in (fig_architettura, fig_adozioni_ora, fig_eccesso, fig_fase1, fig_nmi_ami,
              fig_accoppiamento, fig_regime, fig_stazionarieta, fig_superficie,
              fig_validazione, fig_scala, fig_im_pesata):
        f()
