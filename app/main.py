"""La schermata.

    streamlit run app/main.py

Una pagina sola. A sinistra i comandi, al centro la mappa delle comunita' e
la frontiera, a destra i seed con la ragione per cui sono stati scelti, in
basso i tempi reali di ogni stadio.

Muovere lo slider **e'** il risultato: la frontiera di Pareto smette di
essere una tabella e diventa un gesto.

Una scelta di disegno che vale la pena spiegare: la mappa non mostra i
46.312 nodi, che produrrebbero un gomitolo illeggibile, ma il **grafo delle
comunita' contratto**. Cosi' si vede esattamente la grandezza che lo slider
controlla -- quante e quali comunita' i seed toccano -- invece di una nuvola
in cui non si distingue niente.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app.dossier import GRIGLIA, costruisci
from asnm.config import SITO_PRINCIPALE

K_PER_SITO = {"security": 50, "computergraphics": 20}

matplotlib.use("Agg")

PETROLIO = "#0F6B63"
BORDEAUX = "#8A1F3D"
AMBRA = "#8A5A12"
GRIGIO = "#8A9390"
CHIARO = "#D8DEDC"

st.set_page_config(page_title="Semina", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource(show_spinner="Carico il dossier precalcolato...")
def carica(sito, k):
    return costruisci(sito, k)


def figura_mappa(d, punto, mostra_adottanti: bool):
    """Solo le comunita' significative: le altre sono gruppetti da 2-4 nodi
    e formerebbero un anello di polvere senza aggiungere informazione."""
    vis = d.visibili
    toccate = {int(d.cid[s]) for s in punto.semi} & set(vis.tolist())
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    x, y = d.posizioni[:, 0], d.posizioni[:, 1]
    for a, b in d.archi:
        ax.plot([x[a], x[b]], [y[a], y[b]], color=CHIARO, lw=0.6, zorder=1)

    dim = d.dimensioni[vis]
    grandezza = 80 + 900 * np.sqrt(dim / dim.max())
    if mostra_adottanti:
        ad = d.adottanti_per_comunita[vis]
        ax.scatter(x[vis], y[vis], s=grandezza, c=ad, cmap="YlOrBr",
                   vmin=0, vmax=max(ad.max(), 1), edgecolors="white",
                   linewidths=1.0, zorder=2)
        etichetta = "colore = adottanti reali 2020-2022"
    else:
        ax.scatter(x[vis], y[vis], s=grandezza, color=CHIARO,
                   edgecolors="white", linewidths=1.0, zorder=2)
        etichetta = "grigio = non toccata dai seed"

    indici = np.array(sorted(toccate), dtype=int)
    if len(indici):
        mappa = {c: i for i, c in enumerate(vis.tolist())}
        s_tocc = np.array([grandezza[mappa[c]] for c in indici])
        ax.scatter(x[indici], y[indici], s=s_tocc, facecolors="none",
                   edgecolors=PETROLIO, linewidths=2.6, zorder=3)
    # il testo va scelto sul fondo, altrimenti sui cerchi scuri sparisce
    for c in vis.tolist():
        scuro = (mostra_adottanti and d.adottanti_per_comunita[c]
                 > 0.55 * max(d.adottanti_per_comunita[vis].max(), 1))
        ax.annotate(f"{d.dimensioni[c]:,}", (x[c], y[c]), ha="center",
                    va="center", fontsize=6.5, zorder=4,
                    color="white" if scuro else "#3D4548")

    ax.set_xticks([])
    ax.set_yticks([])
    for lato in ax.spines.values():
        lato.set_visible(False)
    ax.set_title(f"{len(toccate)} comunita' significative toccate su {len(vis)}"
                 f"   ·   cerchio verde = toccata   ·   {etichetta}"
                 f"   ·   il numero e' quanti nodi contiene",
                 fontsize=8.5, color="#3D4548", loc="left")
    fig.tight_layout()
    return fig


def figura_frontiera(d, alpha_corrente: float, asse_reale: bool):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    alpha = sorted(d.punti)
    xs = [d.punti[a].comunita_raggiunte for a in alpha]
    ys = [d.punti[a].adottanti if asse_reale else d.punti[a].diffusione
          for a in alpha]
    ax.plot(xs, ys, color=PETROLIO, lw=1.6, zorder=2)
    ax.scatter(xs, ys, s=26, color=PETROLIO, zorder=3)
    for a, xv, yv in zip(alpha, xs, ys):
        if a in (1.00, 0.70, 0.50):
            ax.annotate(f"α={a:.2f}", (xv, yv), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color="#3D4548")

    p = d.punti[alpha_corrente]
    xc = p.comunita_raggiunte
    yc = p.adottanti if asse_reale else p.diffusione
    ax.scatter([xc], [yc], s=150, facecolors="none", edgecolors=BORDEAUX,
               linewidths=2.2, zorder=4)

    for nome, colore in (("grado massimo", AMBRA), ("casuale", GRIGIO)):
        b = d.baseline[nome]
        ax.scatter([b.comunita_raggiunte],
                   [b.adottanti if asse_reale else b.diffusione],
                   s=44, marker="s", color=colore, zorder=3)
        ax.annotate(nome, (b.comunita_raggiunte,
                           b.adottanti if asse_reale else b.diffusione),
                    textcoords="offset points", xytext=(6, -9),
                    fontsize=7.5, color=colore)

    ax.set_xlabel("comunita' raggiunte dalla cascata", fontsize=9)
    ax.set_ylabel("adottanti reali attesi" if asse_reale
                  else "diffusione attesa (nodi)", fontsize=9)
    ax.grid(alpha=0.18, lw=0.6)
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    fig.tight_layout()
    return fig


def main():
    sito = st.sidebar.selectbox("sito", list(K_PER_SITO))
    k = K_PER_SITO[sito]
    st.sidebar.caption(
        f"k = {k} seed. Il valore cambia col sito: cinquanta seed su 1.653 "
        f"nodi coprirebbero tutto e il vincolo non avrebbe niente da fare.")
    d = carica(sito, k)

    st.sidebar.markdown("### Comandi")
    alpha = st.sidebar.select_slider(
        "diversita'  α", options=list(GRIGLIA), value=0.70,
        help="α=1 e' TIM+ esatto. Abbassandolo, al guadagno di copertura si "
             "somma un premio per le comunita' non ancora toccate.")
    mostra = st.sidebar.toggle("mostra chi ha adottato davvero", value=False)
    asse_reale = st.sidebar.toggle("frontiera su adottanti reali", value=False,
                                   help="Passa dall'asse simulato a quello "
                                        "osservato: e' il confronto che una "
                                        "demo di raccomandazione non puo' fare.")

    st.sidebar.markdown(
        f"---\n**Il tasso di scambio** e' (1−α)/α = **{(1-alpha)/alpha:.3f}**: "
        f"quanta copertura in piu' deve avere un nodo ridondante per battere "
        f"uno che apre una comunita' nuova."
        + ("\n\nA α ≤ 0,5 il rapporto supera 1 ed e' irraggiungibile: "
           "la diversita' domina sempre." if (1 - alpha) / alpha >= 1 else ""))
    st.sidebar.markdown(
        f"---\n**Il tetto.** Il {100*(1-d.tetto):.0f}% degli adottanti reali "
        f"sono utenti iscritti dopo il confine temporale: irraggiungibili per "
        f"costruzione. Tutto quello che si vede vive sul {100*d.tetto:.0f}% "
        f"restante.")

    p = d.punti[alpha]
    base = d.punti[1.00]

    st.markdown(f"#### Semina &nbsp;·&nbsp; `{d.sito}` &nbsp;·&nbsp; "
                f"{d.n_nodi:,} nodi &nbsp;·&nbsp; {d.n_comunita} comunita' "
                f"&nbsp;·&nbsp; addestrato fino a dicembre 2019")

    c = st.columns(4)
    c[0].metric("comunita' dei seed", p.comunita_semi,
                delta=f"{p.comunita_semi - base.comunita_semi:+d}"
                if alpha != 1.0 else None)
    c[1].metric("diffusione attesa", f"{p.diffusione:.0f}",
                delta=f"{100*(p.diffusione/base.diffusione-1):+.1f}%"
                if alpha != 1.0 else None)
    c[2].metric("comunita' raggiunte", f"{p.comunita_raggiunte:.1f}",
                delta=f"{100*(p.comunita_raggiunte/base.comunita_raggiunte-1):+.0f}%"
                if alpha != 1.0 else None)
    c[3].metric("adottanti reali raggiunti", f"{p.adottanti:.1f}",
                delta=f"{100*(p.adottanti/base.adottanti-1):+.1f}%"
                if alpha != 1.0 else None,
                help=f"Su {d.adottanti_noti:,} adottanti reali del 2020-2022 "
                     f"gia' presenti nella rete nel 2019.")

    sinistra, destra = st.columns([2, 1])
    with sinistra:
        st.pyplot(figura_mappa(d, p, mostra), use_container_width=True)
        st.pyplot(figura_frontiera(d, alpha, asse_reale), use_container_width=True)
    with destra:
        st.markdown("**I seed, e perche'**")
        for i, (nodo, guadagno, com, apre) in enumerate(p.motivi[:14], 1):
            if apre and guadagno == 0:
                motivo = f"apre la comunita' {com}, che nessun altro tocca"
                colore = BORDEAUX
            elif apre:
                motivo = f"apre la comunita' {com}; copre {guadagno:,} insiemi RR"
                colore = PETROLIO
            else:
                motivo = f"copre {guadagno:,} insiemi RR; comunita' {com} gia' toccata"
                colore = AMBRA
            st.markdown(
                f"<div style='border-left:3px solid {colore};padding:2px 0 2px 8px;"
                f"margin-bottom:6px'><code>utente {nodo}</code><br>"
                f"<span style='font-size:0.82em;color:#5C6672'>{motivo}</span></div>",
                unsafe_allow_html=True)
        if len(p.motivi) > 14:
            st.caption(f"... altri {len(p.motivi)-14}")

    st.markdown("---")
    tempi = st.columns(len(d.tempi) + 1)
    for i, (nome, secondi) in enumerate(d.tempi.items()):
        tempi[i].markdown(
            f"<span style='font-size:0.75em;color:#6E7679'>{nome}</span><br>"
            f"<b>{secondi:.2f} s</b>", unsafe_allow_html=True)
    tempi[-1].markdown(
        f"<span style='font-size:0.75em;color:#6E7679'>totale precalcolo</span>"
        f"<br><b>{sum(d.tempi.values()):.1f} s</b>", unsafe_allow_html=True)
    st.caption(
        "Tempi misurati durante il precalcolo. Gli insiemi RR si generano una "
        "volta sola e restano in cache: muovere lo slider rilegge un risultato "
        "gia' calcolato, e il greedy da solo costa mezzo secondo.")


if __name__ == "__main__":
    main()
