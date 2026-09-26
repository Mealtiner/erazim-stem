# -*- coding: utf-8 -*-
"""
P07 Magnetometr v mobilu – zpracování měření
---------------------------------------------
Program načte tabulku z měření magnetometrem (phyphox), odečte pole Země,
spočítá velikost pole magnetu a proloží body mocninným modelem B = C · r^(-n)
pomocí linearizace (logaritmus) a lineární regrese.

Spuštění:   python3 zpracovani.py [data.csv] [graf.png]
            (bez parametrů použije vzorova_data.csv a uloží graf.png)

Tabulka (CSV, oddělovač čárka, desetinná tečka):
    r (cm), Bx (µT), By (µT), Bz (µT)
    Řádek s PRÁZDNÝM r = měření bez magnetu (pole Země). Může jich být víc.
Potřebné knihovny: numpy, matplotlib.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")            # graf se jen uloží do souboru, neotevírá se okno
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

MIN_POLE_UT = 2.0   # body, kde je pole magnetu slabší než 2 µT, do regrese nebereme (šum senzoru)


def cz(x, des=2):
    """Číslo jako text s desetinnou čárkou (české psaní čísel)."""
    return f"{x:.{des}f}".replace(".", ",")


def carky(ax):
    """Popisky os grafu s desetinnou čárkou."""
    f = FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def nacti_data(soubor):
    """Vrátí seznam vzdáleností r a seznamy složek Bx, By, Bz (řádky s prázdným r mají r = None)."""
    with open(soubor, encoding="utf-8-sig") as f:
        text = f.read()
    oddelovac = ";" if text.count(";") > text.count(",") else ","      # i CSV z českého Excelu
    radky = list(csv.reader(text.splitlines(), delimiter=oddelovac))
    r, bx, by, bz = [], [], [], []
    for radek in radky[1:]:                       # první řádek je hlavička
        if len(radek) < 4 or not "".join(radek).strip():
            continue
        cisla = [s.strip().replace(",", ".") for s in radek[:4]]
        r.append(float(cisla[0]) if cisla[0] else None)
        bx.append(float(cisla[1]))
        by.append(float(cisla[2]))
        bz.append(float(cisla[3]))
    return r, np.array(bx), np.array(by), np.array(bz)


def primka_regrese(x, y):
    """Lineární regrese y = k·x + q metodou nejmenších čtverců. Vrátí k, q a R² (koeficient determinace)."""
    xp, yp = np.mean(x), np.mean(y)
    k = np.sum((x - xp) * (y - yp)) / np.sum((x - xp) ** 2)
    q = yp - k * xp
    rezidua = y - (k * x + q)
    r2 = 1 - np.sum(rezidua ** 2) / np.sum((y - yp) ** 2)
    return k, q, r2


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else "vzorova_data.csv"
    graf = sys.argv[2] if len(sys.argv) > 2 else "graf.png"
    r, bx, by, bz = nacti_data(soubor)

    # 1) Pole Země = průměr řádků bez magnetu (prázdné r)
    bez = [i for i in range(len(r)) if r[i] is None]
    s = [i for i in range(len(r)) if r[i] is not None]
    if not bez:
        print("Chybí řádek bez magnetu (prázdné r) – pole Země neumím odečíst!")
        return
    b0 = np.array([bx[bez].mean(), by[bez].mean(), bz[bez].mean()])
    print(f"Pole Země (průměr z {len(bez)} měření bez magnetu):")
    print(f"  Bx = {cz(b0[0], 1)} µT, By = {cz(b0[1], 1)} µT, Bz = {cz(b0[2], 1)} µT,"
          f"  velikost |B0| = {cz(np.linalg.norm(b0), 1)} µT")

    # 2) Pole magnetu = naměřený vektor minus pole Země (odečítáme po složkách!)
    rr = np.array([r[i] for i in s])
    dbx, dby, dbz = bx[s] - b0[0], by[s] - b0[1], bz[s] - b0[2]
    B = np.sqrt(dbx ** 2 + dby ** 2 + dbz ** 2)
    print("\n  r (cm)   B magnetu (µT)")
    for ri, bi in zip(rr, B):
        print(f"  {cz(ri, 1):>6}   {cz(bi, 1):>10}")

    # 3) Linearizace: ln B = ln C − n · ln r  →  přímka y = k·x + q, kde x = ln r, y = ln B
    pouzit = B > MIN_POLE_UT
    x, y = np.log(rr[pouzit]), np.log(B[pouzit])
    k, q, r2 = primka_regrese(x, y)
    n, C = -k, np.exp(q)
    print(f"\nMocninný model B = C · r^(−n)  (regrese ln B na ln r, {pouzit.sum()} bodů):")
    print(f"  exponent n = {cz(n, 2)}   (dipól: n = 3, odchylka {cz(abs(n - 3) / 3 * 100, 1)} %)")
    print(f"  konstanta C = {cz(C, 0)}   (B v µT, r v cm)")
    print(f"  koeficient determinace R² = {cz(r2, 4)}")

    # 4) Model s pevným exponentem 3: ln C3 = průměr hodnot ln(B·r³); z C3 magnetický moment magnetu
    C3 = np.exp(np.mean(np.log(B[pouzit] * rr[pouzit] ** 3)))      # µT·cm³
    m = C3 * 1e-6 * 1e-6 / 2e-7          # na ose B = μ0·m / (2π r³) → m = B·r³ / (2·10⁻⁷) v jednotkách SI
    odch = (B[pouzit] - C3 / rr[pouzit] ** 3) / B[pouzit] * 100
    print(f"\nModel s pevným n = 3: C = {cz(C3, 0)} µT·cm³,  magnetický moment m ≈ {cz(m, 3)} A·m²")
    print(f"  relativní odchylky bodů od modelu: od {cz(odch.min(), 1)} % do {cz(odch.max(), 1)} %")

    # 5) Kde je pole magnetu stejně silné jako pole Země?
    r_zeme = (C / np.linalg.norm(b0)) ** (1 / n)
    print(f"\nPole magnetu se vyrovná poli Země ve vzdálenosti asi {cz(r_zeme, 1)} cm.")

    # 6) Graf: vlevo B(r), vpravo log-log linearizace
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    rm = np.linspace(rr.min() * 0.95, rr.max() * 1.05, 200)
    ax1.plot(rr, B, "o", color="#C8651B", label="měření")
    ax1.plot(rm, C * rm ** (-n), "-", color="#2E6DB4", label=f"model B = C·r^(−{cz(n, 2)})")
    ax1.axhline(np.linalg.norm(b0), ls="--", color="gray", label="velikost pole Země")
    ax1.set_xlabel("vzdálenost r (cm)")
    ax1.set_ylabel("magnetická indukce B (µT)")
    ax1.set_title("Pole magnetu v závislosti na vzdálenosti")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.plot(np.log(rr), np.log(B), "o", color="#C8651B", label="měření")
    xm = np.linspace(x.min() - 0.05, x.max() + 0.05, 10)
    ax2.plot(xm, k * xm + q, "-", color="#2E6DB4",
             label=f"přímka y = {cz(k, 2)}·x + {cz(q, 2)}\nR² = {cz(r2, 4)}")
    ax2.set_xlabel("ln (r / cm)")
    ax2.set_ylabel("ln (B / µT)")
    ax2.set_title("Linearizace: log-log graf")
    ax2.grid(alpha=0.3)
    ax2.legend()
    carky(ax1)
    carky(ax2)
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f"\nGraf uložen do souboru {graf}.")


if __name__ == "__main__":
    main()
