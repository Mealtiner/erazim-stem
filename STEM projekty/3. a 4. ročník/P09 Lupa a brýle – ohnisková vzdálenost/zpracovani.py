# -*- coding: utf-8 -*-
"""
P09 Lupa a brýle – ohnisková vzdálenost (zpracování měření)
----------------------------------------------------------
Program načte dvojice vzdáleností předmětu a a obrazu a' (obraz ostrý na papíře),
spočítá ohniskovou vzdálenost ze zobrazovací rovnice 1/a + 1/a' = 1/f třemi způsoby:
  1) z každé dvojice zvlášť a průměr se směrodatnou odchylkou,
  2) linearizací 1/a' = −1/a + 1/f a lineární regresí (úsek na ose = optická mohutnost),
  3) porovnáním s hyperbolou (a − f)(a' − f) = f² (asymptoty a = f, a' = f).

Spuštění:   python3 zpracovani.py [data.csv] [graf.png]
            (bez parametrů použije vzorova_data.csv a uloží graf.png)
Tabulka (CSV, oddělovač čárka, desetinná tečka):  a (cm), a' (cm)
Potřebné knihovny: numpy, matplotlib.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

UDANA_MOHUTNOST_D = 10.0   # optická mohutnost uvedená na lupě nebo brýlích v dioptriích; None = neznám


def cz(x, des=2):
    """Číslo jako text s desetinnou čárkou."""
    return f"{x:.{des}f}".replace(".", ",")


def carky(ax):
    """Popisky os grafu s desetinnou čárkou."""
    f = FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def nacti_data(soubor):
    """Vrátí pole vzdáleností a a a' v centimetrech."""
    with open(soubor, encoding="utf-8-sig") as f:
        text = f.read()
    oddelovac = ";" if text.count(";") > text.count(",") else ","
    a, a2 = [], []
    for radek in list(csv.reader(text.splitlines(), delimiter=oddelovac))[1:]:
        if len(radek) < 2 or not radek[0].strip():
            continue
        a.append(float(radek[0].strip().replace(",", ".")))
        a2.append(float(radek[1].strip().replace(",", ".")))
    return np.array(a), np.array(a2)


def regrese(x, y):
    """Lineární regrese y = k·x + q metodou nejmenších čtverců; vrátí k, q, R²."""
    xp, yp = np.mean(x), np.mean(y)
    k = np.sum((x - xp) * (y - yp)) / np.sum((x - xp) ** 2)
    q = yp - k * xp
    r2 = 1 - np.sum((y - k * x - q) ** 2) / np.sum((y - yp) ** 2)
    return k, q, r2


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else "vzorova_data.csv"
    graf = sys.argv[2] if len(sys.argv) > 2 else "graf.png"
    a, a2 = nacti_data(soubor)
    n = len(a)

    # 1) Ohnisková vzdálenost z každé dvojice: f = a·a' / (a + a')
    f_i = a * a2 / (a + a2)
    print("   a (cm)   a' (cm)   f (cm)   zvětšení Z = −a'/a")
    for ai, bi, fi in zip(a, a2, f_i):
        print(f"  {cz(ai, 1):>7}  {cz(bi, 1):>8}  {cz(fi, 2):>7}   {cz(-bi / ai, 2):>6}")
    f_prum = f_i.mean()
    s = f_i.std(ddof=1)                       # výběrová směrodatná odchylka
    chyba = s / np.sqrt(n)                    # směrodatná odchylka průměru
    print(f"\n1) Průměr z {n} dvojic: f = ({cz(f_prum, 2)} ± {cz(chyba, 2)}) cm,"
          f"  směrodatná odchylka jednoho měření {cz(s, 2)} cm")

    # 2) Linearizace: y = 1/a', x = 1/a (v m⁻¹, tj. v dioptriích) → y = k·x + q, očekáváme k = −1, q = φ
    x, y = 100 / a, 100 / a2
    k, q, r2 = regrese(x, y)
    phi = q
    print(f"2) Regrese 1/a' = k·(1/a) + q:  k = {cz(k, 3)} (teorie −1),  q = {cz(q, 2)} m⁻¹,  R² = {cz(r2, 4)}")
    print(f"   optická mohutnost φ = {cz(phi, 2)} D  →  f = 1/φ = {cz(100 / phi, 2)} cm")
    phi3 = np.mean(x + y)                     # model s pevnou směrnicí −1: φ = průměr (1/a + 1/a')
    print(f"   se směrnicí pevně −1: φ = {cz(phi3, 2)} D, f = {cz(100 / phi3, 2)} cm")

    # 3) Hyperbola a' = f·a / (a − f), tj. (a − f)(a' − f) = f²: střed S[f; f], asymptoty a = f a a' = f
    model = f_prum * a / (a - f_prum)
    odchylky = a2 - model
    print(f"3) Hyperbola (a − f)(a' − f) = f²: střed S[{cz(f_prum, 1)}; {cz(f_prum, 1)}] cm,"
          f" asymptoty a = {cz(f_prum, 1)} cm a a' = {cz(f_prum, 1)} cm")
    print(f"   odchylky naměřených a' od modelu: průměrně {cz(np.mean(np.abs(odchylky)), 2)} cm,"
          f" největší {cz(odchylky[np.argmax(np.abs(odchylky))], 2)} cm (při a = {cz(a[np.argmax(np.abs(odchylky))], 1)} cm)")

    # Porovnání s údajem výrobce a úhlové zvětšení lupy
    if UDANA_MOHUTNOST_D:
        odch = (phi - UDANA_MOHUTNOST_D) / UDANA_MOHUTNOST_D * 100
        print(f"\nÚdaj výrobce {cz(UDANA_MOHUTNOST_D, 1)} D, změřeno {cz(phi, 2)} D → odchylka {cz(odch, 1)} %")
    print(f"Úhlové zvětšení lupy (předmět v ohnisku, d = 25 cm): γ = d/f = {cz(25 / f_prum, 1)}")

    # Graf: vlevo hyperbola a'(a) s asymptotami, vpravo linearizace
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
    am = np.linspace(f_prum * 1.08, min(a.max(), 120), 300)
    ax1.plot(a, a2, "o", color="#C8651B", label="měření")
    ax1.plot(am, am * f_prum / (am - f_prum), "-", color="#2E6DB4", label=f"model, f = {cz(f_prum, 2)} cm")
    ax1.axvline(f_prum, ls="--", color="gray", lw=1, label="asymptoty a = f, a' = f")
    ax1.axhline(f_prum, ls="--", color="gray", lw=1)
    ax1.set_xlim(0, min(a.max(), 120) * 1.05)
    ax1.set_ylim(0, a2.max() * 1.15)
    ax1.set_xlabel("vzdálenost předmětu a (cm)")
    ax1.set_ylabel("vzdálenost obrazu a' (cm)")
    ax1.set_title("Hyperbola a'(a)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    xm = np.linspace(0, x.max() * 1.05, 10)
    ax2.plot(x, y, "o", color="#C8651B", label="měření")
    ax2.plot(xm, k * xm + q, "-", color="#2E6DB4", label=f"1/a' = {cz(k, 3)}·(1/a) + {cz(q, 2)}")
    ax2.plot([0], [q], "s", color="#3C8D40", label=f"úsek = φ = {cz(phi, 2)} D")
    ax2.set_xlim(0, x.max() * 1.05)
    ax2.set_ylim(0, max(y.max(), q) * 1.1)
    ax2.set_xlabel("1/a (m⁻¹)")
    ax2.set_ylabel("1/a' (m⁻¹)")
    ax2.set_title("Linearizace zobrazovací rovnice")
    ax2.grid(alpha=0.3)
    ax2.legend()
    carky(ax1)
    carky(ax2)
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f"\nGraf uložen do souboru {graf}.")


if __name__ == "__main__":
    main()
