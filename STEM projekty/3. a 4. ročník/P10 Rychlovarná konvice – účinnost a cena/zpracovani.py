# -*- coding: utf-8 -*-
"""
P10 Rychlovarná konvice – účinnost a cena (zpracování měření)
------------------------------------------------------------
Program načte teplotu vody v konvici v čase, proloží lineární část přímkou t = k·τ + q
(lineární regrese), z rychlosti ohřevu k spočítá užitečný výkon P_u = c·m·k, účinnost
η = P_u / P a cenu ohřevu vody. Nakonec porovná konvici s dalšími způsoby ohřevu.

Spuštění:   python3 zpracovani.py [data.csv] [graf.png]
            (bez parametrů použije vzorova_data.csv a uloží graf.png)
Tabulka (CSV, oddělovač čárka, desetinná tečka):  cas (s), teplota (°C)
Potřebné knihovny: numpy, matplotlib.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- ZDE ZAPIŠ SVÉ HODNOTY ----
HMOTNOST_VODY_KG = 1.000     # zvážená voda v konvici (kg)
PRIKON_W = 2200.0            # příkon ze štítku konvice (W); s wattmetrem zapiš naměřenou hodnotu
CENA_KWH = 6.00              # cena elektřiny v Kč za 1 kWh (z vyúčtování nebo ceníku, včetně distribuce a DPH)
T_OD, T_DO = 25.0, 85.0      # do regrese beru jen teploty v tomto rozmezí (lineární část ohřevu)
C_VODY = 4180.0              # měrná tepelná kapacita vody J/(kg·K)
LITRU_DENNE = 2.0            # kolik litrů vody denně ohřeješ (pro roční náklady)
# Druhý pokus: stejná voda, víko zavřené, stopky do automatického vypnutí konvice (None = neměřeno)
VAR_DOBA_S = 212.0           # doba od zapnutí do vypnutí (s)
VAR_T_ZACATEK = 14.5         # teplota vody na začátku druhého pokusu (°C)
# Srovnání s jinými způsoby ohřevu: (název, příkon W, hmotnost kg, teplota na začátku °C, na konci °C, doba s)
SROVNANI = [
    ("mikrovlnná trouba", 1200.0, 0.300, 15.0, 45.8, 60.0),
    ("elektrická plotna", 1500.0, 1.000, 15.2, 80.0, 420.0),
]


def cz(x, des=2):
    """Číslo jako text s desetinnou čárkou."""
    return f"{x:.{des}f}".replace(".", ",")


def carky(ax):
    """Popisky os grafu s desetinnou čárkou."""
    f = FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def nacti_data(soubor):
    """Vrátí pole časů τ (s) a teplot t (°C)."""
    with open(soubor, encoding="utf-8-sig") as f:
        text = f.read()
    oddelovac = ";" if text.count(";") > text.count(",") else ","
    tau, t = [], []
    for radek in list(csv.reader(text.splitlines(), delimiter=oddelovac))[1:]:
        if len(radek) < 2 or not radek[0].strip():
            continue
        tau.append(float(radek[0].strip().replace(",", ".")))
        t.append(float(radek[1].strip().replace(",", ".")))
    return np.array(tau), np.array(t)


def regrese(x, y):
    """Lineární regrese y = k·x + q metodou nejmenších čtverců; vrátí k, q, R²."""
    xp, yp = np.mean(x), np.mean(y)
    k = np.sum((x - xp) * (y - yp)) / np.sum((x - xp) ** 2)
    q = yp - k * xp
    r2 = 1 - np.sum((y - k * x - q) ** 2) / np.sum((y - yp) ** 2)
    return k, q, r2


def cena_za_litr(ucinnost, t1=15.0, t2=100.0):
    """Cena ohřevu 1 kg (1 l) vody z t1 na t2 při dané účinnosti (Kč) a potřebná energie v kWh."""
    energie_kwh = C_VODY * 1.0 * (t2 - t1) / ucinnost / 3.6e6
    return energie_kwh * CENA_KWH, energie_kwh


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else "vzorova_data.csv"
    graf = sys.argv[2] if len(sys.argv) > 2 else "graf.png"
    tau, t = nacti_data(soubor)

    # 1) Lineární regrese v lineární části ohřevu
    vyber = (t >= T_OD) & (t <= T_DO)
    k, q, r2 = regrese(tau[vyber], t[vyber])
    print(f"Regrese t = k·τ + q ({vyber.sum()} bodů mezi {cz(T_OD, 0)} a {cz(T_DO, 0)} °C):")
    print(f"  k = {cz(k, 4)} °C/s  ({cz(k * 60, 1)} °C za minutu),  q = {cz(q, 1)} °C,  R² = {cz(r2, 5)}")
    zbytky = t[vyber] - (k * tau[vyber] + q)
    print(f"  největší odchylka bodu od přímky: {cz(np.max(np.abs(zbytky)), 2)} °C")

    # 2) Užitečný výkon a účinnost
    m = HMOTNOST_VODY_KG
    P_u = C_VODY * m * k
    eta = P_u / PRIKON_W
    print(f"\nUžitečný výkon P_u = c·m·k = {cz(C_VODY, 0)} · {cz(m, 3)} · {cz(k, 4)} = {cz(P_u, 0)} W")
    print(f"Účinnost η = P_u / P = {cz(P_u, 0)} / {cz(PRIKON_W, 0)} = {cz(eta, 3)}  →  {cz(eta * 100, 1)} %")

    # 3) Ohřev do varu: odhad z přímky a druhý pokus až do automatického vypnutí
    tau_var = (100 - q) / k
    print(f"\nPodle přímky by voda dosáhla 100 °C asi za {cz(tau_var, 0)} s ({cz(tau_var / 60, 1)} min).")
    eta_celk = None
    if VAR_DOBA_S:
        E_kwh = PRIKON_W * VAR_DOBA_S / 3.6e6
        Q = C_VODY * m * (100 - VAR_T_ZACATEK)
        eta_celk = Q / (PRIKON_W * VAR_DOBA_S)
        print(f"Druhý pokus – var do vypnutí za {cz(VAR_DOBA_S, 0)} s:")
        print(f"  spotřeba E = P·τ = {cz(E_kwh, 4)} kWh = {cz(E_kwh * 3.6e3, 0)} kJ, teplo pro vodu Q = {cz(Q / 1000, 0)} kJ")
        print(f"  účinnost celého ohřevu η = Q / E = {cz(eta_celk * 100, 1)} %,"
              f"  cena jednoho varu {cz(E_kwh * CENA_KWH, 2)} Kč (při {cz(CENA_KWH, 2)} Kč/kWh)")

    # 4) Srovnání způsobů ohřevu: cena ohřevu 1 l vody z 15 °C na 100 °C
    zpusoby = [("konvice (lineární část)", eta)]
    if eta_celk:
        zpusoby.append(("konvice až do vypnutí", eta_celk))
    for nazev, P, mm, t1, t2, doba in SROVNANI:
        zpusoby.append((nazev, C_VODY * mm * (t2 - t1) / (P * doba)))
    print(f"\nSrovnání – ohřev 1 l vody z 15 °C na 100 °C (za rok při {cz(LITRU_DENNE, 1)} l denně):")
    ceny = []
    for nazev, u in zpusoby:
        cena, e = cena_za_litr(u)
        ceny.append(cena)
        print(f"  {nazev:<24} η = {cz(u * 100, 0):>2} %  {cz(e, 3)} kWh  {cz(cena, 2)} Kč"
              f"   za rok {cz(cena * LITRU_DENNE * 365, 0):>5} Kč")

    # 5) Graf: vlevo t(τ) s přímkou, vpravo cena za litr
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot(tau[~vyber], t[~vyber], "o", mfc="white", color="#C8651B", label="měření mimo regresi")
    ax1.plot(tau[vyber], t[vyber], "o", color="#C8651B", label="měření v regresi")
    tm = np.linspace(0, tau.max(), 10)
    ax1.plot(tm, k * tm + q, "-", color="#2E6DB4", label=f"t = {cz(k, 3)}·τ + {cz(q, 1)}")
    ax1.set_xlabel("čas τ (s)")
    ax1.set_ylabel("teplota vody t (°C)")
    ax1.set_title(f"Ohřev vody v konvici, η = {cz(eta * 100, 0)} %")
    ax1.grid(alpha=0.3)
    ax1.legend()
    carky(ax1)
    nazvy = [z[0].replace(" ", "\n", 1) for z in zpusoby]
    sloupce = ax2.bar(nazvy, ceny, color=["#2E6DB4", "#5B8FD0", "#C8651B", "#7B4FA0", "#3C8D40"][:len(ceny)])
    for s, (nazev, u), c in zip(sloupce, zpusoby, ceny):
        ax2.text(s.get_x() + s.get_width() / 2, c, f"{cz(c, 2)} Kč\nη {cz(u * 100, 0)} %", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("cena ohřevu 1 l z 15 °C na 100 °C (Kč)")
    ax2.set_ylim(0, max(ceny) * 1.3)
    ax2.set_title(f"Srovnání při {cz(CENA_KWH, 2)} Kč/kWh")
    ax2.tick_params(axis="x", labelsize=8)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f"\nGraf uložen do souboru {graf}.")


if __name__ == "__main__":
    main()
