# -*- coding: utf-8 -*-
"""
P08 Spektroskop z CD – zpracování fotek spekter
------------------------------------------------
1) Z každé fotky spektra (jpg/png) spočítá PROFIL JASU podél spektra a vypíše polohy
   nejvýraznějších maxim v pixelech (nápověda pro vyplnění tabulky).
2) Z čar zářivky se známou vlnovou délkou udělá KALIBRACI  λ = a·x + b  (lineární regrese)
   a spočítá vlnové délky ostatních míst ve spektrech (vrchol LED, okraje spektra žárovky…).
3) Z „duhového prstence“ na CD spočítá vzdálenost drážek CD (mřížkovou konstantu d).

Spuštění:   python3 zpracovani.py [data.csv] [graf.png]
            (bez parametrů použije vzorova_data.csv a uloží graf.png)
Tabulka (CSV, oddělovač čárka, desetinná tečka):
    snimek, popis, x (px), lambda (nm)
    snimek = název souboru s fotkou (fotky dej do stejné složky jako tabulku);
    x = poloha čáry na fotce – při prvním spuštění ji nech prázdnou, program vypíše polohy maxim;
    lambda vyplň jen u čar zářivky se známou vlnovou délkou (kalibrace).
Všechny fotky musí být pořízené ze STEJNÉ polohy telefonu vůči spektroskopu (jinak kalibrace neplatí)
a spektrum musí na fotce ležet vodorovně.
Potřebné knihovny: numpy, matplotlib.
"""
import sys
import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- Úkol 5: duhový prstenec na CD (svítilna telefonu těsně u fotoaparátu, CD rovnoběžně s telefonem) ----
L_PRSTENEC_MM = 250.0      # vzdálenost telefonu od CD (mm)
R_PRSTENEC_MM = 35.0       # poloměr modrého kruhu na CD (mm) – změř na fotce, CD má průměr 120 mm
LAMBDA_PRSTENEC_NM = None  # vlnová délka modrého vrcholu LED; None = vezme se z kalibrace (řádek „LED“)
D_CD_NM = 1600.0           # vzdálenost drážek CD podle normy (1,6 µm)


def cz(x, des=1):
    """Číslo jako text s desetinnou čárkou."""
    return f"{x:.{des}f}".replace(".", ",")


def carky(ax):
    """Popisky os grafu s desetinnou čárkou."""
    f = FuncFormatter(lambda v, pos: f"{v:g}".replace(".", ","))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def nacti_tabulku(soubor):
    """Vrátí seznam řádků [snimek, popis, x, lambda]; nevyplněné x nebo lambda jsou None."""
    with open(soubor, encoding="utf-8-sig") as f:
        text = f.read()
    oddelovac = ";" if text.count(";") > text.count(",") else ","
    radky = []
    for r in list(csv.reader(text.splitlines(), delimiter=oddelovac))[1:]:
        if len(r) < 3 or not r[0].strip():
            continue
        x = r[2].strip().replace(",", ".")
        lam = r[3].strip().replace(",", ".") if len(r) > 3 else ""
        radky.append([r[0].strip(), r[1].strip(), float(x) if x else None, float(lam) if lam else None])
    return radky


def nacti_snimek(soubor):
    """Načte fotku jako pole čísel 0–1 tvaru (výška, šířka, 3 barvy R, G, B)."""
    obr = plt.imread(soubor).astype(float)
    if obr.max() > 1.0:          # jpg má hodnoty 0–255
        obr = obr / 255.0
    if obr.ndim == 2:            # černobílý obrázek → tři stejné kanály
        obr = np.dstack([obr, obr, obr])
    return obr[:, :, :3]


def profil_jasu(obr):
    """Profil jasu podél spektra: průměr přes řádky, kde leží pruh spektra (jas řádku nad polovinou maxima)."""
    jas = obr.mean(axis=2)                     # jas pixelu = průměr R, G, B
    jas_radku = jas.mean(axis=1)
    pozadi = np.median(jas_radku)
    radky = jas_radku > pozadi + 0.5 * (jas_radku.max() - pozadi)
    profil = jas[radky].mean(axis=0) - np.median(jas[~radky])   # odečteme tmavé pozadí
    barvy = obr[radky].mean(axis=0)            # průměrná barva R, G, B v každém sloupci
    return profil, barvy


def najdi_vrcholy(profil, kolik=6, okno=6, min_podil=0.12):
    """Lokální maxima profilu (vyšší než okolí ±okno px a než min_podil·maximum), seřazená podle výšky."""
    vrcholy = []
    for i in range(okno, len(profil) - okno):
        kus = profil[i - okno:i + okno + 1]
        if profil[i] == kus.max() and profil[i] > min_podil * profil.max():
            vrcholy.append(i)
    vrcholy.sort(key=lambda i: -profil[i])
    return sorted(vrcholy[:kolik])


def nazev_barvy(rgb):
    """Hrubý název barvy podle poměru R, G, B (pomůže poznat čáru)."""
    r, g, b = rgb / max(rgb.max(), 1e-9)
    if b > 0.8 and r < 0.6 and g < 0.6:
        return "fialová/modrá"
    if b > 0.7 and g > 0.7:
        return "modrozelená"
    if g > 0.8 and r < 0.7:
        return "zelená"
    if r > 0.8 and g > 0.6:
        return "žlutá/oranžová"
    if r > 0.8:
        return "červená"
    return "nejasná"


def regrese(x, y):
    """Lineární regrese y = a·x + b metodou nejmenších čtverců."""
    xp, yp = np.mean(x), np.mean(y)
    a = np.sum((x - xp) * (y - yp)) / np.sum((x - xp) ** 2)
    return a, yp - a * xp


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else "vzorova_data.csv"
    graf = sys.argv[2] if len(sys.argv) > 2 else "graf.png"
    slozka = os.path.dirname(os.path.abspath(soubor))
    tab = nacti_tabulku(soubor)

    # 1) Profily jasu a nápověda s polohami maxim
    profily = {}
    for snimek in dict.fromkeys(r[0] for r in tab):          # každý snímek jen jednou, v pořadí tabulky
        cesta = os.path.join(slozka, snimek)
        if not os.path.exists(cesta):
            print(f"(Fotka {snimek} nenalezena – profil nespočítám, použiji jen tabulku.)")
            continue
        profil, barvy = profil_jasu(nacti_snimek(cesta))
        profily[snimek] = profil
        # nápovědu vypíšeme pro snímek s kalibračními čarami a pro snímky, u kterých ještě chybí poloha x
        radky_snimku = [r for r in tab if r[0] == snimek]
        if any(r[3] is not None or r[2] is None for r in radky_snimku):
            print(f"Snímek {snimek}: šířka {len(profil)} px, výrazná maxima jasu:")
            for i in najdi_vrcholy(profil):
                print(f"   x = {i:4d} px   jas {cz(profil[i] / profil.max() * 100, 0):>3} %   barva: {nazev_barvy(barvy[i])}")

    # 2) Kalibrace z řádků se známou vlnovou délkou
    kal = [r for r in tab if r[2] is not None and r[3] is not None]
    if len(kal) < 2:
        print("\nPro kalibraci potřebuji aspoň 2 čáry zářivky s vyplněnou polohou x a známou vlnovou délkou.")
        print("Podle výpisu maxim doplň do tabulky polohy x a spusť program znovu.")
        return
    xk = np.array([r[2] for r in kal])
    lk = np.array([r[3] for r in kal])
    a, b = regrese(xk, lk)
    print(f"\nKalibrace z {len(kal)} čar zářivky:  λ = {cz(a, 4)} · x + {cz(b, 1)}   (λ v nm, x v px)")
    print(f"  jeden pixel odpovídá {cz(a, 3)} nm")
    for r, x, l in zip(kal, xk, lk):
        print(f"  {r[1]:<32} x = {cz(x, 0):>4} px  známá λ = {cz(l, 1)} nm  model {cz(a * x + b, 1)} nm"
              f"  odchylka {cz(a * x + b - l, 1)} nm")

    # 3) Vlnové délky ostatních míst ve spektrech
    print("\nVlnové délky z kalibrace:")
    vysledky = {}
    for r in tab:
        if r[2] is not None and r[3] is None:
            lam = a * r[2] + b
            vysledky[r[1]] = lam
            energie = 1240 / lam                      # energie fotonu v eV (hc ≈ 1240 eV·nm)
            print(f"  {r[0]:<16} {r[1]:<32} x = {cz(r[2], 0):>4} px  →  λ = {cz(lam, 0)} nm,"
                  f"  foton E = {cz(energie, 2)} eV")

    # 4) Mřížková rovnice: úhly ohybu 1. řádu při kolmém dopadu (d·sin α = k·λ)
    print(f"\nÚhly ohybu 1. řádu na CD (d = {cz(D_CD_NM / 1000, 1)} µm, kolmý dopad):")
    for lam in (400, 436, 546, 612, 700):
        print(f"  λ = {lam} nm  →  α = {cz(math.degrees(math.asin(lam / D_CD_NM)), 1)}°")

    # 5) Duhový prstenec: světlo se vrací ke zdroji → 2·d·sin θ = λ, kde tg θ = r / L
    lam_p = LAMBDA_PRSTENEC_NM
    if lam_p is None:
        led = [v for k, v in vysledky.items() if "LED" in k or "led" in k]
        lam_p = led[0] if led else 450.0
    theta = math.atan(R_PRSTENEC_MM / L_PRSTENEC_MM)
    d = lam_p / (2 * math.sin(theta))
    print(f"\nDuhový prstenec: L = {cz(L_PRSTENEC_MM, 0)} mm, r = {cz(R_PRSTENEC_MM, 1)} mm → θ = {cz(math.degrees(theta), 2)}°")
    print(f"  pro λ = {cz(lam_p, 0)} nm vychází d = λ / (2 sin θ) = {cz(d / 1000, 3)} µm ({cz(1e6 / d, 0)} drážek na mm)")
    print(f"  norma CD: 1,6 µm, odchylka {cz((d - D_CD_NM) / D_CD_NM * 100, 1)} %")

    # 6) Graf: kalibrace, fotka zářivky se stupnicí λ, profily všech spekter
    fig = plt.figure(figsize=(10, 7.2))
    ax1 = fig.add_subplot(2, 2, 1)
    xm = np.linspace(xk.min() - 30, xk.max() + 30, 10)
    ax1.plot(xk, lk, "o", color="#C8651B", label="čáry zářivky")
    ax1.plot(xm, a * xm + b, "-", color="#2E6DB4", label=f"λ = {cz(a, 3)}·x + {cz(b, 0)}")
    ax1.set_xlabel("poloha na fotce x (px)")
    ax1.set_ylabel("vlnová délka λ (nm)")
    ax1.set_title("Kalibrace spektroskopu")
    ax1.grid(alpha=0.3)
    ax1.legend()
    carky(ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    prvni = kal[0][0]
    if prvni in profily:
        obr = nacti_snimek(os.path.join(slozka, prvni))
        sirka = obr.shape[1]
        ax2.imshow(obr, extent=[b, a * sirka + b, obr.shape[0], 0], aspect="auto")
        ax2.set_xlim(380, 720)
        ax2.set_yticks([])
        ax2.set_xlabel("vlnová délka λ (nm)")
        ax2.set_title(f"Fotka {prvni} se stupnicí vlnových délek")
        carky(ax2)
    else:
        ax2.axis("off")

    ax3 = fig.add_subplot(2, 1, 2)
    barva_krivky = ["#7B4FA0", "#2E6DB4", "#C8651B", "#3C8D40", "#555555"]
    for i, (snimek, profil) in enumerate(profily.items()):
        lam_osa = a * np.arange(len(profil)) + b
        ax3.plot(lam_osa, profil / profil.max() + 1.1 * i, color=barva_krivky[i % 5],
                 label=os.path.splitext(snimek)[0])
    for l in lk:
        ax3.axvline(l, color="gray", ls=":", lw=1)
    ax3.set_xlim(380, 720)
    ax3.set_xlabel("vlnová délka λ (nm)")
    ax3.set_ylabel("relativní jas (křivky posunuté nad sebe)")
    ax3.set_title("Profily jasu spekter (tečkovaně kalibrační čáry)")
    ax3.set_yticks([])
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)
    carky(ax3)
    ax3.set_yticks([])
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f"\nGraf uložen do souboru {graf}.")


if __name__ == "__main__":
    main()
