"""
Skákající míček – zpracování měření (STEM projekt P05, 1. a 2. ročník)

Spuštění:   python3 zpracovani.py [data.csv] [graf.png]
            (bez parametrů zpracuje vzorova_data.csv a uloží graf.png)

Soubor CSV má hlavičku   serie,H (m),dopad,t (s)
  serie   název měření bez mezer, např. pingpong nebo tenisak_koberec
  H (m)   výška, ze které jsi míček pustil (spodní okraj míčku nad podlahou)
  dopad   pořadí dopadu 1, 2, 3, …
  t (s)   čas dopadu od začátku záznamu (phyphox, Akustické stopky, záložka Více,
          export „Event time (s)“)
Desetinná čísla piš s tečkou (1.214), ne s čárkou.

Model: mezi dvěma dopady letí míček polovinu doby nahoru a polovinu dolů, proto
výška odskoku h = g·t²/8. Výšky tvoří geometrickou posloupnost s kvocientem q = e²,
kde e je koeficient restituce (poměr rychlosti po odrazu a před odrazem).
Program potřebuje knihovny numpy a matplotlib.
"""
import sys
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")            # graf jen ukládáme do souboru, žádné okno neotvíráme
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

G = 9.81                         # tíhové zrychlení (m/s²)
BARVY = ["#2a78d6", "#eb6834", "#1baf7a"]    # barvy pro 1., 2. a 3. sérii


def cislo(x, des=3):
    """Vrátí číslo jako text s desetinnou čárkou, např. 0,905."""
    text = f"{x:.{des}f}"
    if float(text) == 0:
        text = text.lstrip("-")           # místo „-0,0“ napíšeme „0,0“
    return text.replace(".", ",")


def carky_na_osach(graf):
    """Na osách grafu napíše čísla s desetinnou čárkou."""
    format_cz = FuncFormatter(lambda x, pozice: f"{x:g}".replace(".", ","))
    graf.xaxis.set_major_formatter(format_cz)
    graf.yaxis.set_major_formatter(format_cz)


def nacti_data(soubor):
    """Načte CSV; vrátí slovník  série → {"H": výška, "t": pole časů dopadů}."""
    serie = {}
    with open(soubor, encoding="utf-8-sig", newline="") as f:
        for radek in csv.DictReader(f):
            if radek["t (s)"].strip() == "":
                continue                             # prázdný řádek přeskočíme
            nazev = radek["serie"].strip()
            if nazev not in serie:
                serie[nazev] = {"H": float(radek["H (m)"]), "dopady": []}
            serie[nazev]["dopady"].append((int(radek["dopad"]), float(radek["t (s)"])))
    for s in serie.values():
        s["dopady"].sort()                           # seřadíme podle čísla dopadu
        s["t"] = np.array([t for _, t in s["dopady"]])
    return serie


def zpracuj_serii(H, t_dopadu):
    """Spočítá doby letu, výšky, koeficient restituce a regresi pro jednu sérii."""
    doby = np.diff(t_dopadu)               # doba letu mezi n-tým a (n+1)-tým dopadem
    n = np.arange(1, len(doby) + 1)        # číslo odskoku 1, 2, 3, …
    h = G * doby ** 2 / 8                  # výška odskoku (m)
    e_pomery = doby[1:] / doby[:-1]        # e = t(n+1) / t(n)
    # Linearizace: ln h = ln h1 + (n − 1)·ln q  → přímka  ln h = k·n + c
    k, c = np.polyfit(n, np.log(h), 1)
    q = math.exp(k)                        # kvocient geometrické posloupnosti výšek
    return {"doby": doby, "n": n, "h": h, "e_pomery": e_pomery,
            "e_prumer": float(np.mean(e_pomery)), "q": q, "e_fit": math.sqrt(q),
            "h0": math.exp(c),             # výška „před 1. dopadem“ (n = 0) podle modelu
            "k": k, "c": c}


def vypis_serii(nazev, H, t_dopadu, v):
    """Vypíše výsledky jedné série česky."""
    print(f"\nSérie „{nazev}“ – puštěno z výšky H = {cislo(H, 2)} m, zaznamenáno {len(t_dopadu)} dopadů")
    print("   n   doba letu (s)   výška h (cm)   zachovaná energie")
    for i in range(len(v["doby"])):
        energie = "" if i == 0 else f"{cislo(100 * v['e_pomery'][i - 1] ** 2, 1)} %"
        print(f"  {v['n'][i]:>2}   {cislo(v['doby'][i]):>12}   {cislo(100 * v['h'][i], 1):>11}   {energie:>10}")
    print(f"  Koeficient restituce z poměrů dob letu: e = {cislo(v['e_prumer'])}")
    print(f"  Regrese ln h = k·n + c:  k = {cislo(v['k'], 4)},  c = {cislo(v['c'], 4)}")
    print(f"  Kvocient výšek q = e^k = {cislo(v['q'])}  →  e = odmocnina z q = {cislo(v['e_fit'])}")
    print(f"  Při každém odrazu se ztratí {cislo(100 * (1 - v['q']), 1)} % mechanické energie.")
    odchylka = 100 * (v["h0"] - H) / H
    print(f"  Model pro n = 0 dává výšku {cislo(100 * v['h0'], 1)} cm, změřeno H = {cislo(100 * H, 1)} cm "
          f"(relativní odchylka {cislo(odchylka, 1)} %).")
    e1 = v["doby"][0] / (2 * math.sqrt(2 * H / G))
    print(f"  Kontrola 1. odrazu z výšky H: e = t1 / (2·odmocnina(2H/g)) = {cislo(e1)}")
    # Součet geometrické řady: t1 + t1·e + t1·e² + … = t1 / (1 − e)
    e = v["e_fit"]
    t1 = v["doby"][0]
    pocet = len(v["doby"])
    soucet = t1 * (1 - e ** pocet) / (1 - e)
    print(f"  Od 1. do posledního dopadu: změřeno {cislo(t_dopadu[-1] - t_dopadu[0], 2)} s, "
          f"součet {pocet} členů geometrické posloupnosti {cislo(soucet, 2)} s.")
    print(f"  Předpověď: od 1. dopadu by míček skákal celkem {cislo(t1 / (1 - e), 1)} s "
          f"(součet nekonečné geometrické řady t1 / (1 − e)).")
    h1 = v["h"][0]
    n_cm = 1 + math.log(0.01 / h1) / math.log(v["q"])
    print(f"  Výška klesne pod 1 cm přibližně po {math.ceil(n_cm)}. odskoku.")


def nakresli_graf(serie, vysledky, soubor_grafu):
    """Tři grafy: výšky odskoků, linearizace ln h a zachovaná energie."""
    fig, (g1, g2, g3) = plt.subplots(1, 3, figsize=(13, 4.3))
    for i, nazev in enumerate(serie):
        barva = BARVY[i % len(BARVY)]
        v = vysledky[nazev]
        n_osa = np.linspace(0, v["n"][-1], 100)
        model_h = np.exp(v["c"] + v["k"] * n_osa)          # h = h0 · q^n
        # 1) výšky odskoků a model geometrické posloupnosti
        g1.plot(v["n"], 100 * v["h"], "o", color=barva, ms=7, mec="white", label=f"{nazev} – měření")
        g1.plot(n_osa, 100 * model_h, "-", color=barva, lw=2, alpha=0.8, label=f"{nazev} – model")
        g1.plot(0, 100 * serie[nazev]["H"], "*", color="#0b0b0b", ms=12,
                label="výška puštění $H$" if i == 0 else None)
        # 2) linearizace: ln h proti n je přímka
        g2.plot(v["n"], np.log(v["h"]), "o", color=barva, ms=7, mec="white", label=nazev)
        g2.plot(n_osa, v["c"] + v["k"] * n_osa, "-", color=barva, lw=2, alpha=0.8,
                label=f"přímka: k = {cislo(v['k'], 3)}")
        # 3) podíl energie, který zůstal po odrazu
        g3.plot(v["n"][1:], 100 * v["e_pomery"] ** 2, "o-", color=barva, ms=7, lw=1.5, mec="white",
                label=nazev)
        g3.axhline(100 * v["q"], color=barva, lw=1, ls="--")
    g1.set_xlabel("číslo odskoku $n$")
    g1.set_ylabel("výška odskoku $h$ (cm)")
    g1.set_title("Výšky odskoků: $h_n = h_0 \\cdot q^n$")
    g1.legend(fontsize=8)
    g2.set_xlabel("číslo odskoku $n$")
    g2.set_ylabel("$\\ln h$  ($h$ v metrech)")
    g2.set_title("Linearizace: $\\ln h = k n + c$")
    g2.legend(fontsize=8)
    g3.set_xlabel("číslo odrazu $n$")
    g3.set_ylabel("zachovaná energie $E_{n}/E_{n-1}$ (%)")
    g3.set_title("Energie po odrazu (čárkovaně: $q$ z regrese)")
    g3.legend(fontsize=8)
    for g in (g1, g2, g3):
        g.grid(alpha=0.3)
        carky_na_osach(g)
    fig.tight_layout()
    fig.savefig(soubor_grafu, dpi=130)


def main():
    soubor_dat = sys.argv[1] if len(sys.argv) > 1 else "vzorova_data.csv"
    soubor_grafu = sys.argv[2] if len(sys.argv) > 2 else "graf.png"
    serie = nacti_data(soubor_dat)
    print(f"Skákající míček – zpracování souboru {soubor_dat}")
    vysledky = {}
    for nazev, s in serie.items():
        if len(s["t"]) < 4:
            print(f"\nSérie „{nazev}“ má méně než 4 dopady – na výpočet je to málo, přeskakuji ji.")
            continue
        vysledky[nazev] = zpracuj_serii(s["H"], s["t"])
        vypis_serii(nazev, s["H"], s["t"], vysledky[nazev])
    if not vysledky:
        print("Žádná série nemá dost dopadů.")
        return
    serie = {k: serie[k] for k in vysledky}
    nakresli_graf(serie, vysledky, soubor_grafu)
    print(f"\nGraf je uložen v souboru {soubor_grafu}.")


if __name__ == "__main__":
    main()
