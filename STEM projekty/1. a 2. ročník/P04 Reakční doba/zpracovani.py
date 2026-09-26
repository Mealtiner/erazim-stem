"""
Reakční doba – zpracování měření (STEM projekt P04, 1. a 2. ročník)

Spuštění:   python3 zpracovani.py [data.csv] [graf.png]
            (bez parametrů zpracuje vzorova_data.csv a uloží graf.png)

Soubor CSV má hlavičku   metoda,ruka,doba_dne,h (cm),t (s)
  metoda    pravitko (chytání padajícího pravítka) nebo program (mereni_reakce.py)
  ruka      dominantni nebo nedominantni
  doba_dne  rano nebo vecer
  h (cm)    o kolik centimetrů pravítko propadlo (jen u metody pravitko)
  t (s)     reakční doba v sekundách (u pravítka ji program dopočítá z h)
Desetinná čísla piš s tečkou (0.215), ne s čárkou.

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

G = 9.81          # tíhové zrychlení (m/s²)
DH = 0.5          # přesnost čtení polohy prstů na pravítku (cm) – půl dílku
V_AUTA = 50       # rychlost auta pro úvahu o zastavovací dráze (km/h)

# Hodnoty ze souboru píšeme bez diakritiky, ve výpisu je chceme česky.
NAZVY = {"pravitko": "pravítko", "program": "program",
         "dominantni": "dominantní ruka", "nedominantni": "nedominantní ruka",
         "rano": "ráno", "vecer": "večer"}

# Barvy v grafu (modrá = pravítko, oranžová = program, tmavě šedá = model).
BARVY = {"pravitko": "#2a78d6", "program": "#eb6834"}
BARVA_MODELU = "#52514e"


def cislo(x, des=3):
    """Vrátí číslo jako text s desetinnou čárkou, např. 0,215."""
    text = f"{x:.{des}f}"
    if float(text) == 0:
        text = text.lstrip("-")           # místo „-0,0“ napíšeme „0,0“
    return text.replace(".", ",")


def carky_na_osach(graf):
    """Na svislé ose grafu napíše čísla s desetinnou čárkou."""
    graf.yaxis.set_major_formatter(FuncFormatter(lambda x, pozice: f"{x:g}".replace(".", ",")))


def nazev(slovo):
    """Český název skupiny (když ho neznáme, vrátí slovo beze změny)."""
    return NAZVY.get(slovo, slovo)


def doba_padu(h_cm):
    """Doba volného pádu (s) pro dráhu h v centimetrech: t = odmocnina z 2h/g."""
    h = h_cm / 100                       # centimetry převedeme na metry
    return math.sqrt(2 * h / G)


def nacti_data(soubor):
    """Načte CSV a vrátí seznam měření; každé měření je slovník."""
    mereni = []
    with open(soubor, encoding="utf-8-sig", newline="") as f:
        for radek in csv.DictReader(f):
            metoda = radek["metoda"].strip().lower()
            h_text = radek["h (cm)"].strip().replace(",", ".")
            t_text = radek["t (s)"].strip().replace(",", ".")
            if metoda == "pravitko" and h_text != "":
                h = float(h_text)
                t = doba_padu(h)             # čas spočítáme z modelu volného pádu
            elif t_text != "":
                h = None
                t = float(t_text)            # čas změřil přímo program
            else:
                continue                     # prázdný řádek přeskočíme
            mereni.append({"metoda": metoda, "ruka": radek["ruka"].strip().lower(),
                           "doba": radek["doba_dne"].strip().lower(), "h": h, "t": t})
    return mereni


def vyber(mereni, metoda=None, ruka=None, doba=None):
    """Vrátí seznam časů t pro měření, která splňují zadané podmínky."""
    casy = []
    for m in mereni:
        if metoda is not None and m["metoda"] != metoda:
            continue
        if ruka is not None and m["ruka"] != ruka:
            continue
        if doba is not None and m["doba"] != doba:
            continue
        casy.append(m["t"])
    return casy


def statistika(casy):
    """Základní statistické charakteristiky souboru časů."""
    x = np.array(casy)
    return {"n": len(x),
            "prumer": np.mean(x),
            "median": np.median(x),
            "s": np.std(x),                 # směrodatná odchylka (dělíme n, jako v hodině H55)
            "min": np.min(x),
            "max": np.max(x)}


def vypis_skupinu(popis, casy):
    """Vypíše jeden řádek přehledu pro skupinu měření."""
    st = statistika(casy)
    print(f"  {popis:<34} n = {st['n']:>2}   průměr = {cislo(st['prumer'])} s   "
          f"medián = {cislo(st['median'])} s   s = {cislo(st['s'])} s")


def porovnej(popis1, casy1, popis2, casy2):
    """Porovná průměry dvou skupin hrubým pravidlem 2 směrodatných chyb."""
    if len(casy1) < 5 or len(casy2) < 5:
        return False                             # málo dat – porovnání nemá smysl
    a, b = statistika(casy1), statistika(casy2)
    rozdil = a["prumer"] - b["prumer"]
    # Průměr z n měření kolísá méně než jedno měření – zhruba o s / odmocnina(n).
    hranice = 2 * math.sqrt(a["s"] ** 2 / a["n"] + b["s"] ** 2 / b["n"])
    print(f"  {popis1} − {popis2}: rozdíl průměrů {cislo(1000 * rozdil, 1)} ms, "
          f"hranice náhody ±{cislo(1000 * hranice, 1)} ms")
    if abs(rozdil) > hranice:
        print("    → rozdíl je větší než náhodné kolísání: hypotézu data podporují.")
    else:
        print("    → rozdíl je menší než náhodné kolísání: hypotézu data nepotvrzují.")
    return True


def nakresli_graf(mereni, soubor_grafu):
    """Tři grafy: model pravítka, histogram reakčních dob a porovnání skupin."""
    fig, (g1, g2, g3) = plt.subplots(1, 3, figsize=(13, 4.3))

    # 1) Model t = odmocnina(2h/g) a body z měření pravítkem
    h_max = max([35] + [m["h"] + 5 for m in mereni if m["h"] is not None])
    h_osa = np.linspace(0, h_max, 200)
    g1.plot(h_osa, 1000 * np.sqrt(2 * h_osa / 100 / G), color=BARVA_MODELU, lw=2,
            label="model $t = \\sqrt{2h/g}$")
    h_body = [m["h"] for m in mereni if m["h"] is not None]
    t_body = [1000 * m["t"] for m in mereni if m["h"] is not None]
    if h_body:
        g1.plot(h_body, t_body, "o", color=BARVY["pravitko"], ms=7, mec="white", label="měření pravítkem")
    g1.set_xlabel("dráha pádu pravítka $h$ (cm)")
    g1.set_ylabel("reakční doba $t$ (ms)")
    g1.set_title("Stupnice reakčního pravítka")
    g1.legend(loc="lower right")

    # 2) Histogram reakčních dob (sloupce po 10 ms)
    vsechny = [1000 * m["t"] for m in mereni]
    okraje = np.arange(10 * math.floor(min(vsechny) / 10), 10 * math.ceil(max(vsechny) / 10) + 10, 10)
    sady, barvy, popisy = [], [], []
    for metoda in ("pravitko", "program"):
        casy = [1000 * t for t in vyber(mereni, metoda=metoda)]
        if casy:
            sady.append(casy)
            barvy.append(BARVY[metoda])
            popisy.append(nazev(metoda))
    # sloupce obou metod stojí v každém intervalu vedle sebe
    g2.hist(sady, bins=okraje, color=barvy, label=popisy, edgecolor="white")
    casy_p = [1000 * t for t in vyber(mereni, metoda="pravitko")]
    if casy_p:
        g2.axvline(np.mean(casy_p), color="#0b0b0b", lw=1.5, label="průměr (pravítko)")
        g2.axvline(np.median(casy_p), color="#0b0b0b", lw=1.5, ls="--", label="medián (pravítko)")
    g2.set_xlabel("reakční doba $t$ (ms)")
    g2.set_ylabel("počet pokusů")
    g2.set_title("Histogram reakčních dob")
    g2.legend(fontsize=8)

    # 3) Průměr ± směrodatná odchylka pro každou skupinu
    popisky, x = [], 0
    for metoda in ("pravitko", "program"):
        for ruka in ("dominantni", "nedominantni"):
            for doba in ("rano", "vecer"):
                casy = vyber(mereni, metoda, ruka, doba)
                if len(casy) < 2:
                    continue
                st = statistika(casy)
                g3.errorbar(x, 1000 * st["prumer"], yerr=1000 * st["s"], fmt="o", ms=8,
                            color=BARVY.get(metoda, "#2a78d6"), capsize=4, lw=2)
                popisky.append(f"{nazev(metoda)}\n{'D' if ruka == 'dominantni' else 'N'}, {nazev(doba)}")
                x += 1
    g3.set_xticks(range(len(popisky)))
    g3.set_xticklabels(popisky, fontsize=8)
    g3.set_ylabel("reakční doba: průměr ± s (ms)")
    g3.set_xlabel("D = dominantní, N = nedominantní ruka")
    g3.set_title("Porovnání skupin")

    for g in (g1, g2, g3):
        g.grid(alpha=0.3)
        carky_na_osach(g)
    fig.tight_layout()
    fig.savefig(soubor_grafu, dpi=130)


def main():
    soubor_dat = sys.argv[1] if len(sys.argv) > 1 else "vzorova_data.csv"
    soubor_grafu = sys.argv[2] if len(sys.argv) > 2 else "graf.png"
    mereni = nacti_data(soubor_dat)
    if not mereni:
        print("V souboru nejsou žádná měření.")
        return
    print(f"Reakční doba – zpracování souboru {soubor_dat} (počet pokusů: {len(mereni)})")

    print("\n1) Přehled skupin")
    for metoda in ("pravitko", "program"):
        for ruka in ("dominantni", "nedominantni"):
            for doba in ("rano", "vecer"):
                casy = vyber(mereni, metoda, ruka, doba)
                if casy:
                    vypis_skupinu(f"{nazev(metoda)}, {nazev(ruka)}, {nazev(doba)}", casy)
    for metoda in ("pravitko", "program"):
        casy = vyber(mereni, metoda=metoda)
        if casy:
            st = statistika(casy)
            print(f"  {nazev(metoda)} celkem: n = {st['n']}, průměr = {cislo(st['prumer'])} s, "
                  f"medián = {cislo(st['median'])} s, s = {cislo(st['s'])} s, "
                  f"rozpětí {cislo(st['min'])}–{cislo(st['max'])} s, "
                  f"variační koeficient {cislo(100 * st['s'] / st['prumer'], 1)} %")

    h_all = [m["h"] for m in mereni if m["h"] is not None]
    if h_all:
        h_typ = float(np.median(h_all))
        dt = doba_padu(h_typ + DH) - doba_padu(h_typ)
        print("\n2) Přesnost pravítka")
        print(f"  Typická dráha h = {cislo(h_typ, 1)} cm odpovídá t = {cislo(doba_padu(h_typ))} s.")
        print(f"  Chyba čtení {cislo(DH, 1)} cm změní čas jen o {cislo(1000 * dt, 1)} ms – "
              f"rozptyl výsledků způsobuje hlavně člověk, ne pravítko.")

    print("\n3) Hypotézy (rozdíl je průkazný, když je větší než 2 směrodatné chyby)")
    a = porovnej("nedominantní", vyber(mereni, "pravitko", "nedominantni"),
                 "dominantní", vyber(mereni, "pravitko", "dominantni"))
    b = porovnej("ráno", vyber(mereni, "pravitko", doba="rano"),
                 "večer", vyber(mereni, "pravitko", doba="vecer"))
    c = porovnej("program", vyber(mereni, "program", "dominantni"),
                 "pravítko", vyber(mereni, "pravitko", "dominantni"))
    if not (a or b or c):
        print("  Na porovnání je málo dat – ve skupině potřebuješ aspoň 5 pokusů.")

    casy_p = vyber(mereni, metoda="pravitko")          # člověk bez zpoždění techniky
    t_prumer = np.mean(casy_p if casy_p else [m["t"] for m in mereni])
    s_auto = V_AUTA / 3.6 * t_prumer
    print(f"\n4) Doprava: při {V_AUTA} km/h ujede auto za průměrnou reakční dobu "
          f"{cislo(t_prumer)} s dráhu {cislo(s_auto, 1)} m (a to ještě nebrzdí).")

    nakresli_graf(mereni, soubor_grafu)
    print(f"\nGraf je uložen v souboru {soubor_grafu}.")


if __name__ == "__main__":
    main()
