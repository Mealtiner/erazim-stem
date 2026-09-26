"""
STEM projekt P01 – Volný pád a mobil: zpracování měření
========================================================
Z doby pádu t z různých výšek h určí tíhové zrychlení g.

Model volného pádu z klidu:   h = 1/2 * g * t^2
Linearizace:  když na vodorovnou osu dáme t^2 (místo t), body leží na přímce h = k * t^2,
              která prochází počátkem. Ze směrnice přímky dostaneme g = 2 * k.

Použití:   python3 zpracovani.py [data.csv] [graf.png]
           (bez parametrů načte vzorova_data.csv a uloží graf.png)

Formát CSV: první řádek je hlavička, pak řádky  výška_v_metrech, čas1, čas2, čas3 …
            (desetinná tečka, oddělovač čárka; zvládne i český Excel se středníkem a čárkou).
Potřebuje:  Python 3, numpy, matplotlib.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')            # graf jen ukládáme do souboru, nic se neotevírá
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

G_TABULKOVE = 9.81               # tíhové zrychlení v ČR (m/s^2) – pro porovnání
MEZ_HRUBE_CHYBY = 0.10           # čas, který se od mediánu řádku liší o víc než 10 %, považujeme za omyl


def cz(x, des=3):
    """Číslo jako text s desetinnou čárkou (česky), např. cz(9.8123, 2) -> '9,81'."""
    if abs(x) < 0.5 * 10 ** (-des):      # ať se nevypisuje „-0,0“
        x = 0.0
    return f'{x:.{des}f}'.replace('.', ',')


def carka_na_osach(ax):
    """Popisky os s desetinnou čárkou (0,5 místo 0.5)."""
    f = FuncFormatter(lambda v, pos: f'{v:g}'.replace('.', ','))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def nacti_csv(soubor):
    """Načte CSV soubor a vrátí seznam řádků s čísly (prázdná buňka = nan)."""
    with open(soubor, encoding='utf-8-sig') as f:
        radky = [r.strip() for r in f if r.strip()]
    oddelovac = ';' if ';' in radky[0] else ','          # český Excel ukládá se středníkem
    pocet_sloupcu = len(radky[0].split(oddelovac))
    tabulka = []
    for r in radky[1:]:
        bunky = r.split(oddelovac)
        cisla = []
        for b in bunky:
            b = b.strip().replace(',', '.')              # desetinná čárka -> tečka
            cisla.append(float(b) if b else np.nan)
        while len(cisla) < pocet_sloupcu:                # doplníme chybějící buňky na konci řádku
            cisla.append(np.nan)
        tabulka.append(cisla)
    return np.array(tabulka)


def vyrad_hrube_chyby(h, casy):
    """V každém řádku vyřadí čas, který se od mediánu řádku liší o víc než MEZ_HRUBE_CHYBY."""
    casy = casy.copy()
    for i in range(len(h)):
        median = np.nanmedian(casy[i])
        for j in range(casy.shape[1]):
            c = casy[i, j]
            if not np.isnan(c) and abs(c - median) > MEZ_HRUBE_CHYBY * median:
                print(f'  ! Výška {cz(h[i], 2)} m: vyřazuji čas {cz(c)} s '
                      f'(liší se od ostatních o víc než {100 * MEZ_HRUBE_CHYBY:.0f} %) – pravděpodobně hrubá chyba.')
                casy[i, j] = np.nan
    return casy


def primka_pocatkem(x, y):
    """Metoda nejmenších čtverců pro přímku y = k * x (prochází počátkem): k = součet(x*y) / součet(x^2)."""
    return np.sum(x * y) / np.sum(x * x)


def obecna_primka(x, y):
    """Metoda nejmenších čtverců pro přímku y = k * x + q; vrátí (k, q)."""
    k, q = np.polyfit(x, y, 1)
    return k, q


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else 'vzorova_data.csv'
    graf = sys.argv[2] if len(sys.argv) > 2 else 'graf.png'

    data = nacti_csv(soubor)
    h = data[:, 0]                       # výšky (m)
    casy = data[:, 1:]                   # jednotlivé pokusy (s)
    print(f'Projekt P01 – Volný pád: soubor {soubor}, {len(h)} výšek\n')

    # 1) kontrola dat a průměrný čas pro každou výšku
    casy = vyrad_hrube_chyby(h, casy)
    t = np.nanmean(casy, axis=1)                  # průměrná doba pádu
    t2 = t ** 2
    g_i = 2 * h / t2                              # g z každé výšky zvlášť: g = 2h / t^2

    print('\n  h (m)   průměr t (s)   t^2 (s^2)   g = 2h/t^2 (m/s^2)')
    for i in range(len(h)):
        print(f'  {cz(h[i], 2):>5}   {cz(t[i]):>10}   {cz(t2[i], 4):>9}   {cz(g_i[i], 2):>8}')

    # 2) statistika hodnot g_i: průměr, absolutní a relativní odchylka
    g_prumer = np.mean(g_i)
    dg = np.mean(np.abs(g_i - g_prumer))          # průměrná absolutní odchylka
    print(f'\nPrůměr z jednotlivých výšek: g = ({cz(g_prumer, 2)} ± {cz(dg, 2)}) m/s^2, '
          f'relativní odchylka {cz(100 * dg / g_prumer, 1)} %')

    # 3) linearizace h = k * t^2 a regrese
    k = primka_pocatkem(t2, h)
    g_reg = 2 * k
    k2, q = obecna_primka(t2, h)
    zbytky = h - k * t2                           # odchylky bodů od modelu (m)
    print(f'\nRegrese přímkou počátkem h = k·t^2:  k = {cz(k, 3)} m/s^2  =>  g = 2k = {cz(g_reg, 2)} m/s^2')
    print(f'Obecná přímka h = k·t^2 + q:  k = {cz(k2, 3)} m/s^2 (g = {cz(2 * k2, 2)} m/s^2), '
          f'q = {cz(100 * q, 1)} cm')
    print(f'Největší odchylka bodu od modelu: {cz(100 * np.max(np.abs(zbytky)), 1)} cm')
    rozdil = 100 * (g_reg - G_TABULKOVE) / G_TABULKOVE
    print(f'Porovnání s tabulkovou hodnotou {cz(G_TABULKOVE, 2)} m/s^2: rozdíl {cz(rozdil, 1)} %')
    if abs(q) > 0.02:
        print('  Pozor: q se výrazně liší od nuly – hledej soustavnou chybu (výška, zpoždění zvuku, start stopek).')

    # 4) graf: vlevo h(t) s parabolou, vpravo h(t^2) s přímkou
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    tt = np.linspace(0, 1.1 * max(t), 200)
    ax1.plot(t, h, 'o', color='#C8651B', label='měření')
    ax1.plot(tt, 0.5 * g_reg * tt ** 2, '-', color='#2E6DB4',
             label=f'model $h = \\frac{{1}}{{2}} g t^2$, g = {cz(g_reg, 2)} m/s²')
    ax1.set_xlabel('doba pádu $t$ (s)')
    ax1.set_ylabel('výška $h$ (m)')
    ax1.set_title('Závislost výšky na čase – parabola')
    ax1.grid(alpha=0.3)
    ax1.legend()

    xx = np.linspace(0, 1.1 * max(t2), 50)
    ax2.plot(t2, h, 'o', color='#C8651B', label='měření')
    ax2.plot(xx, k * xx, '-', color='#2E6DB4', label=f'přímka $h = k t^2$, k = {cz(k, 3)} m/s²')
    ax2.set_xlabel('druhá mocnina času $t^2$ (s²)')
    ax2.set_ylabel('výška $h$ (m)')
    ax2.set_title('Linearizace: $h$ v závislosti na $t^2$')
    ax2.grid(alpha=0.3)
    ax2.legend()
    for ax in (ax1, ax2):
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        carka_na_osach(ax)
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f'\nGraf uložen do souboru {graf}.')


if __name__ == '__main__':
    main()
