"""
STEM projekt P02 – Kyvadlo: perioda a délka – zpracování měření
===============================================================
Z doby 10 kmitů pro různé délky kyvadla určí periodu T, tíhové zrychlení g
a statistiku opakovaného měření.

Model matematického kyvadla (malé výchylky):   T = 2*pi*sqrt(l/g)
Linearizace:  T^2 = (4*pi^2/g) * l  – na vodorovné ose l, na svislé T^2 → přímka počátkem.
              Ze směrnice k dostaneme g = 4*pi^2 / k.

Použití:   python3 zpracovani.py [data.csv] [graf.png]
           (bez parametrů načte vzorova_data.csv a uloží graf.png)

Formát CSV: hlavička, pak řádky  délka_v_metrech, doba_10_kmitů_v_sekundách
            (každé měření na zvláštní řádek; opakovaná měření = více řádků se stejnou délkou).
Potřebuje:  Python 3, numpy, matplotlib.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')            # graf jen ukládáme do souboru
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

POCET_KMITU = 10                 # kolik kmitů jsme stopovali
CHYBA_DELKY = 0.005              # odchylka měření délky metrem (m)
G_TABULKOVE = 9.81               # pro porovnání (m/s^2)


def cz(x, des=3):
    """Číslo jako text s desetinnou čárkou, např. cz(9.8123, 2) -> '9,81'."""
    if abs(x) < 0.5 * 10 ** (-des):      # ať se nevypisuje „-0,0“
        x = 0.0
    return f'{x:.{des}f}'.replace('.', ',')


def carka_na_osach(ax):
    """Popisky os s desetinnou čárkou."""
    f = FuncFormatter(lambda v, pos: f'{v:g}'.replace('.', ','))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def nacti_csv(soubor):
    """Načte dva sloupce (délka, čas); zvládne i středník a desetinnou čárku z českého Excelu."""
    delky, casy = [], []
    with open(soubor, encoding='utf-8-sig') as f:
        radky = [r.strip() for r in f if r.strip()]
    oddelovac = ';' if ';' in radky[0] else ','
    for r in radky[1:]:
        bunky = [b.strip().replace(',', '.') for b in r.split(oddelovac)]
        if len(bunky) >= 2 and bunky[0] and bunky[1]:
            delky.append(float(bunky[0]))
            casy.append(float(bunky[1]))
    return np.array(delky), np.array(casy)


def statistika(hodnoty):
    """Vrátí průměr, směrodatnou odchylku s (dělíme n) a průměrnou absolutní odchylku."""
    prumer = np.mean(hodnoty)
    s = np.sqrt(np.mean((hodnoty - prumer) ** 2))
    abs_odch = np.mean(np.abs(hodnoty - prumer))
    return prumer, s, abs_odch


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else 'vzorova_data.csv'
    graf = sys.argv[2] if len(sys.argv) > 2 else 'graf.png'
    l_vse, t_vse = nacti_csv(soubor)
    print(f'Projekt P02 – Kyvadlo: soubor {soubor}, {len(l_vse)} měření doby {POCET_KMITU} kmitů\n')

    # 1) seskupíme měření podle délky a spočítáme průměrnou periodu
    delky = np.unique(np.round(l_vse, 3))
    T, sT, n = [], [], []
    print('  l (m)   počet   průměr t10 (s)   T (s)    T^2 (s^2)   g = 4π²l/T² (m/s^2)')
    for l in delky:
        vyber = t_vse[np.round(l_vse, 3) == l]
        prumer, s, _ = statistika(vyber)
        Ti = prumer / POCET_KMITU
        T.append(Ti)
        sT.append(s / POCET_KMITU)
        n.append(len(vyber))
        gi = 4 * np.pi ** 2 * l / Ti ** 2
        print(f'  {cz(l, 2):>5}   {len(vyber):>5}   {cz(prumer, 2):>13}   {cz(Ti, 3):>6}   {cz(Ti ** 2, 3):>8}   {cz(gi, 2):>8}')
    T, sT, n = np.array(T), np.array(sT), np.array(n)
    T2 = T ** 2

    # 2) linearizace T^2 = k*l a regrese (metoda nejmenších čtverců)
    k = np.sum(delky * T2) / np.sum(delky ** 2)          # přímka počátkem
    g_reg = 4 * np.pi ** 2 / k
    k2, q = np.polyfit(delky, T2, 1)                     # obecná přímka T^2 = k2*l + q
    print(f'\nRegrese přímkou počátkem T^2 = k·l:  k = {cz(k, 3)} s^2/m  =>  g = 4π²/k = {cz(g_reg, 2)} m/s^2')
    print(f'Obecná přímka T^2 = k·l + q:  k = {cz(k2, 3)} s^2/m (g = {cz(4 * np.pi ** 2 / k2, 2)} m/s^2), '
          f'q = {cz(q, 3)} s^2  (odpovídá posunu délky o {cz(100 * q / k2, 1)} cm)')
    print(f'Porovnání s tabulkovou hodnotou {cz(G_TABULKOVE, 2)} m/s^2: rozdíl '
          f'{cz(100 * (g_reg - G_TABULKOVE) / G_TABULKOVE, 1)} %')
    print(f'Předpověď modelu: sekundové kyvadlo (T = 2 s) má mít délku l = gT²/(4π²) = '
          f'{cz(g_reg * 4 / (4 * np.pi ** 2), 3)} m')

    # 3) statistika opakovaného měření pro délku s nejvíce měřeními
    i = int(np.argmax(n))
    l0 = delky[i]
    serie = t_vse[np.round(l_vse, 3) == l0]
    prumer, s, dt = statistika(serie)
    T0, dT = prumer / POCET_KMITU, dt / POCET_KMITU
    g0 = 4 * np.pi ** 2 * l0 / T0 ** 2
    delta_g = CHYBA_DELKY / l0 + 2 * dT / T0            # relativní odchylka nepřímého měření
    print(f'\nOpakované měření pro l = {cz(l0, 2)} m ({len(serie)}×):')
    print(f'  průměr t10 = {cz(prumer, 2)} s, medián {cz(np.median(serie), 2)} s, '
          f'min {cz(serie.min(), 2)} s, max {cz(serie.max(), 2)} s')
    print(f'  směrodatná odchylka s = {cz(s, 3)} s, průměrná absolutní odchylka = {cz(dt, 3)} s')
    print(f'  perioda T = ({cz(T0, 3)} ± {cz(dT, 3)}) s, relativní odchylka {cz(100 * dT / T0, 2)} %')
    print(f'  g = {cz(g0, 2)} m/s^2 ± {cz(100 * delta_g, 1)} %  (tj. ± {cz(g0 * delta_g, 2)} m/s^2)')

    # 4) grafy: T(l) s modelem, T^2(l) s přímkou, rozložení opakovaného měření
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.3))
    ll = np.linspace(0, 1.1 * max(delky), 200)
    ax1.errorbar(delky, T, yerr=sT, fmt='o', color='#C8651B', capsize=3, label='měření (± s)')
    ax1.plot(ll, 2 * np.pi * np.sqrt(ll / g_reg), '-', color='#2E6DB4',
             label=f'model $T = 2\\pi\\sqrt{{l/g}}$, g = {cz(g_reg, 2)} m/s²')
    ax1.set_xlabel('délka kyvadla $l$ (m)')
    ax1.set_ylabel('perioda $T$ (s)')
    ax1.set_title('Perioda roste s odmocninou délky')
    ax1.legend(fontsize=9)

    ax2.plot(delky, T2, 'o', color='#C8651B', label='měření')
    ax2.plot(ll, k * ll, '-', color='#2E6DB4', label=f'přímka $T^2 = k\\,l$, k = {cz(k, 3)} s²/m')
    ax2.set_xlabel('délka kyvadla $l$ (m)')
    ax2.set_ylabel('druhá mocnina periody $T^2$ (s²)')
    ax2.set_title('Linearizace: $T^2$ v závislosti na $l$')
    ax2.legend(fontsize=9)

    poradi = np.arange(1, len(serie) + 1)
    ax3.plot(poradi, serie, 'o', color='#3C8D40', label='jednotlivá měření')
    ax3.axhline(prumer, color='#2E6DB4', label=f'průměr {cz(prumer, 2)} s')
    ax3.axhspan(prumer - s, prumer + s, color='#2E6DB4', alpha=0.15, label=f'průměr ± s (s = {cz(s, 2)} s)')
    ax3.set_xlabel('pořadí měření')
    ax3.set_ylabel(f'doba {POCET_KMITU} kmitů $t_{{10}}$ (s)')
    ax3.set_title(f'Opakované měření, $l$ = {cz(l0, 2)} m')
    rozpeti = serie.max() - serie.min()
    ax3.set_ylim(serie.min() - 0.15 * rozpeti, serie.max() + 0.7 * rozpeti)   # místo nahoře pro legendu
    ax3.legend(fontsize=9, loc='upper center', ncol=2)
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3)
        carka_na_osach(ax)
    for ax in (ax1, ax2):
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f'\nGraf uložen do souboru {graf}.')


if __name__ == '__main__':
    main()
