"""
STEM projekt P03 – Chladnutí čaje: zpracování měření
====================================================
Porovná chladnutí vody v hrnku bez víčka a s víčkem.

Model (Newtonův zákon ochlazování):  T(t) = To + (T0 - To) * e^(-k*t)
    To = teplota okolí, T0 = počáteční teplota, k = konstanta chladnutí (1/min).
Linearizace:  ln(T - To) = ln(T0 - To) - k*t   → v grafu ln(T - To) proti t je přímka se směrnicí -k.
Simulace po krocích (Eulerova metoda):  T_nová = T - k*(T - To)*dt

Použití:   python3 zpracovani.py [data.csv] [graf.png]
           (bez parametrů načte vzorova_data.csv a uloží graf.png)

Formát CSV: hlavička, pak řádky  čas_min, teplota_bez_víčka, teplota_s_víčkem, teplota_okolí
            (prázdná buňka = v tu minutu se neměřilo; desetinná tečka, oddělovač čárka).
Potřebuje:  Python 3, numpy, matplotlib.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')            # graf jen ukládáme do souboru
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

NAZVY = ['bez víčka', 's víčkem']        # 2. a 3. sloupec souboru
BARVY = ['#C8651B', '#2E6DB4']
CILOVE_TEPLOTY = [60, 40]                # kdy bude čaj pitelný (60 °C) a vlažný (40 °C)
MIN_ROZDIL = 1.0                         # body s T - To < 1 °C do logaritmu nebereme (moc nepřesné)


def cz(x, des=2):
    """Číslo jako text s desetinnou čárkou, např. cz(0.0245, 4) -> '0,0245'."""
    if abs(x) < 0.5 * 10 ** (-des):      # ať se nevypisuje „-0,0“
        x = 0.0
    return f'{x:.{des}f}'.replace('.', ',')


def carka_na_osach(ax):
    """Popisky os s desetinnou čárkou."""
    f = FuncFormatter(lambda v, pos: f'{v:g}'.replace('.', ','))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)


def nacti_csv(soubor):
    """Načte CSV do tabulky čísel; prázdná buňka = nan. Zvládne i středník a desetinnou čárku."""
    with open(soubor, encoding='utf-8-sig') as f:
        radky = [r.strip() for r in f if r.strip()]
    oddelovac = ';' if ';' in radky[0] else ','
    pocet = len(radky[0].split(oddelovac))
    tabulka = []
    for r in radky[1:]:
        bunky = [b.strip().replace(',', '.') for b in r.split(oddelovac)]
        cisla = [float(b) if b else np.nan for b in bunky]
        cisla += [np.nan] * (pocet - len(cisla))
        tabulka.append(cisla[:pocet])
    return np.array(tabulka)


def proloz_exponencialu(t, T, To):
    """Linearizace: přímka ln(T - To) = a + b*t metodou nejmenších čtverců. Vrátí k = -b a T0."""
    rozdil = T - To
    dobre = rozdil > MIN_ROZDIL
    b, a = np.polyfit(t[dobre], np.log(rozdil[dobre]), 1)
    return -b, To + np.exp(a)


def model(t, T0, To, k):
    """Teplota podle Newtonova zákona ochlazování."""
    return To + (T0 - To) * np.exp(-k * t)


def euler(T_start, To, k, dt, t_start, t_konec):
    """Simulace po krocích: z teploty v čase t spočítáme teplotu v čase t + dt."""
    casy = [t_start]
    teploty = [T_start]
    while casy[-1] < t_konec:
        T = teploty[-1]
        teploty.append(T - k * (T - To) * dt)    # o kolik za krok klesne = k * rozdíl * dt
        casy.append(casy[-1] + dt)
    return np.array(casy), np.array(teploty)


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else 'vzorova_data.csv'
    graf = sys.argv[2] if len(sys.argv) > 2 else 'graf.png'
    data = nacti_csv(soubor)
    cas = data[:, 0]
    okoli = data[:, 3] if data.shape[1] > 3 else np.array([np.nan])
    if np.all(np.isnan(okoli)):
        To = 22.0
        print('Pozor: v souboru chybí teplota okolí, počítám s 22 °C.')
    else:
        To = np.nanmean(okoli)
    print(f'Projekt P03 – Chladnutí čaje: soubor {soubor}')
    print(f'Teplota okolí To = {cz(To, 1)} °C (průměr změřených hodnot)\n')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    vysledky = []
    for j, nazev in enumerate(NAZVY):
        T = data[:, 1 + j]
        platne = ~np.isnan(T)
        t, T = cas[platne], T[platne]
        k, T0 = proloz_exponencialu(t, T, To)
        polocas = np.log(2) / k                              # rozdíl teplot klesne na polovinu
        odchylky = T - model(t, T0, To, k)
        vysledky.append((nazev, k, polocas))

        print(f'Hrnek {nazev} ({len(t)} měření, {cz(T[0], 1)} °C → {cz(T[-1], 1)} °C):')
        print(f'  k = {cz(k, 4)} 1/min,  T0 z modelu = {cz(T0, 1)} °C,  poločas chladnutí = {cz(polocas, 1)} min')
        print(f'  největší odchylka měření od modelu: {cz(np.max(np.abs(odchylky)), 1)} °C')
        for cil in CILOVE_TEPLOTY:
            if T0 > cil > To:
                t_cil = np.log((T0 - To) / (cil - To)) / k       # z rovnice cil = To + (T0 - To)e^(-kt)
                print(f'  {cil} °C podle modelu v čase t = {cz(t_cil, 1)} min')
        # simulace po krocích a porovnání s přesným řešením (exponenciálou);
        # startujeme z hodnoty modelu v čase prvního měření, aby byl vidět jen vliv délky kroku
        T_start = model(t[0], T0, To, k)
        for dt in (1, 10):
            ts, Ts = euler(T_start, To, k, dt, t[0], t[-1])
            presne = model(ts, T0, To, k)
            print(f'  Euler, krok {dt} min: největší rozdíl od exponenciály {cz(np.max(np.abs(Ts - presne)), 2)} °C')
        print()

        # graf vlevo: teplota v čase; vpravo: linearizace
        tt = np.linspace(0, max(t), 300)
        ax1.plot(t, T, 'o', ms=4, color=BARVY[j], label=f'{nazev}: měření')
        ax1.plot(tt, model(tt, T0, To, k), '-', color=BARVY[j], label=f'{nazev}: model, k = {cz(k, 4)} 1/min')
        if j == 0:
            ts, Ts = euler(T_start, To, k, 10, t[0], t[-1])
            ax1.plot(ts, Ts, 's--', ms=4, color='#7B4FA0', label='Euler s krokem 10 min')
        rozdil = T - To
        dobre = rozdil > MIN_ROZDIL
        ax2.plot(t[dobre], np.log(rozdil[dobre]), 'o', ms=4, color=BARVY[j], label=f'{nazev}: měření')
        ax2.plot(tt, np.log(T0 - To) - k * tt, '-', color=BARVY[j], label=f'{nazev}: přímka, směrnice −k')

    (n1, k1, p1), (n2, k2, p2) = vysledky
    print(f'Víčko zmenší konstantu chladnutí na {cz(100 * k2 / k1, 0)} % '
          f'(poločas {cz(p1, 1)} min → {cz(p2, 1)} min).')

    ax1.axhline(To, color='gray', ls=':', label=f'teplota okolí {cz(To, 1)} °C')
    ax1.set_xlabel('čas $t$ (min)')
    ax1.set_ylabel('teplota $T$ (°C)')
    ax1.set_title('Chladnutí vody v hrnku')
    ax1.legend(fontsize=8)
    ax2.set_xlabel('čas $t$ (min)')
    ax2.set_ylabel(r'$\ln(T - T_\mathrm{o})$')
    ax2.set_title('Linearizace: logaritmus rozdílu teplot')
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        carka_na_osach(ax)
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f'Graf uložen do souboru {graf}.')


if __name__ == '__main__':
    main()
