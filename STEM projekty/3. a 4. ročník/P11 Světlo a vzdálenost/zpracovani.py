"""P11 Světlo a vzdálenost – zpracování měření osvětlení luxmetrem v mobilu.

Použití:  python3 zpracovani.py [data.csv] [graf.png]
          (bez parametrů zpracuje vzorova_data.csv a uloží graf.png)

Sloupce CSV (desetinná tečka):
  r_m       vzdálenost svítilny od čidla v metrech
  alfa_deg  úhel dopadu ve stupních (0 = svítilna míří kolmo na čidlo)
  E_zap_lx  osvětlení se zapnutou svítilnou (lx)
  E_vyp_lx  osvětlení s vypnutou svítilnou na stejném místě (lx) = okolní světlo
"""
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


def cz(x, des=2):
    """Číslo jako text s desetinnou čárkou (pro český výpis)."""
    x = round(float(x), des) + 0.0          # + 0.0 odstraní „zápornou nulu“ -0,0
    return f'{x:.{des}f}'.replace('.', ',')


def osy_s_carkou(ax, log=False):
    """Popisky os s desetinnou čárkou; u logaritmických os značky 1, 2, 5, 10, 20, 50 …"""
    popis = FuncFormatter(lambda x, pos: f'{x:g}'.replace('.', ','))
    for osa in (ax.xaxis, ax.yaxis):
        if log:
            osa.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            osa.set_minor_formatter(NullFormatter())
        osa.set_major_formatter(popis)


def nacti_data(soubor):
    """Načte CSV a vrátí slovník: název sloupce -> numpy pole čísel."""
    sloupce = {}
    with open(soubor, encoding='utf-8') as f:
        for radek in csv.DictReader(f):
            if not radek['r_m'].strip():          # prázdné řádky přeskočíme
                continue
            for nazev, hodnota in radek.items():
                sloupce.setdefault(nazev, []).append(float(hodnota))
    return {nazev: np.array(hodnoty) for nazev, hodnoty in sloupce.items()}


def linearni_regrese(x, y):
    """Proloží body přímkou y = k*x + q (metoda nejmenších čtverců), vrátí k, q a R^2."""
    k, q = np.polyfit(x, y, 1)
    y_model = k * x + q
    R2 = 1 - np.sum((y - y_model) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return k, q, R2


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else 'vzorova_data.csv'
    graf = sys.argv[2] if len(sys.argv) > 2 else 'graf.png'
    d = nacti_data(soubor)

    # 1) Osvětlení jen od svítilny = se zapnutou minus s vypnutou (odečteme okolní světlo)
    E = d['E_zap_lx'] - d['E_vyp_lx']
    r = d['r_m']
    alfa = d['alfa_deg']

    # 2) Řada „vzdálenost“: jen měření s kolmým dopadem (alfa = 0), seřazená podle r
    kolmo = alfa == 0
    poradi = np.argsort(r[kolmo])
    r_k, E_k = r[kolmo][poradi], E[kolmo][poradi]

    print('P11 Světlo a vzdálenost – výsledky')
    print(f'Počet měření ve vzdálenostech: {len(r_k)} (r = {cz(r_k.min())} až {cz(r_k.max())} m)')
    print(f'Okolní světlo (průměr E_vyp): {cz(np.mean(d["E_vyp_lx"]), 1)} lx')

    # 3) Log-log regrese: E = I / r^n  =>  log E = log I - n * log r  (přímka)
    k, q, R2 = linearni_regrese(np.log10(r_k), np.log10(E_k))
    n = -k
    I_log = 10 ** q                       # osvětlení ve vzdálenosti 1 m = svítivost v cd
    odchylka_n = abs(n - 2) / 2 * 100
    print('\nA) Log-log regrese  log E = log I - n log r')
    print(f'   exponent n = {cz(n, 3)} (teorie 2, relativní odchylka {cz(odchylka_n, 1)} %), R^2 = {cz(R2, 4)}')
    print(f'   svítivost I = {cz(I_log, 1)} cd')

    # 4) Linearizace: 1/sqrt(E) = (r + r0)/sqrt(I)  => přímka v proměnné r
    #    směrnice = 1/sqrt(I), průsečík s osou y = r0/sqrt(I); r0 = posunutí skutečného zdroje
    s, q2, R2b = linearni_regrese(r_k, 1 / np.sqrt(E_k))
    I_lin = 1 / s ** 2
    r0 = q2 / s
    print('\nB) Linearizace  1/sqrt(E) = (r + r0)/sqrt(I)')
    print(f'   svítivost I = {cz(I_lin, 1)} cd, posunutí zdroje r0 = {cz(r0 * 100, 1)} cm, R^2 = {cz(R2b, 4)}')

    # 5) Odchylky měření od modelu E = I/(r + r0)^2
    E_model = I_lin / (r_k + r0) ** 2
    odch = (E_k - E_model) / E_model * 100
    i_max = np.argmax(np.abs(odch))
    print(f'   největší odchylka od modelu: {cz(odch[i_max], 1)} % při r = {cz(r_k[i_max])} m,'
          f' průměrná |odchylka| {cz(np.mean(np.abs(odch)), 1)} %')

    # 6) Volitelná řada „úhel“: měření se stejnou vzdáleností a různým úhlem, model E = E0 cos(alfa)
    sikmo = alfa != 0
    mame_uhly = np.any(sikmo)
    if mame_uhly:
        r_u = r[sikmo][0]
        vyber = np.isclose(r, r_u)
        a_u, E_u = alfa[vyber], E[vyber]
        c = np.cos(np.radians(a_u))
        E0 = np.sum(E_u * c) / np.sum(c ** 2)   # nejlepší E0 pro přímku E = E0*c přes počátek
        print(f'\nC) Úhel dopadu při r = {cz(r_u)} m, model E = E0 cos(alfa), E0 = {cz(E0, 1)} lx')
        for a, e in zip(a_u, E_u):
            m = E0 * np.cos(np.radians(a))
            print(f'   alfa = {a:4.0f}°: E = {cz(e, 1):>6} lx, model {cz(m, 1):>6} lx, odchylka {cz((e - m) / m * 100, 1):>6} %')

    # 7) Graf
    sloupcu = 3 if mame_uhly else 2
    fig, ax = plt.subplots(1, sloupcu, figsize=(4.4 * sloupcu, 4))
    rr = np.linspace(r_k.min() * 0.9, r_k.max() * 1.1, 200)
    ax[0].loglog(r_k, E_k, 'o', label='měření')
    ax[0].loglog(rr, I_log / rr ** n, '-', label=f'regrese: n = {cz(n)}')
    ax[0].loglog(rr, I_log / rr ** 2, '--', label='zákon $1/r^2$')
    ax[0].set_xlabel('vzdálenost $r$ (m)')
    ax[0].set_ylabel('osvětlení $E$ (lx)')
    ax[0].set_title('Log-log graf')
    ax[0].legend()
    ax[0].grid(True, which='both', alpha=0.3)
    osy_s_carkou(ax[0], log=True)

    ax[1].plot(r_k, 1 / np.sqrt(E_k), 'o', label='měření')
    ax[1].plot(rr, s * rr + q2, '-', label=f'přímka, $r_0$ = {cz(r0 * 100, 1)} cm')
    ax[1].set_xlabel('vzdálenost $r$ (m)')
    ax[1].set_ylabel(r'$1/\sqrt{E}$ (lx$^{-1/2}$)')
    ax[1].set_title('Linearizace')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    osy_s_carkou(ax[1])

    if mame_uhly:
        aa = np.linspace(0, 90, 100)
        ax[2].plot(a_u, E_u, 'o', label='měření')
        ax[2].plot(aa, E0 * np.cos(np.radians(aa)), '-', label=r'$E_0 \cos\alpha$')
        ax[2].set_xlabel(r'úhel dopadu $\alpha$ (°)')
        ax[2].set_ylabel('osvětlení $E$ (lx)')
        ax[2].set_title(f'Úhel dopadu (r = {cz(r_u)} m)')
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        osy_s_carkou(ax[2])

    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f'\nGraf uložen do souboru {graf}')


if __name__ == '__main__':
    main()
