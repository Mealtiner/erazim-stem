"""P13 Barometr v mobilu – tlak a výška: zpracování měření tlaku v patrech domu.

Použití:  python3 zpracovani.py [data.csv] [graf.png]
          (bez parametrů zpracuje vzorova_data.csv a uloží graf.png)

Sloupce CSV (desetinná tečka):
  patro       číslo podlaží (0 = přízemí)
  smer        nahoru / dolu (cesta po schodech nahoru a pak zpět dolů)
  cas_min     čas od začátku měření v minutách
  h_metr_m    výška nad přízemím změřená „metrem“ (počet schodů × výška schodu) v metrech
  p_hPa       tlak z barometru v mobilu (průměr za asi 30 s) v hPa
  teplota_C   teplota vzduchu na schodišti ve °C
"""
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

M = 0.02896       # molární hmotnost vzduchu (kg/mol)
R = 8.314         # molární plynová konstanta (J/(K·mol))
G = 9.81          # tíhové zrychlení (m/s^2)


def cz(x, des=2):
    """Číslo jako text s desetinnou čárkou (pro český výpis)."""
    x = round(float(x), des) + 0.0          # + 0.0 odstraní „zápornou nulu“ -0,0
    return f'{x:.{des}f}'.replace('.', ',')


def osy_s_carkou(ax):
    """Popisky os s desetinnou čárkou (bez posunutí a násobitele u osy)."""
    popis = FuncFormatter(lambda x, pos: f'{x:g}'.replace('.', ','))
    ax.xaxis.set_major_formatter(popis)
    ax.yaxis.set_major_formatter(popis)


def nacti_data(soubor):
    """Načte CSV a vrátí slovník: název sloupce -> numpy pole (sloupec smer zůstane textový)."""
    sloupce = {}
    with open(soubor, encoding='utf-8') as f:
        for radek in csv.DictReader(f):
            if not radek['p_hPa'].strip():
                continue
            for nazev, hodnota in radek.items():
                if nazev == 'smer':
                    hodnota = hodnota.strip().lower().replace('ů', 'u')   # „dolů“ i „dolu“
                else:
                    hodnota = float(hodnota)
                sloupce.setdefault(nazev, []).append(hodnota)
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
    patro, cas, h_metr, p, smer = d['patro'], d['cas_min'], d['h_metr_m'], d['p_hPa'], d['smer']

    print('P13 Barometr v mobilu – výsledky')
    print(f'Počet měření: {len(p)}, podlaží {int(patro.min())} až {int(patro.max())}, doba měření {cz(cas.max() - cas.min(), 1)} min')

    # 1) Změna počasí během měření (drift): porovnáme tlak v přízemí na začátku a na konci
    prizemi = np.where(patro == patro.min())[0]
    if len(prizemi) >= 2 and cas[prizemi[-1]] > cas[prizemi[0]]:
        i1, i2 = prizemi[0], prizemi[-1]
        drift = (p[i2] - p[i1]) / (cas[i2] - cas[i1]) * 60
        print(f'Drift tlaku v přízemí: {cz(drift, 2)} hPa za hodinu (počasí se během měření mění)')

    # 2) Průměr za každé patro (nahoru i dolů) – průměrováním se lineární drift téměř vyruší
    patra = np.unique(patro)
    p_pr = np.array([np.mean(p[patro == n]) for n in patra])
    h_pr = np.array([np.mean(h_metr[patro == n]) for n in patra])

    # 3) Teorie: hustota vzduchu v přízemí ze stavové rovnice ideálního plynu, rho = p*M/(R*T)
    T = np.mean(d['teplota_C']) + 273.15
    rho_teor = p_pr[0] * 100 * M / (R * T)
    print(f'\nTeplota {cz(T - 273.15, 1)} °C, hustota vzduchu v přízemí rho = pM/(RT) = {cz(rho_teor, 3)} kg/m^3,'
          f' spád rho*g = {cz(rho_teor * G / 100, 4)} hPa/m')

    # 4) Lineární regrese p = p0 + k*h (hydrostatický model: k = -rho*g)
    k, p0, R2 = linearni_regrese(h_pr, p_pr)
    rho_exp = -k * 100 / G
    print(f'Regrese p = p0 + k*h: p0 = {cz(p0, 2)} hPa, k = {cz(k, 4)} hPa/m, R^2 = {cz(R2, 4)}')
    print(f'  hustota vzduchu z měření rho = -k/g = {cz(rho_exp, 3)} kg/m^3,'
          f' odchylka od teorie {cz((rho_exp - rho_teor) / rho_teor * 100, 1)} %')

    # 5) Výška z barometru: lineární (hydrostatický) a exponenciální (barometrický) model
    h_lin = (p_pr[0] - p_pr) * 100 / (rho_teor * G)
    H = R * T / (M * G)                                   # výšková škála (m)
    h_exp = H * np.log(p_pr[0] / p_pr)
    print(f'\nVýška z tlaku: lin. (p_přízemí - p)/(rho*g), exp. H ln(p_přízemí/p),'
          f' H = RT/(Mg) = {cz(H, 0)} m')
    print(f'  nejvyšší patro: lineárně {cz(h_lin[-1], 2)} m, exponenciálně {cz(h_exp[-1], 2)} m,'
          f' rozdíl {cz(abs(h_exp[-1] - h_lin[-1]) * 100, 1)} cm → v domě stačí lineární model')
    print('  patro   h metrem   h barometrem   rozdíl')
    for n, hm, hb in zip(patra, h_pr, h_lin):
        print(f'  {int(n):5d} {cz(hm):>9} m {cz(hb):>12} m {cz((hb - hm) * 100, 0):>6} cm')

    # 6) Výška patra: směrnice přímky h(patro) z barometru a z metru
    vp_baro, _, _ = linearni_regrese(patra, h_lin)
    vp_metr, _, _ = linearni_regrese(patra, h_pr)
    print(f'\nVýška patra: barometrem {cz(vp_baro, 2)} m, metrem {cz(vp_metr, 2)} m,'
          f' relativní odchylka {cz((vp_baro - vp_metr) / vp_metr * 100, 1)} %')

    # 7) Pro srovnání jen cesta nahoru (drift se nevyruší)
    nahoru = smer == 'nahoru'
    if np.any(~nahoru) and len(np.unique(patro[nahoru])) >= 3:
        p_n = p[nahoru]
        h_n = (p_n[0] - p_n) * 100 / (rho_teor * G)    # výšky jen z cesty nahoru
        vp_n, _, _ = linearni_regrese(patro[nahoru], h_n)
        print(f'  jen z cesty nahoru by vyšla výška patra {cz(vp_n, 2)} m'
              f' (odchylka {cz((vp_n - vp_metr) / vp_metr * 100, 1)} %) – proto měříme nahoru i dolů')

    # 8) Graf: tlak v čase, tlak podle výšky, rozdíl výšky z barometru a metrem
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for s, znacka in (('nahoru', '^'), ('dolu', 'v')):
        vyber = smer == s
        ax[0].plot(cas[vyber], p[vyber], znacka + '-', label=s)
        ax[1].plot(h_metr[vyber], p[vyber], znacka, alpha=0.6, label=s)
    ax[0].set_xlabel('čas $t$ (min)')
    ax[0].set_ylabel('tlak $p$ (hPa)')
    ax[0].set_title('Tlak během měření')
    hh = np.linspace(0, h_pr.max(), 50)
    ax[1].plot(h_pr, p_pr, 'ko', label='průměr za patro')
    ax[1].plot(hh, p0 + k * hh, 'r-', label=f'regrese: {cz(k, 4)} hPa/m')
    ax[1].set_xlabel('výška $h$ (metrem) (m)')
    ax[1].set_ylabel('tlak $p$ (hPa)')
    ax[1].set_title('Tlak a výška')
    ax[2].bar(patra, (h_lin - h_pr) * 100, color='tab:green', label='výška barometrem − metrem')
    ax[2].axhline(0, color='black', lw=0.8)
    ax[2].set_xlabel('patro')
    ax[2].set_ylabel('rozdíl výšek (cm)')
    ax[2].set_title(f'Výška patra: barometr {cz(vp_baro)} m, metr {cz(vp_metr)} m', fontsize=10)
    for osa in ax:
        osa.grid(True, alpha=0.3)
        osa.legend(fontsize=8)
        osy_s_carkou(osa)
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f'\nGraf uložen do souboru {graf}')


if __name__ == '__main__':
    main()
