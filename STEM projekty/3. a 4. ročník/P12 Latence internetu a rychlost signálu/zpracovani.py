"""P12 Latence internetu a rychlost signálu – zpracování měření z příkazu ping.

Použití:  python3 zpracovani.py [data.csv] [graf.png]
          (bez parametrů zpracuje vzorova_data.csv a uloží graf.png)

Sloupce CSV (desetinná tečka) – stejné, jaké ukládá mereni_ping.py:
  pripojeni, poskytovatel          typ připojení (wifi / kabel / mobil) a poskytovatel
  doma_lat_deg, doma_lon_deg       zeměpisná šířka a délka domova (°)
  server, mesto                    adresa serveru a město, kde stojí
  server_lat_deg, server_lon_deg   zeměpisná šířka a délka serveru (°)
  rtt_min_ms, rtt_prumer_ms        nejkratší a průměrná doba odezvy (ms); prázdné = bez odpovědi
  ztraceno_pct                     ztracené pakety (%)
Každá kombinace poskytovatel + připojení se zpracuje zvlášť (např. Wi-Fi × mobilní data,
nebo data spolužáků s jinými poskytovateli spojená do jednoho souboru).
"""
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

R_ZEME = 6371.0               # poloměr Země (km)
C = 299792.458                # rychlost světla ve vakuu (km/s)
N_VLAKNO = 1.47               # index lomu skla optického vlákna (tabulková hodnota)


def cz(x, des=2):
    """Číslo jako text s desetinnou čárkou (pro český výpis)."""
    x = round(float(x), des) + 0.0          # + 0.0 odstraní „zápornou nulu“ -0,0
    return f'{x:.{des}f}'.replace('.', ',')


def osy_s_carkou(ax):
    """Popisky os s desetinnou čárkou."""
    popis = FuncFormatter(lambda x, pos: f'{x:g}'.replace('.', ','))
    ax.xaxis.set_major_formatter(popis)
    ax.yaxis.set_major_formatter(popis)


def vzdalenost_km(lat1, lon1, lat2, lon2):
    """Vzdálenost dvou míst po povrchu Země (ortodroma), kosinová věta sférické trigonometrie:
    cos(theta) = sin(f1) sin(f2) + cos(f1) cos(f2) cos(l2 - l1),  d = R * theta (theta v radiánech)."""
    f1, f2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    cos_theta = np.sin(f1) * np.sin(f2) + np.cos(f1) * np.cos(f2) * np.cos(dl)
    theta = np.arccos(np.clip(cos_theta, -1, 1))    # clip hlídá zaokrouhlení těsně nad 1
    return R_ZEME * theta


def nacti_data(soubor):
    """Načte CSV; vrátí seznam měření (slovníky) a počet serverů, které neodpověděly."""
    mereni, bez_odpovedi = [], 0
    with open(soubor, encoding='utf-8') as f:
        for radek in csv.DictReader(f):
            if not radek['rtt_min_ms'].strip():
                bez_odpovedi += 1
                continue
            d = vzdalenost_km(float(radek['doma_lat_deg']), float(radek['doma_lon_deg']),
                              float(radek['server_lat_deg']), float(radek['server_lon_deg']))
            mereni.append({'skupina': f'{radek["poskytovatel"]} ({radek["pripojeni"]})',
                           'mesto': radek['mesto'], 'd': d,
                           'rtt': float(radek['rtt_min_ms']), 'prumer': float(radek['rtt_prumer_ms']),
                           'ztraceno': float(radek['ztraceno_pct'] or 0)})
    return mereni, bez_odpovedi


def linearni_regrese(x, y):
    """Proloží body přímkou y = k*x + q (metoda nejmenších čtverců), vrátí k, q a R^2."""
    k, q = np.polyfit(x, y, 1)
    y_model = k * x + q
    R2 = 1 - np.sum((y - y_model) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return k, q, R2


def vyhodnot_skupinu(nazev, data):
    """Regrese RTT = t0 + a*d pro jednu skupinu měření, výpis výsledků; vrátí (d, rtt, a, t0)."""
    d = np.array([m['d'] for m in data])
    rtt = np.array([m['rtt'] for m in data])
    a, t0, R2 = linearni_regrese(d, rtt)          # a v ms/km, t0 v ms
    v_ef = 2 / a * 1000                            # signál letí tam i zpět: 2d = v*(RTT - t0); km/s
    n_ef = C / v_ef
    v_vlakno = C / N_VLAKNO
    kolisani = np.mean([m['prumer'] - m['rtt'] for m in data])
    ztraty = np.mean([m['ztraceno'] for m in data])
    print(f'\n{nazev}: {len(d)} serverů, {cz(d.min(), 0)} až {cz(d.max(), 0)} km')
    print(f'  regrese RTT = t0 + a*d: t0 = {cz(t0, 1)} ms, a = {cz(a * 1000, 2)} ms na 1000 km, R^2 = {cz(R2, 4)}')
    print(f'  efektivní rychlost v = 2/a = {cz(v_ef / 1e5, 2)}·10^8 m/s = {cz(v_ef / C, 2)} c,'
          f' „efektivní index lomu“ c/v = {cz(n_ef, 2)}')
    print(f'  ve vlákně (n = {cz(N_VLAKNO)}) v = {cz(v_vlakno / 1e5, 2)}·10^8 m/s → trasa asi'
          f' {cz(v_vlakno / v_ef, 2)}× delší než ortodroma (+ směrovače)')

    # Rychlost pro jednotlivé vzdálené servery: v = 2d/(RTT - t0); nejvyšší je nejblíž skutečnosti
    nejprimejsi = ''
    daleko = [m for m in data if m['d'] > 1000 and m['rtt'] > t0]
    if daleko:
        rychlosti = [2 * m['d'] / (m['rtt'] - t0) * 1000 for m in daleko]
        i = int(np.argmax(rychlosti))
        nejprimejsi = f', nejpřímější trasa: {daleko[i]["mesto"]} {cz(rychlosti[i] / 1e5, 2)}·10^8 m/s'
    print(f'  kolísání (průměr - min.) {cz(kolisani, 1)} ms, ztráty {cz(ztraty, 1)} %{nejprimejsi}')
    # Kontrola dat: rychleji než světlo ve vlákně po ortodromě to nejde
    for m in data:
        if m['d'] > 300 and m['rtt'] > t0 and 2 * m['d'] / (m['rtt'] - t0) * 1000 > v_vlakno:
            print(f'  POZOR: {m["mesto"]} vychází rychleji než světlo ve vlákně – server je asi blíž, než udává seznam')
    # Největší „objížďky“: body nejvíc nad přímkou
    odchylky = rtt - (t0 + a * d)
    poradi = np.argsort(odchylky)[::-1][:3]
    print('  největší zpoždění navíc oproti přímce: '
          + ', '.join(f'{data[i]["mesto"]} +{cz(odchylky[i], 1)} ms' for i in poradi))
    return d, rtt, a, t0


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else 'vzorova_data.csv'
    graf = sys.argv[2] if len(sys.argv) > 2 else 'graf.png'
    mereni, bez_odpovedi = nacti_data(soubor)

    print('P12 Latence internetu a rychlost signálu – výsledky')
    print(f'Změřených dvojic domov–server: {len(mereni)}, bez odpovědi (vynechány): {bez_odpovedi}')

    # Rozdělení měření do skupin podle poskytovatele a připojení (pořadí jako v souboru)
    skupiny = {}
    for m in mereni:
        skupiny.setdefault(m['skupina'], []).append(m)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    prvni_t0 = None
    for nazev, data in skupiny.items():
        if len(data) < 3:
            print(f'\n{nazev}: jen {len(data)} měření – na regresi je potřeba aspoň 3')
            continue
        d, rtt, a, t0 = vyhodnot_skupinu(nazev, data)
        if prvni_t0 is None:
            prvni_t0 = t0
        for osa in ax:
            body = osa.plot(d, rtt, 'o', ms=4, label=nazev)
            dd = np.linspace(0, d.max() * 1.03, 100)
            osa.plot(dd, t0 + a * dd, '-', color=body[0].get_color(), lw=1)

    # Teoretické přímky pro přímou trasu (ortodromu) od t0 první skupiny
    if prvni_t0 is not None:
        dmax = max(m['d'] for m in mereni) * 1.03
        dd = np.linspace(0, dmax, 100)
        for osa in ax:
            osa.plot(dd, prvni_t0 + 2 * dd / (C / N_VLAKNO) * 1000, '--', color='gray',
                     label=f'přímo ve vlákně (n = {cz(N_VLAKNO)})')
            osa.plot(dd, prvni_t0 + 2 * dd / C * 1000, ':', color='black', label='přímo ve vakuu')

    ax[0].set_title('Všechny servery')
    ax[1].set_title('Detail: Evropa (do 2000 km)')
    ax[1].set_xlim(0, 2000)
    blizko = [m['rtt'] for m in mereni if m['d'] <= 2000]
    if blizko:
        ax[1].set_ylim(0, max(blizko) * 1.15)
    for osa in ax:
        osa.set_xlabel('vzdálenost po povrchu Země $d$ (km)')
        osa.set_ylabel('nejkratší doba odezvy RTT (ms)')
        osa.grid(True, alpha=0.3)
        osy_s_carkou(osa)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(graf, dpi=150)
    print(f'\nGraf uložen do souboru {graf}')


if __name__ == '__main__':
    main()
