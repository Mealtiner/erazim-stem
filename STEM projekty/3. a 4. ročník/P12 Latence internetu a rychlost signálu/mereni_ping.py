"""P12 Latence internetu – automatické měření příkazem ping.

Program postupně „pingne“ servery ze seznamu SERVERY, z odpovědí zjistí nejkratší a průměrnou
dobu odezvy (RTT) a ztrátu paketů a výsledky uloží do CSV souboru se stejnou hlavičkou jako
sablona_dat.csv. Nic neinstaluje – používá jen standardní knihovnu Pythonu a systémový příkaz ping
(Windows, macOS i Linux).

Použití:  python3 mereni_ping.py [vystup.csv] [pocet_pingu]
          (výchozí moje_data.csv a 10 pingů na server; celé měření trvá asi 5–6 minut)

Když soubor už existuje, nové řádky se připíšou na konec – můžeš tak změřit zvlášť domácí Wi-Fi
a zvlášť mobilní data (sdílený hotspot) a zpracovat obojí najednou programem zpracovani.py.
Měření kdykoli ukončíš klávesami Ctrl+C; co už bylo změřeno, zůstane uložené.
"""
import sys
import os
import re
import csv
import platform
import subprocess

# Seznam serverů: (adresa, město, zeměpisná šířka °, zeměpisná délka °).
# Souřadnice jsou přibližné souřadnice města, kde server stojí (stačí s přesností na desítky km).
# Vlastní server přidáš dalším řádkem – ale NE velké weby jako google.com: ty mají servery
# v mnoha městech a odpoví ti ten nejbližší, takže bys neznal vzdálenost.
SERVERY = [
    ('www.cvut.cz', 'Praha', 50.103, 14.392),
    ('www.ujep.cz', 'Ústí nad Labem', 50.661, 14.032),
    ('www.tul.cz', 'Liberec', 50.767, 15.056),
    ('www.uhk.cz', 'Hradec Králové', 50.209, 15.833),
    ('www.jcu.cz', 'České Budějovice', 48.975, 14.474),
    ('www.muni.cz', 'Brno', 49.195, 16.607),
    ('www.upol.cz', 'Olomouc', 49.594, 17.251),
    ('www.utb.cz', 'Zlín', 49.227, 17.671),
    ('www.stuba.sk', 'Bratislava', 48.149, 17.108),
    ('www.univie.ac.at', 'Vídeň', 48.208, 16.374),
    ('www.elte.hu', 'Budapešť', 47.498, 19.040),
    ('www.agh.edu.pl', 'Krakov', 50.065, 19.945),
    ('www.uw.edu.pl', 'Varšava', 52.230, 21.012),
    ('fra-de-ping.vultr.com', 'Frankfurt', 50.111, 8.682),
    ('ams-nl-ping.vultr.com', 'Amsterdam', 52.368, 4.904),
    ('par-fr-ping.vultr.com', 'Paříž', 48.857, 2.352),
    ('lon-gb-ping.vultr.com', 'Londýn', 51.507, -0.128),
    ('sto-se-ping.vultr.com', 'Stockholm', 59.329, 18.069),
    ('hel1-speed.hetzner.com', 'Helsinky', 60.170, 24.938),
    ('mad-es-ping.vultr.com', 'Madrid', 40.417, -3.704),
    ('tlv-il-ping.vultr.com', 'Tel Aviv', 32.085, 34.782),
    ('nj-us-ping.vultr.com', 'New Jersey (USA)', 40.555, -74.464),
    ('il-us-ping.vultr.com', 'Chicago', 41.878, -87.630),
    ('tx-us-ping.vultr.com', 'Dallas', 32.777, -96.797),
    ('lax-ca-us-ping.vultr.com', 'Los Angeles', 34.052, -118.244),
    ('bom-in-ping.vultr.com', 'Bombaj', 19.076, 72.878),
    ('jnb-za-ping.vultr.com', 'Johannesburg', -26.204, 28.047),
    ('sgp-ping.vultr.com', 'Singapur', 1.352, 103.820),
    ('sao-br-ping.vultr.com', 'São Paulo', -23.551, -46.633),
    ('hnd-jp-ping.vultr.com', 'Tokio', 35.676, 139.650),
    ('syd-au-ping.vultr.com', 'Sydney', -33.869, 151.209),
]

HLAVICKA = ['pripojeni', 'poskytovatel', 'doma_lat_deg', 'doma_lon_deg', 'server', 'mesto',
            'server_lat_deg', 'server_lon_deg', 'rtt_min_ms', 'rtt_prumer_ms', 'ztraceno_pct']


def ping(adresa, pocet):
    """Spustí systémový ping a vrátí seznam naměřených dob odezvy v ms (prázdný, když server neodpověděl)."""
    if platform.system() == 'Windows':
        prikaz = ['ping', '-n', str(pocet), adresa]
    else:
        prikaz = ['ping', '-c', str(pocet), adresa]
    try:
        vysledek = subprocess.run(prikaz, capture_output=True, timeout=3 * pocet + 10)
        vystup = vysledek.stdout
    except subprocess.TimeoutExpired as chyba:
        vystup = chyba.stdout or b''            # použijeme aspoň to, co ping stihl vypsat
    text = vystup.decode('utf-8', errors='replace')
    casy = []
    for radek in text.splitlines():
        # Řádek s odpovědí obsahuje TTL, např. „... ttl=57 time=21.6 ms“ nebo „... čas=15ms TTL=57“
        if 'ttl' not in radek.lower():
            continue
        nalez = re.search(r'[=<]\s*(\d+(?:[.,]\d+)?)\s*ms', radek)
        if nalez:
            casy.append(float(nalez.group(1).replace(',', '.')))
    return casy


def nacti_cislo(vyzva):
    """Zeptá se na desetinné číslo (přijme tečku i čárku), dokud ho uživatel nezadá správně."""
    while True:
        text = input(vyzva).strip().replace(',', '.')
        try:
            return float(text)
        except ValueError:
            print('  Zadej číslo, např. 50.0755')


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else 'moje_data.csv'
    pocet = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print('Měření latence internetu (ping). Souřadnice domova najdeš např. na mapy.cz nebo')
    print('v mapách v mobilu (podrž prst na místě). Zeměpisná délka na východ od Greenwiche je kladná.')
    lat = nacti_cislo('Zeměpisná šířka domova (°): ')
    lon = nacti_cislo('Zeměpisná délka domova (°): ')
    pripojeni = input('Připojení (wifi / kabel / mobil): ').strip() or 'neuvedeno'
    poskytovatel = input('Poskytovatel internetu (stačí obecně, např. „domácí optika“): ').strip() or 'neuvedeno'

    novy = not os.path.exists(soubor)
    with open(soubor, 'a', newline='', encoding='utf-8') as f:
        zapis = csv.writer(f)
        if novy:
            zapis.writerow(HLAVICKA)
        print(f'\nMěřím {len(SERVERY)} serverů po {pocet} pinzích, výsledky ukládám do {soubor}.')
        try:
            for i, (adresa, mesto, s_lat, s_lon) in enumerate(SERVERY, 1):
                casy = ping(adresa, pocet)
                ztraceno = round(100 * (pocet - len(casy)) / pocet)
                if casy:
                    rtt_min = f'{min(casy):.1f}'
                    rtt_prumer = f'{sum(casy) / len(casy):.1f}'
                    info = f'min {rtt_min} ms, průměr {rtt_prumer} ms'
                else:
                    rtt_min = rtt_prumer = ''          # server neodpověděl (nebo ping blokuje)
                    info = 'bez odpovědi'
                print(f'{i:2d}/{len(SERVERY)} {mesto:18s} {adresa:26s} {info}, ztráta {ztraceno} %')
                zapis.writerow([pripojeni, poskytovatel, lat, lon, adresa, mesto, s_lat, s_lon,
                                rtt_min, rtt_prumer, ztraceno])
                f.flush()                               # průběžně uložit na disk
        except KeyboardInterrupt:
            print('\nMěření přerušeno – dosud změřené řádky jsou uložené.')
    print(f'\nHotovo. Data zpracuješ příkazem: python3 zpracovani.py {soubor} graf.png')


if __name__ == '__main__':
    main()
