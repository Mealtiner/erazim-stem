"""
Zvuk láhví – zpracování měření (STEM projekt P06, 1. a 2. ročník)

Spuštění:   python3 zpracovani.py [data.csv] [graf.png] [zaznam.csv]
            (bez parametrů zpracuje vzorova_data.csv a uloží graf.png)

Soubor CSV s měřením má hlavičku   L (cm),f (Hz)
  L (cm)   výška vzduchového sloupce = od hladiny vody k hornímu okraji nádoby
  f (Hz)   frekvence tónu (phyphox, Zvukové spektrum, „Maximální frekvence“)
Desetinná čísla piš s tečkou (12.5), ne s čárkou.

Nepovinný třetí soubor je export surových dat z phyphoxu (Zvukové spektrum →
Exportovat data → Raw data, formát CSV s desetinnou tečkou): sloupce čas a záznam.
Když ho nezadáš, program si vyrobí vzorový signál sám.

Model: vzduchový sloupec je píšťala zavřená na jednom konci (u vody). Na délku
L + ΔL se vejde čtvrtina vlnové délky, takže f = v / (4·(L + ΔL)).
Po převrácení je to přímka:  1/f = (4/v)·L + 4·ΔL/v.
Program potřebuje knihovny numpy a matplotlib.
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")            # graf jen ukládáme do souboru, žádné okno neotvíráme
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

TEPLOTA = 22.0       # teplota vzduchu v místnosti (°C) – přepiš na svou hodnotu
PRUMER = 5.0         # vnitřní průměr hrdla nádoby (cm) – přepiš na svou hodnotu
FS = 48000           # vzorkovací frekvence vzorového signálu (Hz), jako mikrofon mobilu
N_VZORKU = 8192      # počet vzorků pro Fourierovu transformaci (jako v phyphoxu)
MODRA, ORANZOVA, SEDA = "#2a78d6", "#eb6834", "#52514e"


def cislo(x, des=1):
    """Vrátí číslo jako text s desetinnou čárkou, např. 344,6."""
    text = f"{x:.{des}f}"
    if float(text) == 0:
        text = text.lstrip("-")           # místo „-0,0“ napíšeme „0,0“
    return text.replace(".", ",")


def carky_na_osach(graf):
    """Na osách grafu napíše čísla s desetinnou čárkou."""
    format_cz = FuncFormatter(lambda x, pozice: f"{x:g}".replace(".", ","))
    graf.xaxis.set_major_formatter(format_cz)
    graf.yaxis.set_major_formatter(format_cz)


def nacti_mereni(soubor):
    """Načte délky vzduchového sloupce (v metrech) a frekvence (Hz)."""
    delky, frekvence = [], []
    with open(soubor, encoding="utf-8-sig", newline="") as f:
        for radek in csv.DictReader(f):
            if radek["L (cm)"].strip() == "" or radek["f (Hz)"].strip() == "":
                continue
            delky.append(float(radek["L (cm)"]) / 100)      # cm → m
            frekvence.append(float(radek["f (Hz)"]))
    return np.array(delky), np.array(frekvence)


def nacti_zaznam(soubor):
    """Načte export surových dat z phyphoxu: 1. sloupec čas (s), 2. sloupec signál."""
    data = np.genfromtxt(soubor, delimiter=",", skip_header=1)
    cas, signal = data[:, 0], data[:, 1]
    fs = 1 / np.mean(np.diff(cas))                           # vzorkovací frekvence
    return signal[:N_VZORKU], fs


def vzorovy_signal(f0, fs, pocet):
    """Tón uzavřené píšťaly: základní frekvence f0 a slabší liché násobky 3·f0, 5·f0 + šum."""
    t = np.arange(pocet) / fs                                # časy vzorků
    nahoda = np.random.default_rng(1)                        # pevné „náhodné“ číslo → stejný šum
    signal = (1.0 * np.sin(2 * np.pi * f0 * t)
              + 0.30 * np.sin(2 * np.pi * 3 * f0 * t + 0.5)
              + 0.12 * np.sin(2 * np.pi * 5 * f0 * t + 1.0)
              + 0.08 * nahoda.normal(0, 1, pocet))           # šum foukání a místnosti
    return signal


def spektrum(signal, fs):
    """Fourierova transformace (numpy.fft): vrátí frekvence a velikosti složek."""
    velikosti = np.abs(np.fft.rfft(signal)) / len(signal) * 2
    frekvence = np.fft.rfftfreq(len(signal), 1 / fs)
    return frekvence, velikosti


def najdi_vrcholy(frekvence, velikosti, pocet=3):
    """Najde nejvyšší místní maxima spektra (vrcholy) a vrátí jejich frekvence."""
    vrcholy = []
    for i in range(1, len(velikosti) - 1):
        je_maximum = velikosti[i] > velikosti[i - 1] and velikosti[i] >= velikosti[i + 1]
        if je_maximum and velikosti[i] > 0.05 * velikosti.max() and frekvence[i] > 50:
            vrcholy.append((velikosti[i], frekvence[i]))
    vrcholy.sort(reverse=True)                               # od nejsilnějšího
    return sorted(f for _, f in vrcholy[:pocet])


def main():
    soubor_dat = sys.argv[1] if len(sys.argv) > 1 else "vzorova_data.csv"
    soubor_grafu = sys.argv[2] if len(sys.argv) > 2 else "graf.png"
    L, f = nacti_mereni(soubor_dat)
    print(f"Zvuk láhví – zpracování souboru {soubor_dat} ({len(L)} měření)")

    # 1) Lineární regrese 1/f = a·L + b
    a, b = np.polyfit(L, 1 / f, 1)
    v = 4 / a                            # rychlost zvuku (m/s)
    dL = b / a                           # koncová korekce (m)
    v_tab = 331.3 + 0.606 * TEPLOTA      # rychlost zvuku ve vzduchu při dané teplotě
    f_model = v / (4 * (L + dL))
    print("\n1) Linearizace: perioda T = 1/f proti délce L je přímka")
    print(f"  T = a·L + b:  a = {cislo(1000 * a, 3)} ms/m,  b = {cislo(1000 * b, 4)} ms")
    print(f"  Rychlost zvuku z měření v = 4/a = {cislo(v)} m/s")
    print(f"  Tabulková hodnota při {cislo(TEPLOTA)} °C: v = 331,3 + 0,606·t = {cislo(v_tab)} m/s "
          f"(odchylka {cislo(100 * (v - v_tab) / v_tab)} %)")
    print(f"  Koncová korekce ΔL = b/a = {cislo(100 * dL, 2)} cm "
          f"(odhad 0,3 × průměr hrdla = {cislo(0.3 * PRUMER, 2)} cm)")

    print("\n2) Tabulka: vlnová délka λ = v/f = 4·(L + ΔL)")
    print("  L (cm)   f (Hz)   T (ms)   λ (cm)   f model (Hz)   odchylka")
    for i in range(len(L)):
        odch = 100 * (f[i] - f_model[i]) / f_model[i]
        print(f"  {cislo(100 * L[i]):>6}   {f[i]:>6.0f}   {cislo(1000 / f[i], 3):>6}   "
              f"{cislo(100 * v / f[i]):>6}   {f_model[i]:>12.0f}   {cislo(odch):>6} %")
    print(f"  Průměrná absolutní odchylka od modelu: {cislo(np.mean(np.abs(f - f_model) / f_model) * 100)} %")

    # 3) Vzorkování a Fourierova transformace (FFT)
    if len(sys.argv) > 3:
        signal, fs = nacti_zaznam(sys.argv[3])
        f0 = None
        print(f"\n3) FFT záznamu {sys.argv[3]} ({len(signal)} vzorků, fs = {fs:.0f} Hz)")
    else:
        fs = FS
        f0 = f[np.argmin(L)]                                 # nejvyšší tón z měření
        signal = vzorovy_signal(f0, fs, N_VZORKU)
        print(f"\n3) FFT vzorového signálu: tón {f0:.0f} Hz s lichými vyššími harmonickými a šumem")
        print(f"  {N_VZORKU} vzorků při vzorkovací frekvenci {fs} Hz = záznam {cislo(1000 * N_VZORKU / fs)} ms")
    frekvence, velikosti = spektrum(signal, fs)
    vrcholy = najdi_vrcholy(frekvence, velikosti)
    print(f"  Rozlišení spektra fs/N = {cislo(fs / len(signal), 2)} Hz")
    print("  Vrcholy spektra:", ", ".join(f"{x:.0f} Hz" for x in vrcholy))
    if vrcholy:
        print("  Poměr k nejnižšímu vrcholu:", " : ".join(cislo(x / vrcholy[0]) for x in vrcholy),
              " (uzavřená píšťala má jen liché násobky 1 : 3 : 5)")
    if f0 is not None:
        fs_mala = 1.5 * f0                                   # méně než 2·f0 – porušený vzorkovací teorém
        t_mala = np.arange(int(0.2 * fs_mala)) / fs_mala
        fr_m, vel_m = spektrum(np.sin(2 * np.pi * f0 * t_mala), fs_mala)
        print(f"  Aliasing: při vzorkování jen {fs_mala:.0f} Hz (méně než 2·f = {2 * f0:.0f} Hz) "
              f"se tón {f0:.0f} Hz jeví jako {fr_m[np.argmax(vel_m)]:.0f} Hz.")

    # 4) Graf
    fig, ((g1, g2), (g3, g4)) = plt.subplots(2, 2, figsize=(11, 7.6))
    L_osa = np.linspace(0.8 * L.min(), 1.1 * L.max(), 200)
    g1.plot(100 * L, f, "o", color=MODRA, ms=7, mec="white", label="měření")
    g1.plot(100 * L_osa, v / (4 * (L_osa + dL)), "-", color=SEDA, lw=2,
            label="model $f = v / (4(L + \\Delta L))$")
    g1.set_xlabel("výška vzduchového sloupce $L$ (cm)")
    g1.set_ylabel("frekvence $f$ (Hz)")
    g1.set_title("Frekvence tónu – lineární lomená funkce")
    g1.legend()

    L_car = np.linspace(-dL, 1.1 * L.max(), 50)
    g2.plot(100 * L, 1000 / f, "o", color=MODRA, ms=7, mec="white", label="měření")
    g2.plot(100 * L_car, 1000 * (a * L_car + b), "-", color=SEDA, lw=2,
            label=f"přímka: v = {cislo(v)} m/s")
    g2.axvline(0, color="#0b0b0b", lw=0.8)
    g2.plot(-100 * dL, 0, "s", color=ORANZOVA, ms=8, label=f"$-\\Delta L$ = {cislo(-100 * dL, 2)} cm")
    g2.set_xlabel("výška vzduchového sloupce $L$ (cm)")
    g2.set_ylabel("perioda $T = 1/f$ (ms)")
    g2.set_title("Linearizace: $T = aL + b$")
    g2.legend()

    f_graf = f0 if f0 is not None else vrcholy[0]
    pocet = int(3 * fs / f_graf) + 1                         # asi 3 periody
    t_ms = 1000 * np.arange(pocet) / fs
    g3.plot(t_ms, signal[:pocet], "-", color=MODRA, lw=1.2, label="signál")
    g3.plot(t_ms, signal[:pocet], "o", color=MODRA, ms=3, label=f"vzorky ({fs:.0f} za sekundu)")
    g3.plot(t_ms, np.sin(2 * np.pi * f_graf * t_ms / 1000), "--", color=SEDA, lw=1.5,
            label=f"sinusoida, $f$ = {f_graf:.0f} Hz")
    g3.set_xlabel("čas $t$ (ms)")
    g3.set_ylabel("výchylka (relativní jednotky)")
    g3.set_title("Navzorkovaný zvuk (3 periody)")
    g3.set_ylim(-2.4, 1.8)                                   # místo pro legendu dole
    g3.legend(fontsize=8, loc="lower center", ncol=3)

    horni = 6 * f_graf
    vyber = frekvence <= horni
    g4.plot(frekvence[vyber], velikosti[vyber], "-", color=MODRA, lw=1.2)
    for x in vrcholy:
        g4.annotate(f"{x:.0f} Hz", (x, velikosti[np.argmin(np.abs(frekvence - x))]),
                    textcoords="offset points", xytext=(5, 3), fontsize=9, color="#0b0b0b")
    g4.set_xlabel("frekvence $f$ (Hz)")
    g4.set_ylabel("amplituda (relativní jednotky)")
    g4.set_title("Spektrum (FFT) – vrcholy jsou tóny ve zvuku")

    for g in (g1, g2, g3, g4):
        g.grid(alpha=0.3)
        carky_na_osach(g)
    fig.tight_layout()
    fig.savefig(soubor_grafu, dpi=120)
    print(f"\nGraf je uložen v souboru {soubor_grafu}.")


if __name__ == "__main__":
    main()
