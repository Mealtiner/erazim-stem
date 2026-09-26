"""
Měření reakční doby v terminálu (STEM projekt P04, 1. a 2. ročník)

Spuštění:   python3 mereni_reakce.py [soubor.csv]
            (bez parametru zapisuje do souboru moje_data.csv)

Jak to funguje:
  1. Program se zeptá, kterou rukou mačkáš Enter a jestli je ráno, nebo večer.
  2. V každém pokusu počká náhodnou dobu (1,5 až 4,5 s), pak vypíše TEĎ!
     a změří, za jak dlouho stiskneš Enter. Další pokus začne sám.
  3. Když stiskneš Enter dřív, než se TEĎ! objeví, je to předčasný start
     a pokus se opakuje. Stejně tak pokus delší než 1 s (nedával jsi pozor).
  4. Výsledky připíše na konec CSV souboru ve stejném tvaru jako sablona_dat.csv,
     takže je hned zpracuje program zpracovani.py.

Pozor: změřený čas obsahuje i zpoždění klávesnice a obrazovky (desítky milisekund).
Potřebuje jen standardní knihovnu Pythonu.
"""
import sys
import os
import time
import random
import statistics

HLAVICKA = "metoda,ruka,doba_dne,h (cm),t (s)"
POCET = 10                          # výchozí počet platných pokusů
MIN_CEKANI, MAX_CEKANI = 1.5, 4.5   # náhodné čekání před signálem (s) – nedá se odhadnout
PRILIS_RYCHLE = 0.100               # rychleji člověk reagovat nedokáže → stiskl předem
PRILIS_POMALE = 1.0                 # delší pokus je nepozornost, ne reakce
PAUZA = 1.0                         # oddech mezi pokusy (s)


def cislo(x, des=3):
    """Vrátí číslo jako text s desetinnou čárkou."""
    text = f"{x:.{des}f}"
    if float(text) == 0:
        text = text.lstrip("-")           # místo „-0,0“ napíšeme „0,0“
    return text.replace(".", ",")


def zeptej_se(otazka, moznosti):
    """Opakuje otázku, dokud uživatel nezadá jednu z povolených odpovědí."""
    while True:
        odpoved = input(otazka).strip().lower()
        if odpoved in moznosti:
            return moznosti[odpoved]
        print("   Nerozumím, zadej prosím jednu z možností:", ", ".join(moznosti))


def zeptej_se_na_pocet():
    """Zeptá se na počet pokusů; prázdná odpověď znamená výchozí hodnotu."""
    while True:
        odpoved = input(f"Kolik platných pokusů? [Enter = {POCET}]: ").strip()
        if odpoved == "":
            return POCET
        if odpoved.isdigit() and 3 <= int(odpoved) <= 50:
            return int(odpoved)
        print("   Zadej celé číslo od 3 do 50.")


def zahod_stisknute_klavesy():
    """Zahodí klávesy stisknuté navíc, aby se nezapočítaly do dalšího pokusu."""
    try:
        if os.name == "nt":                 # Windows
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getwch()
        else:                               # Linux a macOS
            import termios
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass                                # vstup není terminál – nic nezahazujeme


def jeden_pokus(poradi, pocet):
    """Provede jeden pokus a vrátí naměřený čas v sekundách."""
    print(f"\nPokus {poradi}/{pocet}: připrav se…", flush=True)
    time.sleep(random.uniform(MIN_CEKANI, MAX_CEKANI))
    print("   >>>>>>  TEĎ!  Stiskni Enter  <<<<<<", flush=True)
    start = time.perf_counter()        # přesné stopky počítače
    input()
    return time.perf_counter() - start


def uloz(soubor, ruka, doba, casy):
    """Připíše výsledky na konec CSV souboru (hlavičku zapíše jen do nového souboru)."""
    novy = not os.path.exists(soubor) or os.path.getsize(soubor) == 0
    with open(soubor, "a", encoding="utf-8") as f:
        if novy:
            f.write(HLAVICKA + "\n")
        for t in casy:
            f.write(f"program,{ruka},{doba},,{t:.3f}\n")


def main():
    soubor = sys.argv[1] if len(sys.argv) > 1 else "moje_data.csv"
    print("Měření reakční doby – když se objeví TEĎ!, co nejrychleji stiskni Enter.")
    ruka = zeptej_se("Kterou rukou mačkáš Enter? (d = dominantní, n = nedominantní): ",
                     {"d": "dominantni", "n": "nedominantni"})
    doba = zeptej_se("Je ráno, nebo večer? (r = ráno, v = večer): ",
                     {"r": "rano", "v": "vecer"})
    pocet = zeptej_se_na_pocet()
    input("Až budeš připraven, stiskni Enter a sleduj obrazovku… ")

    casy = []                                   # seznam platných reakčních dob
    while len(casy) < pocet:
        t = jeden_pokus(len(casy) + 1, pocet)
        if t < PRILIS_RYCHLE:
            print("   Předčasný start! Enter jsi stiskl dřív, než se objevilo TEĎ. Opakujeme.")
        elif t > PRILIS_POMALE:
            print(f"   {cislo(t, 2)} s – to je nepozornost, ne reakce. Opakujeme.")
        else:
            casy.append(t)
            print(f"   Reakční doba: {round(1000 * t)} ms")
        time.sleep(PAUZA)
        zahod_stisknute_klavesy()

    uloz(soubor, ruka, doba, casy)
    print("\nVýsledky:", ", ".join(str(round(1000 * t)) for t in casy), "ms")
    print(f"Průměr {cislo(statistics.mean(casy))} s, medián {cislo(statistics.median(casy))} s, "
          f"směrodatná odchylka {cislo(statistics.pstdev(casy))} s, nejlepší {cislo(min(casy))} s.")
    print(f"Uloženo do souboru {soubor}. Zpracuješ ho příkazem: python3 zpracovani.py {soubor}")


if __name__ == "__main__":
    main()
