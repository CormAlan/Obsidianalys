# Obsidianalys 2
En strukturell analysmotor för din Obsidian-vault – nu med beräkningskärna i **C++** och gränssnitt i **React**.

Verktyget behandlar vaulten som en kunskapsgraf och ger både kvantitativ och strukturell analys:
- Ordstatistik per fil och per mapp
- Intern länkgraf (in-/ut-grad, föräldralösa anteckningar, olösta och tvetydiga länkar)
- PageRank-centralitet
- Mappkoppling: intern länkdensitet och länkar mellan mappar
- Sökning i filnamn, innehåll och rubriker

## Snabbare än v1
Mätt på en vault med 2 073 anteckningar, 51 mappar och 9 487 länkar (varm diskcache):

| | Python (v1) | C++ (v2) |
|---|---:|---:|
| En skanning (ord + graf + PageRank + mappar) | 484 ms | **15 ms** |
| Sökning i innehåll | 44 ms | ~5 ms (direkt medan du skriver) |
| Växla ekvationer/kod | ny skanning | direkt (alla varianter räknas i samma skanning) |
| "Visa alla" med 2 000+ rader | segt (Tk-treeview) | direkt (virtualiserad tabell) |

Resultaten är **identiska** med v1 – verifierat med `tools/parity_check.py` (se nedan).
## Arkitektur
```
┌──────────────────────── obsidianalys (en enda binär) ────────────────────────┐
│  C++20-kärna                         Lokal HTTP-server (127.0.0.1)           │
│  ├─ text.cpp      UTF-8, strippning,  ├─ /api/scan    hela analysen som JSON  │
│  │                ordräkning, länkar, ├─ /api/search  sökning i minnet        │
│  │                rubriker            ├─ /api/note    rubriker, in-/utlänkar  │
│  ├─ analysis.cpp  parallell skanning, ├─ /api/open    öppna fil i editor      │
│  │                CSR-graf, PageRank, ├─ /api/ls      mappväljare             │
│  │                mappstatistik       └─ /            inbäddat React-UI       │
└──────────────────────────────────────────────────────────────────────────────┘
```
- **Ingen regex-motor.** Pythons regexar (`$$…$$`, `$…$`, ```` ```…``` ````, `` `…` ``, `\b\w+\b`, `[[…]]`) är ersatta av linjära skanningar med exakt samma semantik.
- **Unicode som Python.** `\w`, `\s`, `str.lower()` och `str.casefold()` kommer från tabeller som genereras ur Pythons egen Unicode-databas (`core/tools/gen_unicode_tables.py`).
- **Parallellt.** Filerna läses och analyseras på alla kärnor. Grafen lagras i CSR-form (komprimerade grannlistor).
- **Frontend inbäddad.** `npm run build` bäddar in `web/dist` i binären – det blir en fil att distribuera, precis som PyInstaller-exe:n.
## Bygga
Kräver en C++20-kompilator (GCC 11+, Clang 14+ eller MSVC 2022), CMake 3.20+ och Node 20+.
### Linux / macOS
```bash
./build.sh
```
### Windows (PowerShell, Visual Studio 2022)
```powershell
cd web; npm ci; npm run build; cd ..
cmake -S . -B build
cmake --build build --config Release
```
Binären hamnar i `build\Release\obsidianalys.exe`, med ikonen `obkol.ico` och versionsinformation. Den byggs som fönsterprogram (utan konsol); använd `-DOBSIDIANALYS_WINDOWED=OFF` för att få ett konsolfönster.
## Köra
```bash
./build/obsidianalys                 # öppnar gränssnittet i webbläsaren
./build/obsidianalys ~/min/vault     # skannar direkt en viss vault
```
Senast använda vault, Top N, växlar och tema sparas i `~/.config/obsidianalys/config.ini` (Windows: `%APPDATA%\Obsidianalys\config.ini`) och vaulten skannas automatiskt vid start.

Övriga lägen:
```bash
obsidianalys --json VAULT           # hela analysen som JSON (inkl. rubriker och länkar)
obsidianalys --bench VAULT 10       # mät skanningstid
obsidianalys --port 8000 --no-browser --threads 4
```
## Funktioner
### Toppfiler
Ordräkning per fil, Top N eller alla. Sorterbara kolumner (`#`-kolumnen är låst och numreras om), stapeldiagram, dubbelklick/Enter öppnar filen.
### Sök
Sök direkt medan du skriver i **filnamn**, **innehåll** eller **rubriker** (skiftlägesokänsligt). `/` eller `Ctrl+K` fokuserar sökfältet.
### Mappar
Ord per mapp, snitt ord/fil och antal filer. Stapeldiagram och munkdiagram (andel ord). Dubbelklick öppnar mappanalysen.
### Mappanalys
Filerna direkt i en vald mapp, med ord, in- och utlänkar samt mappens interna länkdensitet.
### Länkar
In-grad, ut-grad och PageRank per anteckning. Filtrera på in-länkar, ut-länkar, PageRank eller alla. Stapeldiagram för topp 25 in-länkar och topp 25 PageRank.
### Mappkoppling
Per mapp: antal anteckningar, interna länkar, densitet, utgående och inkommande länkar. Topp 200 mapp → mapp-länkpar och en värmekarta över länkflödet mellan de mest kopplade mapparna.
### Högerpanel
Rubrikerna (H1–H6) för vald fil, länkar hit och härifrån, samt knappar för att öppna filen, öppna den i Obsidian (`obsidian://`), kopiera sökvägen och kopiera alla rubriker.
### Ordräkning
Växlar för att räkna med eller ignorera LaTeX-ekvationer (`$…$`, `$$…$$`) och kod (`` `…` ``, ```` ```…``` ````). Växlarna påverkar bara ordräkningen – länkar i kod och LaTeX ignoreras alltid.
# Matematisk modell
Låt $V$ vara mängden anteckningar och $E \subseteq V \times V$ de riktade kanterna från wikilänkar. Grannmatrisen $A$ definieras som
$$A_{ij} = \begin{cases} 1 & \text{om anteckning } i \text{ länkar till anteckning } j \\ 0 & \text{annars} \end{cases}$$
Multipla kanter och självloopar tillåts inte.
## PageRank
PageRank beräknas med power iteration:
$$p_{k+1} = \alpha P^T p_k + (1-\alpha)\frac{1}{n}\mathbf{1}$$
där $\alpha = 0.85$, $P$ är övergångsmatrisen från utgående länkar och noder utan utgående länkar fördelar sin sannolikhet jämnt. Iterationen stoppar när $\lVert p_{k+1} - p_k \rVert_1 < 10^{-8}$ eller efter 100 iterationer.
## Mappdensitet
För en mapp med $n$ anteckningar är det maximala antalet riktade kanter $n(n-1)$, och
$$\rho = \frac{E_{\text{intern}}}{n(n-1)}$$
- $0$ → helt okopplad
- $1$ → fullständigt sammankopplad riktad graf

Hög densitet antyder ett tätt integrerat konceptområde.
# Regler för länkhantering
### Wikilänkar som stöds
```
[[NoteName]]
[[NoteName|Alias]]
[[NoteName#Heading]]
[[NoteName^BlockID]]
[[mapp/NoteName]]
```
### Ignoreras
- `![[image.png]]` (inbäddningar)
- `[[image.png]]` (bildfiler)
- `[[#Heading]]` (rubrik i samma fil)
- Länkar i kodblock, inline-kod och LaTeX
### Namnupplösning
- Länkar matchas mot filnamn (utan `.md`) i hela vaulten
- Exakt 1 träff → kant skapas
- Ingen exakt träff → skiftlägesokänslig (casefold) träff om den är unik
- 0 träffar → olöst (räknas)
- Fler än 1 träff → tvetydig (ignoreras, räknas)
- Dubblettlänkar i samma fil räknas som en kant
# Tolkning av grafmått
- **In-grad** – hur många anteckningar som refererar hit. Högt värde → grundläggande eller centralt begrepp.
- **Ut-grad** – hur många anteckningar denna refererar till. Högt värde → översikts- eller indexanteckning.
- **PageRank** – strukturell betydelse i hela vaulten: att bli refererad av viktiga anteckningar. I högerpanelen visas det som en multipel av snittet ($p_i \cdot n$).
# Utveckling
```bash
# Terminal 1 – C++-servern i dev-läge (fast port 7331, token "dev")
cmake --build build && ./build/obsidianalys --dev
# Terminal 2 – Vite med hot reload
cd web && npm run dev
```
### Paritetstest mot v1
```bash
python3 tools/parity_check.py ~/Documents/Obsidian/Anteckningar
```
Kör v1:s analysfunktioner (utan Tk) och `obsidianalys --json` på samma vault och jämför ordräkning för alla fyra växelkombinationer, alla kanter, olösta/tvetydiga länkar, PageRank, mappstatistik och rubriker.
### Unicode-tabeller
`core/src/unicode_tables.hpp` genereras med `python3 core/tools/gen_unicode_tables.py`.
## Säkerhet
Servern lyssnar bara på `127.0.0.1`. Den kontrollerar `Host`-headern (skydd mot DNS-rebinding) och varje API-anrop kräver en slumpad token som bara finns i den serverade sidan. Därför kan andra webbplatser i webbläsaren inte anropa API:t. `/api/open` öppnar endast filer från den senaste skanningen, aldrig godtyckliga sökvägar.
# Skillnader mot v1
- Rubriker inuti kodblock (t.ex. `# kommentar` i Python-kod) räknas inte längre som rubriker.
- Mappar visas med relativ sökväg (v1 visade bara mappnamnet, vilket var tvetydigt).
- Nytt: in-/utlänkar i högerpanelen, "Öppna i Obsidian", värmekarta för mappkoppling, mörkt tema och sparade inställningar.
# Begränsningar
- Tvetydiga filnamn ignoreras
- Endast `.md`-filer inkluderas
- Undermappar behandlas separat (ingen rekursiv densitetsgruppering)
# v1 (Python)
Originalversionen finns kvar som `Obsidianalys.py` (Tkinter + matplotlib):
```
pyinstaller --windowed --noconsole --icon=obkol.ico --name Obsidianalys --version-file version_info.txt --splash ObSplash.png --add-data "obkol.ico;." --clean Obsidianalys.py
```
# Licens
För personligt bruk och experiment.
