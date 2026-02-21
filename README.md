Installation:
```
pyinstaller --windowed --noconsole --icon=obkol.ico --name Obsidianalys --version-file version_info.txt --splash ObSplash.png --add-data "obkol.ico;." --clean Obsidianalys.py
```
### Vad är detta?
Ett grafiskt Python-verktyg för att analysera en Obsidian-vault med hjälp av:
- Ordstatistik
- Aggregering på mappnivå
- Analys av intern länk-graf
- PageRank-centralitet
- Mappkopplingsmått

Detta verktyg behandlar din vault som en strukturerad kunskapsgraf och ger både kvantitativ och strukturell analys av dina anteckningar.
# Översikt
Detta program skannar en Obsidian-vault (endast `.md`-filer) och bygger upp två analyslager:
## 1. Textanalyslager
- Ordräkning per fil
- Ordräkning per mapp
- Antal filer per mapp
- Valfri exkludering av:
  - Kodblock
  - Inline-kod
  - LaTeX-ekvationer
## 2. Grafanalyslager
Din vault tolkas som en **riktad graf**:

- Varje anteckning = nod  
- Varje `[[wikilink]]` = riktad kant  
- Kanter är **unika per källfil** (inga dubbletter räknas)

Grafen används för att beräkna:
- In-grad (hur många anteckningar som länkar till denna)
- Ut-grad (hur många anteckningar denna länkar till)
- PageRank (strukturell betydelse)
- Länkdensitet på mappnivå
- Kopplingsmatris mellan mappar
# Matematisk modell
Låt:
- \( $V$ \) = mängden anteckningar  
- \( $E \subseteq V \times V$ \) = riktade kanter från wikilänkar  

Grannmatrisen \( $A$ \) definieras som:

$$
A_{ij} =
\begin{cases}
1 & \text{om anteckning } i \text{ länkar till anteckning } j \\
0 & \text{annars}
\end{cases}
$$

Multipla kanter tillåts inte.
## PageRank
PageRank beräknas med power iteration:
$$
p_{k+1} = \alpha P^T p_k + (1-\alpha)\frac{1}{n}\mathbf{1}
$$
Där:
- \( $\alpha = 0.85$ \)
- \( $P$ \) är övergångsmatrisen som bygger på utgående länkar
- Noder utan utgående länkar fördelar sannolikhet jämnt

# Regler för länkhantering (Viktigt)
Programmet följer strikta regler:

### Wikilänkar som stöds
```
[[NoteName]]
[[NoteName|Alias]]
[[NoteName#Heading]]
[[NoteName^BlockID]]
```

### Ignoreras
- `![[image.png]]` (inbäddningar)
- `[[image.png]]` (bildfiler)
- Länkar inuti:
  - Kodblock ``` ```
  - Inline-kod `code`
  - LaTeX: `$...$`, `$$...$$`

### Namnupplösning
- Länkar matchas **endast mot filnamn**
- Hela vaulten genomsöks
- Om:
  - Exakt 1 match → kant skapas
  - 0 matchningar → olöst (räknas)
  - >1 matchningar → tvetydig (ignoreras, räknas)

Dubblettlänkar i samma fil räknas som **en enda kant**.

# Funktioner

## Fliken Filer
Visar:
- Ordräkning per fil

Stöd för:
- Sortering
- Top-N-filtrering
- Dubbelklick för att öppna fil

## Fliken Mappar
Visar:
- Ordräkning per mapp
- Antal filer per mapp

## Fliken Länkar
Visar per anteckning:
- In-grad
- Ut-grad
- PageRank

Filtrera efter:
- In-länkar
- Ut-länkar
- PageRank
- Alla

Inkluderar:
- Stapeldiagram (Top In / Top PageRank)
- Öppna fil med dubbelklick

## Fliken Mappkoppling
Visar:
- Antal anteckningar per mapp
- Antal interna länkar
- Intern länkdensitet:
$$
\rho = \frac{E_{\text{internal}}}{n(n-1)}
$$
- Utgående länkar mellan mappar
- Inkommande länkar mellan mappar
Visar även:
- Topp mapp → mapp-länkpar
# Installation
Kräver:
- Python 3.9+
- tkinter (ingår oftast i Python)
- matplotlib

Installera matplotlib vid behov:
```bash
pip install matplotlib
```

# Körning
```bash
python Obsidianalys.py
```

Välj din vault-mapp och tryck **Scan**.

# Tolkning av mappdensitet
Antag att en mapp innehåller \( n \) anteckningar. Maximalt antal möjliga riktade kanter:
$$
n(n-1)
$$
Densitet mäter hur internt sammankopplad mappen är:
- 0.0 → helt okopplad  
- 1.0 → fullständigt sammankopplad riktad graf  
Hög densitet antyder ett tätt integrerat konceptområde.
# Tolkning av grafmått
## In-grad
Hur många anteckningar refererar till denna.

Högt värde → grundläggande eller centralt begrepp.
## Ut-grad
Hur många anteckningar denna refererar till.
Högt värde → översikts- eller indexanteckning.
## PageRank
Strukturell betydelse i hela vaulten.
Fångar:
- Att bli refererad av viktiga anteckningar
- Position i kunskapsflödet
# Arkitekturöversikt
Huvudpipeline:
1. `analyze_vault()` → textstatistik  
2. `build_link_graph()` → adjacenslistor  
3. `compute_folder_link_stats()` → mappmått  
4. `compute_pagerank()` → centralitet  
5. UI-rendering  
Datastrukturer:
- Grannmatris (`list[set[int]]`)
- Namnupplösning via dictionary
- Gles power iteration (inga täta matriser)
# Prestanda
- Grafen är gles (effektiva adjacency sets)
- Ingen tät adjacensmatris används
- PageRank konvergerar inom ≤100 iterationer
- Skanning sker i bakgrundstråd

Fungerar väl för vaultar med tusentals anteckningar.
# Begränsningar
- Tvetydiga filnamn ignoreras
- Skiftlägeskänslighet kan påverka matchning
- Endast `.md`-filer inkluderas
- Undermappar behandlas separat (ingen rekursiv densitetsgruppering)
# Framtida utbyggnader
Möjliga uppgraderingar:

- Betweenness-centralitet (brygganteckningar)
- Starkt sammanhängande komponenter
- Community detection (Louvain-klustring)
- Grafexport (GraphML / CSV)
- Tidsutveckling av grafen
- Visualisering av adjacensmatrisens spektrum

# Konceptuella användningsområden
Detta verktyg låter dig analysera:

- Kunskapsfragmentering
- Centrala begrepp
- Strukturella blinda fläckar
- Mapp-modularitet
- Övercentralisering
- Kunskapsöar

Det förvandlar din vault till ett mätbart kunskapsnätverk.
# Licens
För personligt bruk och experiment.
# Slutord
Detta är inte bara en ordräknare.  
Det är en strukturell analysmotor för ditt kunskapssystem.