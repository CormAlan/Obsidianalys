#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Obsidian Summary GUI (Tkinter)
------------------------------
Grafisk sammanfattning av en Obsidian-vault:
- Välj rotmapp
- Skanna .md-filer
- Visa totalord, totalfiler, toppfiler, mapplista
- Ny tab: Mappanalys (toppfiler inom vald mapp)
- Sortera på kolumnrubriker (låst '#' kolumn), renumerera rank
- Default-sort: Toppfiler Ord↓, Mappar Totalt ord↓
- 'Visa alla' <-> 'Visa Top N' toggle (minns tidigare N)
- Sök: Filnamn / Innehåll / Rubriker
- Sidopanel: rubriker (H1–H6) för vald fil
- Dubbelklick öppnar fil i standardeditor
- Lila tema för hela appen
- Grafer (om matplotlib finns)
- NYTT: Växlar för att räkna med/ignorera ekvationer ($...$, $$...$$) och kod (`, ```)
"""

from __future__ import annotations

import json, os
import re
import sys
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# För grafer (valfritt)
_MPL_OK = True
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    _MPL_OK = False


# ===================== Hjälplogik =====================

# --- Mönster ---
# Latex
LATEX_BLOCK  = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)  # $$ ... $$
LATEX_INLINE = re.compile(r"\$(.*?)\$",     flags=re.DOTALL)  # $  ... $
# Kod
FENCED_CODE  = re.compile(r"```.*?```",     flags=re.DOTALL)  # ``` ... ``` (språk valfritt)
INLINE_CODE  = re.compile(r"`[^`]*`")                              # ` ... ` (ingen nestning)
# Ord
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
HEADING_RE   = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

# Wikilänkar (Obsidian)
# - (?<!!)... ignorerar embeds ![[...]]
WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\[\]]+?)\]\]")
_WS_RE      = re.compile(r"\s+")
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tif", ".tiff", ".ico", ".heic"}

APP_NAME = "Obsidianalys"
CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
CONFIG_PATH = CONFIG_DIR / "config.json"

def resource_path(relative: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / relative)
    return str(Path(__file__).resolve().parent / relative)

def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def strip_equations(text: str) -> str:
    """Tar bort LaTeX-ekvationer ($$...$$ och $...$) ur texten."""
    text = re.sub(LATEX_BLOCK, "", text)
    text = re.sub(LATEX_INLINE, "", text)
    return text

def strip_code(text: str) -> str:
    """Tar bort kodblock (```...```) och inline-kod (`...`) ur texten."""
    # Viktigt: ta bort fenced först, annars 'äter' inline regex delar i fenced
    text = re.sub(FENCED_CODE, "", text)
    text = re.sub(INLINE_CODE, "", text)
    return text

def count_words_in_markdown(path: Path, *, include_equations: bool = False, include_code: bool = False) -> int:
    """
    Räknar ord i en markdownfil.
    - include_equations=False -> ignorerar LaTeX ($...$, $$...$$)
    - include_code=False      -> ignorerar kod (inline `...` och fenced ```...```)
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    if not include_equations:
        text = strip_equations(text)
    if not include_code:
        text = strip_code(text)

    return len(re.findall(WORD_PATTERN, text))

def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")

def extract_headings(path: Path) -> List[Tuple[int, str]]:
    """Returnerar lista av (nivå, rubriktext) för #-rubriker i filen."""
    headings: List[Tuple[int, str]] = []
    try:
        for line in read_text_safe(path).splitlines():
            m = HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                headings.append((level, text))
    except Exception:
        pass
    return headings

def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False




@dataclass
class LinkGraph:
    # Node-index
    id_to_path: List[Path]                 # id -> filpath
    path_to_id: Dict[Path, int]            # filpath -> id
    name_to_ids: Dict[str, List[int]]      # "NoteName" -> [id, ...] (kan vara flera vid namnkrock)

    # Graf (riktad, enkel: inga multikanter)
    out_neighbors: List[set[int]]          # out_neighbors[u] = {v, ...}
    in_degree: List[int]
    out_degree: List[int]

    # Diagnostik
    unresolved_links: int                  # targets som inte matchar någon fil (räknas per källa-fil, unikt per fil)
    ambiguous_links: int                   # targets som matchar >1 fil (räknas per källa-fil, unikt per fil)
    total_unique_edges: int

    # Linjär-algebra-mått (valfritt)
    pagerank: Optional[List[float]] = None


@dataclass
class FolderLinkStats:
    folder_to_ids: Dict[Path, List[int]]           # folder -> node ids som ligger direkt i foldern
    internal_edges: Dict[Path, int]                # folder -> antal edges u->v där u,v i folder
    density: Dict[Path, float]                     # folder -> E_in / (n(n-1)) (0 om n<2)
    outbound_edges: Dict[Path, int]                # folder -> edges från folder -> utanför
    inbound_edges: Dict[Path, int]                 # folder -> edges från utanför -> folder
    folder_matrix: Dict[Tuple[Path, Path], int]    # (from_folder, to_folder) -> antal edges

@dataclass
class ScanResult:
    total_words: int
    file_word_counts: Dict[Path, int]
    dir_word_counts: Dict[Path, int]
    dir_file_counts: Dict[Path, int]
    md_dirs: List[Path]
    leaf_dirs: List[Path]
    # NYTT: länk-grafanalys (kan vara None om inga .md-filer)
    link_graph: Optional[LinkGraph] = None
    folder_link_stats: Optional[FolderLinkStats] = None

def analyze_vault(root: Path, *, include_equations: bool, include_code: bool) -> ScanResult:
    file_word_counts: Dict[Path, int] = {}
    dir_word_counts: Dict[Path, int] = defaultdict(int)
    dir_file_counts: Dict[Path, int] = defaultdict(int)
    md_dirs: set[Path] = set()

    for md_file in root.rglob("*.md"):
        words = count_words_in_markdown(md_file, include_equations=include_equations, include_code=include_code)
        file_word_counts[md_file] = words
        dir_word_counts[md_file.parent] += words
        dir_file_counts[md_file.parent] += 1
        md_dirs.add(md_file.parent)

    if not file_word_counts:
        return ScanResult(
            total_words=0,
            file_word_counts={},
            dir_word_counts={},
            dir_file_counts={},
            md_dirs=[],
            leaf_dirs=[],
        )

    total_words = sum(file_word_counts.values())

    md_dirs_list = list(md_dirs)
    # Inkludera ALLA mappar som direkt innehåller minst en .md
    leaf_dirs = md_dirs_list

    return ScanResult(
        total_words=total_words,
        file_word_counts=file_word_counts,
        dir_word_counts=dict(dir_word_counts),
        dir_file_counts=dict(dir_file_counts),
        md_dirs=md_dirs_list,
        leaf_dirs=leaf_dirs,
    )




# ===================== Länk-grafanalys =====================

def _normalize_note_name(name: str) -> str:
    """Normalisera namn (för Obsidian-note-resolve) utan att bli för aggressiv."""
    name = name.strip()
    name = _WS_RE.sub(" ", name)
    return name


def extract_unique_wikilink_targets(
    text: str,
    *,
    ignore_code: bool = True,
    ignore_equations: bool = True
) -> set[str]:
    """
    Extraherar unika wikilänk-targets ur texten:
    - Räknar endast [[...]] (ej ![[...]])
    - Hanterar [[target|alias]], [[target#heading]], [[target^block]]
    - Tar bort .md-suffix och folder-prefix (folder/note -> note)
    - Ignorerar länkar i kod/LaTeX om flaggor True
    """
    if ignore_equations:
        text = strip_equations(text)
    if ignore_code:
        text = strip_code(text)

    targets: set[str] = set()
    for m in WIKILINK_RE.finditer(text):
        inner = m.group(1).strip()
        if not inner:
            continue

        # Alias
        target = inner.split("|", 1)[0].strip()
        if not target:
            continue

        # [[#heading]] (rubrik i samma fil) -> hoppa över (ingen note-target)
        if target.startswith("#"):
            continue

        # Heading / blockref
        target = target.split("#", 1)[0].split("^", 1)[0].strip()
        if not target:
            continue

        # .md-suffix
        if target.lower().endswith(".md"):
            target = target[:-3]

        # Folder-prefix: folder/note -> note
        if "/" in target:
            target = target.split("/")[-1].strip()

        # Ignorera bildlänkar (attachments)
        low = target.lower().strip()
        if any(low.endswith(ext) for ext in IMAGE_EXTS):
            continue

        target = _normalize_note_name(target)
        if target:
            targets.add(target)

    return targets


def _build_vault_index(md_files: List[Path]) -> Tuple[List[Path], Dict[Path, int], Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Bygger index:
    - id_to_path, path_to_id
    - name_to_ids (exakt), name_cf_to_ids (casefold fallback)
    """
    md_files_sorted = sorted(md_files)
    id_to_path = md_files_sorted
    path_to_id: Dict[Path, int] = {p: i for i, p in enumerate(id_to_path)}

    name_to_ids: Dict[str, List[int]] = defaultdict(list)
    name_cf_to_ids: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(id_to_path):
        name = _normalize_note_name(p.stem)
        name_to_ids[name].append(i)
        name_cf_to_ids[name.casefold()].append(i)

    return id_to_path, path_to_id, dict(name_to_ids), dict(name_cf_to_ids)


def _resolve_target_to_id(target: str, name_to_ids: Dict[str, List[int]], name_cf_to_ids: Dict[str, List[int]]) -> Tuple[Optional[int], str]:
    """
    Resolve policy:
    - exakt match på normaliserat namn
    - annars casefold-match (endast om unik)
    Returnerar (id, status) där status ∈ {"ok","unresolved","ambiguous"}.
    """
    t = _normalize_note_name(target)
    ids = name_to_ids.get(t, [])
    if len(ids) == 1:
        return ids[0], "ok"
    if len(ids) > 1:
        return None, "ambiguous"

    # fallback: casefold
    ids_cf = name_cf_to_ids.get(t.casefold(), [])
    if len(ids_cf) == 1:
        return ids_cf[0], "ok"
    if len(ids_cf) > 1:
        return None, "ambiguous"

    return None, "unresolved"


def build_link_graph(
    md_files: List[Path],
    *,
    ignore_code: bool = True,
    ignore_equations: bool = True
) -> LinkGraph:
    """
    Bygger en riktad, enkel graf av wikilänkar mellan .md-filer.
    Regler:
    - Endast [[...]] (ej ![[...]])
    - Multipla länkar mellan samma två filer räknas en gång
    - Länkar i kod/LaTeX ignoreras
    - Resolve via note-namn (filnamn utan .md) globalt i vault
    - Vid namnkrock: ambiguous -> ingen edge
    """
    id_to_path, path_to_id, name_to_ids, name_cf_to_ids = _build_vault_index(md_files)
    n = len(id_to_path)
    out_neighbors: List[set[int]] = [set() for _ in range(n)]

    unresolved = 0
    ambiguous = 0

    for u, path in enumerate(id_to_path):
        try:
            txt = read_text_safe(path)
        except Exception:
            continue

        targets = extract_unique_wikilink_targets(
            txt,
            ignore_code=ignore_code,
            ignore_equations=ignore_equations
        )

        for t in targets:
            vid, status = _resolve_target_to_id(t, name_to_ids, name_cf_to_ids)
            if status == "ok" and vid is not None:
                if vid != u:  # ignorera self-loop
                    out_neighbors[u].add(vid)
            elif status == "ambiguous":
                ambiguous += 1
            elif status == "unresolved":
                unresolved += 1

    out_degree = [len(s) for s in out_neighbors]
    in_degree = [0] * n
    total_edges = 0
    for u in range(n):
        total_edges += len(out_neighbors[u])
        for v in out_neighbors[u]:
            in_degree[v] += 1

    return LinkGraph(
        id_to_path=id_to_path,
        path_to_id=path_to_id,
        name_to_ids=name_to_ids,
        out_neighbors=out_neighbors,
        in_degree=in_degree,
        out_degree=out_degree,
        unresolved_links=unresolved,
        ambiguous_links=ambiguous,
        total_unique_edges=total_edges,
        pagerank=None
    )


def compute_pagerank(
    graph: LinkGraph,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-8
) -> List[float]:
    """PageRank via power iteration på gles graf."""
    n = len(graph.id_to_path)
    if n == 0:
        return []

    p = [1.0 / n] * n
    out_deg = graph.out_degree
    out_nbr = graph.out_neighbors

    for _ in range(max_iter):
        new_p = [(1.0 - alpha) / n] * n

        dangling_sum = 0.0
        for u in range(n):
            if out_deg[u] == 0:
                dangling_sum += p[u]
        if dangling_sum:
            add = alpha * dangling_sum / n
            for i in range(n):
                new_p[i] += add

        for u in range(n):
            d = out_deg[u]
            if d == 0:
                continue
            share = alpha * p[u] / d
            for v in out_nbr[u]:
                new_p[v] += share

        diff = sum(abs(new_p[i] - p[i]) for i in range(n))
        p = new_p
        if diff < tol:
            break

    s = sum(p)
    if s > 0:
        p = [x / s for x in p]
    return p


def compute_folder_link_stats(graph: LinkGraph) -> FolderLinkStats:
    """Aggregerar grafen på mappnivå (direkt-folder för varje fil)."""
    folder_to_ids: Dict[Path, List[int]] = defaultdict(list)
    folders: List[Path] = []

    for i, p in enumerate(graph.id_to_path):
        f = p.parent
        folder_to_ids[f].append(i)
        folders.append(f)

    internal_edges: Dict[Path, int] = defaultdict(int)
    outbound_edges: Dict[Path, int] = defaultdict(int)
    inbound_edges: Dict[Path, int] = defaultdict(int)
    folder_matrix: Dict[Tuple[Path, Path], int] = defaultdict(int)

    folder_of = [p.parent for p in graph.id_to_path]
    for u, nbrs in enumerate(graph.out_neighbors):
        fu = folder_of[u]
        for v in nbrs:
            fv = folder_of[v]
            folder_matrix[(fu, fv)] += 1
            if fu == fv:
                internal_edges[fu] += 1
            else:
                outbound_edges[fu] += 1
                inbound_edges[fv] += 1

    density: Dict[Path, float] = {}
    for folder, ids in folder_to_ids.items():
        n = len(ids)
        if n < 2:
            density[folder] = 0.0
        else:
            density[folder] = internal_edges.get(folder, 0) / (n * (n - 1))

    return FolderLinkStats(
        folder_to_ids=dict(folder_to_ids),
        internal_edges=dict(internal_edges),
        density=density,
        outbound_edges=dict(outbound_edges),
        inbound_edges=dict(inbound_edges),
        folder_matrix=dict(folder_matrix),
    )

# ===================== Tkinter-app =====================

class ObsidianSummaryApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master.title("Obsidian – Grafisk sammanfattning")
        self.master.geometry("1320x840")
        self.pack(fill="both", expand=True)

        # State
        self.root_dir_var = tk.StringVar(value=str(
            Path.home() / "OneDrive/Documents/Utbildning och skolarbete/Obsidian anteckningar/Anteckningar"
        ))
        self.top_n_var = tk.IntVar(value=30)
        self.status_var = tk.StringVar(value="Redo.")
        self.search_var = tk.StringVar(value="")
        self.search_mode_var = tk.StringVar(value="Filnamn")  # Filnamn | Innehåll | Rubriker

        # NYTT: växlar
        self.include_equations_var = tk.BooleanVar(value=False)  # standard: ignorera ekvationer (som tidigare)
        self.include_code_var      = tk.BooleanVar(value=False)  # standard: ignorera kod

        self.result: Optional[ScanResult] = None
        self.last_root: Optional[Path] = None

        # Toggle-state för "Visa alla"
        self.showing_all: bool = False
        self.prev_top_n: Optional[int] = None
        self.show_all_btn: Optional[ttk.Button] = None

        # Mappningar för att hitta Path från val i tabeller
        self._file_iid_to_path: Dict[str, Path] = {}
        self._folder_iid_to_path: Dict[str, Path] = {}
        self._dir_iid_to_path: Dict[str, Path] = {}

        # NYTT: länk-graf-tabbar
        self.links_mode_var = tk.StringVar(value="In-länkar")  # In-länkar | Ut-länkar | PageRank | Alla
        self._link_iid_to_path: Dict[str, Path] = {}


        self._build_ui()
        try:
            import pyi_splash
            pyi_splash.close()
        except Exception:
            pass

    def _build_ui(self):
        # Paned – vänster (huvud) + höger (rubriker)
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned, width=380)
        paned.add(left, weight=3)
        paned.add(right, weight=1)

        # --- Topbar
        topbar = ttk.Frame(left)
        topbar.pack(side="top", fill="x", padx=12, pady=(10, 6))

        ttk.Label(topbar, text="Rotmapp:").pack(side="left")
        ttk.Entry(topbar, textvariable=self.root_dir_var, width=70).pack(side="left", padx=6)
        ttk.Button(topbar, text="Välj…", command=self.on_browse).pack(side="left", padx=4)

        ttk.Label(topbar, text="Top N filer:").pack(side="left", padx=(16, 4))
        ttk.Spinbox(topbar, from_=1, to=200000, textvariable=self.top_n_var, width=8).pack(side="left")

        self.show_all_btn = ttk.Button(topbar, text="Visa alla", command=self.on_show_all_clicked)
        self.show_all_btn.pack(side="left", padx=6)

        ttk.Button(topbar, text="Skanna", command=self.on_scan_clicked).pack(side="left", padx=8)

        # --- Växlare för räkning
        toggles = ttk.Frame(left)
        toggles.pack(side="top", fill="x", padx=12, pady=(0, 8))

        eq_cb = ttk.Checkbutton(toggles, text="Räkna ekvationer ($, $$)", variable=self.include_equations_var)
        code_cb = ttk.Checkbutton(toggles, text="Räkna kodblock (`, ```)", variable=self.include_code_var)
        eq_cb.pack(side="left")
        ttk.Label(toggles, text="  ").pack(side="left")
        code_cb.pack(side="left")

        # --- Search bar
        searchbar = ttk.Frame(left)
        searchbar.pack(side="top", fill="x", padx=12, pady=(0, 8))

        ttk.Label(searchbar, text="Sök:").pack(side="left")
        search_entry = ttk.Entry(searchbar, textvariable=self.search_var, width=40)
        search_entry.pack(side="left", padx=6)
        search_entry.bind("<Return>", lambda e: self.on_search_clicked())

        ttk.Label(searchbar, text="i").pack(side="left", padx=(8, 4))
        self.search_mode = ttk.Combobox(searchbar, textvariable=self.search_mode_var,
                                        values=["Filnamn", "Innehåll", "Rubriker"],
                                        state="readonly", width=12)
        self.search_mode.pack(side="left")

        ttk.Button(searchbar, text="Sök", command=self.on_search_clicked).pack(side="left", padx=6)
        ttk.Button(searchbar, text="Rensa", command=self.on_clear_search).pack(side="left")

        # --- Summary bar
        summary = ttk.Frame(left)
        summary.pack(side="top", fill="x", padx=12, pady=(0, 10))

        self.total_words_label = ttk.Label(summary, text="Totalt antal ord: –", font=("Segoe UI", 11, "bold"))
        self.total_words_label.pack(side="left")

        ttk.Label(summary, text="   |   ").pack(side="left")
        self.total_files_label = ttk.Label(summary, text="Totalt antal filer: –", font=("Segoe UI", 11, "bold"))
        self.total_files_label.pack(side="left")

        # --- Notebook (tabs)
        self.nb = ttk.Notebook(left)
        self.nb.pack(side="top", fill="both", expand=True, padx=12, pady=6)

        # Tab 1: Toppfiler
        self.tab_files = ttk.Frame(self.nb)
        self.nb.add(self.tab_files, text="Toppfiler")

        self.files_tree = self._make_tree(self.tab_files, columns=("rank", "path", "words"))
        self.files_tree.heading("rank", text="#")  # låst
        self.files_tree.heading("path", text="Fil (relativ)",
                                command=lambda: self.treeview_sort_column(self.files_tree, "path", False))
        self.files_tree.heading("words", text="Ord",
                                command=lambda: self.treeview_sort_column(self.files_tree, "words", False))
        self.files_tree.bind("<Double-1>", self.on_open_selected_file)
        self.files_tree.bind("<<TreeviewSelect>>", self.on_files_selection_changed)

        files_btns = ttk.Frame(self.tab_files)
        files_btns.pack(side="bottom", fill="x", padx=6, pady=8)
        ttk.Button(files_btns, text="Visa stapeldiagram (toppfiler)", command=self.on_plot_top_files).pack(side="left")
        ttk.Button(files_btns, text="Kopiera vald fils sökväg", command=self.on_copy_selected_file).pack(side="left", padx=6)
        ttk.Button(files_btns, text="Öppna fil", command=lambda: self.on_open_selected_file(None)).pack(side="left", padx=6)

        # Tab 2: Mappar
        self.tab_dirs = ttk.Frame(self.nb)
        self.nb.add(self.tab_dirs, text="Mappar (med filer)")

        self.dirs_tree = self._make_tree(self.tab_dirs, columns=("rank", "dir", "total", "avg", "count"))
        self.dirs_tree.heading("rank", text="#")
        self.dirs_tree.heading("dir", text="Mappnamn",
                               command=lambda: self.treeview_sort_column(self.dirs_tree, "dir", False))
        self.dirs_tree.heading("total", text="Totalt ord",
                               command=lambda: self.treeview_sort_column(self.dirs_tree, "total", False))
        self.dirs_tree.heading("avg", text="Snitt ord/fil",
                               command=lambda: self.treeview_sort_column(self.dirs_tree, "avg", False))
        self.dirs_tree.heading("count", text="Antal filer",
                               command=lambda: self.treeview_sort_column(self.dirs_tree, "count", False))

        dirs_btns = ttk.Frame(self.tab_dirs)
        dirs_btns.pack(side="bottom", fill="x", padx=6, pady=8)
        ttk.Button(dirs_btns, text="Visa stapeldiagram (mappar)", command=self.on_plot_top_dirs).pack(side="left")
        ttk.Button(dirs_btns, text="Visa sektordiagram (mappar)", command=self.on_plot_pie_dirs).pack(side="left", padx=6)
        ttk.Button(dirs_btns, text="Analysera mapp", command=self.on_analyze_selected_dir).pack(side="left", padx=12)

        # Tab 3: Mappanalys
        self.tab_folder = ttk.Frame(self.nb)
        self.nb.add(self.tab_folder, text="Mappanalys")

        header_folder = ttk.Frame(self.tab_folder)
        header_folder.pack(side="top", fill="x", padx=6, pady=(6, 0))
        ttk.Label(header_folder, text="Vald mapp:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.folder_header_label = ttk.Label(header_folder, text="—")
        self.folder_header_label.pack(side="left", padx=6)

        self.folder_tree = self._make_tree(self.tab_folder, columns=("rank", "path", "words"))
        self.folder_tree.heading("rank", text="#")
        self.folder_tree.heading("path", text="Fil (i mappen)",
                                 command=lambda: self.treeview_sort_column(self.folder_tree, "path", False))
        self.folder_tree.heading("words", text="Ord",
                                 command=lambda: self.treeview_sort_column(self.folder_tree, "words", False))
        self.folder_tree.bind("<Double-1>", self.on_open_selected_folder_file)
        self.folder_tree.bind("<<TreeviewSelect>>", self.on_folder_selection_changed)

        folder_btns = ttk.Frame(self.tab_folder)
        folder_btns.pack(side="bottom", fill="x", padx=6, pady=8)
        ttk.Button(folder_btns, text="Kopiera filsökväg", command=self.on_copy_selected_folder_file).pack(side="left")
        ttk.Button(folder_btns, text="Öppna fil", command=lambda: self.on_open_selected_folder_file(None)).pack(side="left", padx=6)

        # Tab 4: Länkar (anteckningar)
        self.tab_links = ttk.Frame(self.nb)
        self.nb.add(self.tab_links, text="Länkar")

        links_header = ttk.Frame(self.tab_links)
        links_header.pack(side="top", fill="x", padx=6, pady=(6, 0))
        ttk.Label(links_header, text="Länkgraf – sammanfattning:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.links_summary_label = ttk.Label(links_header, text="—")
        self.links_summary_label.pack(side="left", padx=6)

        links_controls = ttk.Frame(self.tab_links)
        links_controls.pack(side="top", fill="x", padx=6, pady=(6, 0))
        ttk.Label(links_controls, text="Visa:").pack(side="left")
        self.links_mode = ttk.Combobox(
            links_controls,
            textvariable=self.links_mode_var,
            values=["In-länkar", "Ut-länkar", "PageRank", "Alla"],
            state="readonly",
            width=12
        )
        self.links_mode.pack(side="left", padx=6)
        self.links_mode.bind("<<ComboboxSelected>>", lambda e: self.on_links_mode_changed())
        ttk.Label(links_controls, text="(Top N styrs av 'Top N filer' ovan, om inte 'Alla')", foreground="#555").pack(side="left", padx=6)

        self.links_tree = self._make_tree(self.tab_links, columns=("rank", "path", "in", "out", "pr"))
        self.links_tree.heading("rank", text="#")
        self.links_tree.heading("path", text="Fil (relativ)",
                        command=lambda: self.treeview_sort_column(self.links_tree, "path", False))
        self.links_tree.heading("in", text="In",
                        command=lambda: self.treeview_sort_column(self.links_tree, "in", True))
        self.links_tree.heading("out", text="Ut",
                        command=lambda: self.treeview_sort_column(self.links_tree, "out", True))
        self.links_tree.heading("pr", text="PageRank",
                        command=lambda: self.treeview_sort_column(self.links_tree, "pr", True))
        self.links_tree.bind("<Double-1>", self.on_open_selected_link_file)
        self.links_tree.bind("<<TreeviewSelect>>", self.on_links_selection_changed)

        links_btns = ttk.Frame(self.tab_links)
        links_btns.pack(side="bottom", fill="x", padx=6, pady=8)
        ttk.Button(links_btns, text="Visa stapeldiagram (Top In)", command=self.on_plot_top_in_links).pack(side="left")
        ttk.Button(links_btns, text="Visa stapeldiagram (Top PageRank)", command=self.on_plot_top_pagerank).pack(side="left", padx=6)
        ttk.Button(links_btns, text="Kopiera vald fils sökväg", command=self.on_copy_selected_link_file).pack(side="left", padx=12)
        ttk.Button(links_btns, text="Öppna fil", command=lambda: self.on_open_selected_link_file(None)).pack(side="left", padx=6)

        # Tab 5: Mappkoppling (länkar)
        self.tab_link_dirs = ttk.Frame(self.nb)
        self.nb.add(self.tab_link_dirs, text="Mappkoppling")

        link_dirs_header = ttk.Frame(self.tab_link_dirs)
        link_dirs_header.pack(side="top", fill="x", padx=6, pady=(6, 0))
        ttk.Label(link_dirs_header, text="Mappnivå – sammanfattning:", font=("Segoe UI", 10, "bold")).pack(side="left")
        self.link_dirs_summary_label = ttk.Label(link_dirs_header, text="—")
        self.link_dirs_summary_label.pack(side="left", padx=6)

        # Dela upp tabben i två delar: mappstatistik + topp mapp->mapp länkar
        link_paned = ttk.Panedwindow(self.tab_link_dirs, orient="vertical")
        link_paned.pack(side="top", fill="both", expand=True, padx=6, pady=6)

        link_upper = ttk.Frame(link_paned)
        link_lower = ttk.Frame(link_paned)
        link_paned.add(link_upper, weight=2)
        link_paned.add(link_lower, weight=1)

        ttk.Label(link_upper, text="Mappar (direkt innehåll av .md):", font=("Segoe UI", 10, "bold")).pack(side="top", anchor="w", padx=6, pady=(2, 0))
        self.link_dirs_tree = self._make_tree(link_upper, columns=("rank", "dir", "notes", "internal", "density", "outbound", "inbound"))
        self.link_dirs_tree.heading("rank", text="#")
        self.link_dirs_tree.heading("dir", text="Mapp (relativ)",
                            command=lambda: self.treeview_sort_column(self.link_dirs_tree, "dir", False))
        self.link_dirs_tree.heading("notes", text="Anteckn.",
                            command=lambda: self.treeview_sort_column(self.link_dirs_tree, "notes", True))
        self.link_dirs_tree.heading("internal", text="Interna länkar",
                            command=lambda: self.treeview_sort_column(self.link_dirs_tree, "internal", True))
        self.link_dirs_tree.heading("density", text="Densitet",
                            command=lambda: self.treeview_sort_column(self.link_dirs_tree, "density", True))
        self.link_dirs_tree.heading("outbound", text="Ut",
                            command=lambda: self.treeview_sort_column(self.link_dirs_tree, "outbound", True))
        self.link_dirs_tree.heading("inbound", text="In",
                            command=lambda: self.treeview_sort_column(self.link_dirs_tree, "inbound", True))

        ttk.Label(link_lower, text="Topp mapp → mapp-länkar:", font=("Segoe UI", 10, "bold")).pack(side="top", anchor="w", padx=6, pady=(2, 0))
        self.link_pairs_tree = self._make_tree(link_lower, columns=("rank", "from", "to", "edges"))
        self.link_pairs_tree.heading("rank", text="#")
        self.link_pairs_tree.heading("from", text="Från",
                             command=lambda: self.treeview_sort_column(self.link_pairs_tree, "from", False))
        self.link_pairs_tree.heading("to", text="Till",
                           command=lambda: self.treeview_sort_column(self.link_pairs_tree, "to", False))
        self.link_pairs_tree.heading("edges", text="Länkar",
                             command=lambda: self.treeview_sort_column(self.link_pairs_tree, "edges", True))



        # --- Höger: Rubriker
        right_header = ttk.Frame(right)
        right_header.pack(side="top", fill="x", padx=10, pady=(10, 4))
        ttk.Label(right_header, text="Rubriker (vald fil)", font=("Segoe UI", 11, "bold")).pack(side="left")
        self.heading_path_label = ttk.Label(right_header, text="—", foreground="#555")
        self.heading_path_label.pack(side="left", padx=8)

        right_buttons = ttk.Frame(right)
        right_buttons.pack(side="top", fill="x", padx=10, pady=(0, 6))
        ttk.Button(right_buttons, text="Öppna fil", command=self.on_open_selected_any_file).pack(side="left")
        ttk.Button(right_buttons, text="Kopiera alla rubriker", command=self.on_copy_headings).pack(side="left", padx=6)

        self.headings_tree = self._make_tree(right, columns=("level", "heading"))
        self.headings_tree.heading("level", text="Nivå")
        self.headings_tree.heading("heading", text="Rubrik")
        self.headings_tree.column("level", width=60, anchor="e")
        self.headings_tree.column("heading", width=260, anchor="w")

        # --- Statusbar
        statusbar = ttk.Frame(self)
        statusbar.pack(side="bottom", fill="x")

        self.status_label = ttk.Label(statusbar, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side="left", padx=8, pady=6)

        if not _MPL_OK:
            warn = ttk.Label(
                statusbar,
                text="(Tips: Installera matplotlib för grafer: pip install matplotlib)",
                foreground="#8a2f2f"
            )
            warn.pack(side="right", padx=8)

    def _make_tree(self, parent: tk.Widget, columns: Tuple[str, ...]) -> ttk.Treeview:
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, padx=6, pady=6)

        tree = ttk.Treeview(container, columns=columns, show="headings", height=18)
        tree.pack(side="left", fill="both", expand=True)

        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        vsb.pack(side="right", fill="y")
        tree.configure(yscrollcommand=vsb.set)

        for col in columns:
            if col == "rank":
                tree.column(col, width=60, anchor="e")
            elif col in ("words", "total", "avg", "count", "in", "out", "pr", "notes", "internal", "density", "outbound", "inbound", "edges"):
                tree.column(col, width=120, anchor="e")
            elif col in ("path", "heading"):
                tree.column(col, width=700 if col == "path" else 280, anchor="w")
            elif col == "dir":
                tree.column(col, width=420, anchor="w")
            elif col in ("from", "to"):
                tree.column(col, width=460, anchor="w")
            else:
                tree.column(col, width=120, anchor="w")

        return tree

    # ---------------- Actions ----------------

    def on_browse(self):
        initial = Path(self.root_dir_var.get()).expanduser()
        selected = filedialog.askdirectory(
            initialdir=str(initial if initial.exists() else Path.home()),
            title="Välj rotmapp för Obsidian-anteckningar"
        )
        if selected:
            self.root_dir_var.set(selected)

    def on_scan_clicked(self):
        root = Path(self.root_dir_var.get()).expanduser()
        if not root.exists() or not root.is_dir():
            messagebox.showerror("Fel", "Ogiltig rotmapp.")
            return
        self.status_var.set(self._status_text_before_scan())
        self._set_controls_state(tk.DISABLED)
        threading.Thread(target=self._scan_thread, args=(root,
                                                         self.include_equations_var.get(),
                                                         self.include_code_var.get()), daemon=True).start()

    def _status_text_before_scan(self) -> str:
        parts = ["Skannar…"]
        parts.append("ekvationer: " + ("med" if self.include_equations_var.get() else "ignoreras"))
        parts.append("kod: " + ("med" if self.include_code_var.get() else "ignoreras"))
        return " | ".join(parts)

    def on_show_all_clicked(self):
        if not self.result or not self.last_root:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return
        total_files = len(self.result.file_word_counts)
        if not self.showing_all:
            self.prev_top_n = int(self.top_n_var.get())
            self.top_n_var.set(total_files)
            self._populate_top_files(self.last_root, self.result)
            self.nb.select(self.tab_files)
            self.status_var.set(f"Visar alla filer ({total_files}).")
            self.showing_all = True
            if self.show_all_btn:
                self.show_all_btn.config(text="Visa Top N")
        else:
            restored = self.prev_top_n if self.prev_top_n and self.prev_top_n > 0 else 30
            self.top_n_var.set(restored)
            self._populate_top_files(self.last_root, self.result)
            self.nb.select(self.tab_files)
            self.status_var.set(f"Visar Top {restored}.")
            self.showing_all = False
            if self.show_all_btn:
                self.show_all_btn.config(text="Visa alla")

    def on_search_clicked(self):
        if not self.result or not self.last_root:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return
        q = self.search_var.get().strip()
        if not q:
            messagebox.showinfo("Sök", "Skriv något att söka efter.")
            return
        mode = self.search_mode_var.get()
        q_low = q.lower()
        matches: Dict[Path, int] = {}
        if mode == "Filnamn":
            for p, words in self.result.file_word_counts.items():
                rel = str(p.relative_to(self.last_root)).lower()
                if q_low in rel:
                    matches[p] = words
        elif mode == "Innehåll":
            for p, words in self.result.file_word_counts.items():
                try:
                    txt = read_text_safe(p).lower()
                except Exception:
                    continue
                if q_low in txt:
                    matches[p] = words
        elif mode == "Rubriker":
            for p, words in self.result.file_word_counts.items():
                hs = extract_headings(p)
                if any(q_low in h[1].lower() for h in hs):
                    matches[p] = words
        if not matches:
            messagebox.showinfo("Sök", f"Inga träffar för “{q}” ({mode}).")
            return
        self._populate_top_files_from_map(self.last_root, matches)
        self.nb.select(self.tab_files)
        self.status_var.set(f"Hittade {len(matches)} träff(ar) för “{q}” ({mode}).")

    def on_clear_search(self):
        self.search_var.set("")
        if self.result and self.last_root:
            self._populate_top_files(self.last_root, self.result)
            self.nb.select(self.tab_files)
            self.status_var.set("Sökningen rensad.")

    def _scan_thread(self, root: Path, include_equations: bool, include_code: bool):
        try:
            result = analyze_vault(root, include_equations=include_equations, include_code=include_code)
    
            # NYTT: Bygg länk-grafen (ignorerar alltid kod/LaTeX för länkar)
            if result.file_word_counts:
                md_files = list(result.file_word_counts.keys())
                graph = build_link_graph(md_files, ignore_code=True, ignore_equations=True)
    
                # PageRank är bara meningsfullt om det finns kanter
                if graph.total_unique_edges > 0 and len(graph.id_to_path) > 0:
                    graph.pagerank = compute_pagerank(graph)
                else:
                    graph.pagerank = [0.0] * len(graph.id_to_path)
    
                result.link_graph = graph
                result.folder_link_stats = compute_folder_link_stats(graph)
    
        except Exception as e:
            self.master.after(0, lambda: self._on_scan_error(e))
            return
        self.master.after(0, lambda: self._on_scan_done(root, result))
    
    def _on_scan_error(self, err: Exception):
        self.status_var.set("Fel vid skanning.")
        self._set_controls_state(tk.NORMAL)
        messagebox.showerror("Skanningsfel", f"Något gick fel:\n{err!r}")

    def _on_scan_done(self, root: Path, result: ScanResult):
        self.result = result
        self.last_root = root

        if result.total_words == 0:
            self.total_words_label.config(text="Totalt antal ord: 0 (inga .md-filer hittades)")
            self.total_files_label.config(text="Totalt antal filer: 0")
            self._clear_trees()
            self.status_var.set("Klar (inga .md-filer hittades).")
            self._set_controls_state(tk.NORMAL)
            self.showing_all = False
            self.prev_top_n = None
            if self.show_all_btn:
                self.show_all_btn.config(text="Visa alla")
            self.heading_path_label.config(text="—")
            self._populate_headings_panel(None)
            return

        self.total_words_label.config(text=f"Totalt antal ord: {result.total_words:,}".replace(",", " "))
        self.total_files_label.config(text=f"Totalt antal filer: {len(result.file_word_counts):,}".replace(",", " "))

        self._populate_top_files(root, result)
        self._populate_leaf_dirs(result)
        self._populate_links(root, result)
        self._populate_link_dirs(root, result)

        self.showing_all = False
        self.prev_top_n = None
        if self.show_all_btn:
            self.show_all_btn.config(text="Visa alla")

        self.heading_path_label.config(text="—")
        self._populate_headings_panel(None)

        # Reflektera inställningar i status
        # Reflektera inställningar i status
        settings = (
            f"ekvationer: {'med' if self.include_equations_var.get() else 'ignoreras'}, "
            f"kod: {'med' if self.include_code_var.get() else 'ignoreras'}"
        )
        self.status_var.set(f"Klar. ({settings})")
        self._set_controls_state(tk.NORMAL)


    def _clear_trees(self):
        for t in (self.files_tree, self.dirs_tree, self.headings_tree, self.folder_tree, getattr(self, 'links_tree', None), getattr(self, 'link_dirs_tree', None), getattr(self, 'link_pairs_tree', None)):
            if t is None:
                continue
            for row in t.get_children():
                t.delete(row)
        self._file_iid_to_path.clear()
        self._dir_iid_to_path.clear()
        self._folder_iid_to_path.clear()
        self._link_iid_to_path.clear()
        self.folder_header_label.config(text="—")
        if hasattr(self, 'links_summary_label'):
            self.links_summary_label.config(text="—")
        if hasattr(self, 'link_dirs_summary_label'):
            self.link_dirs_summary_label.config(text="—")

    def _populate_top_files(self, root: Path, result: ScanResult):
        self._file_iid_to_path.clear()
        for row in self.files_tree.get_children():
            self.files_tree.delete(row)

        n = max(1, int(self.top_n_var.get()))
        items = Counter(result.file_word_counts).most_common(n)
        for idx, (path, words) in enumerate(items, start=1):
            rel = str(path.relative_to(root))
            iid = self.files_tree.insert("", "end", values=(idx, rel, words))
            self._file_iid_to_path[iid] = path

        self.treeview_sort_column(self.files_tree, "words", True)
        self.files_tree.yview_moveto(0)

    def _populate_top_files_from_map(self, root: Path, file_map: Dict[Path, int]):
        self._file_iid_to_path.clear()
        for row in self.files_tree.get_children():
            self.files_tree.delete(row)

        items = sorted(file_map.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (path, words) in enumerate(items, start=1):
            rel = str(path.relative_to(root))
            iid = self.files_tree.insert("", "end", values=(idx, rel, words))
            self._file_iid_to_path[iid] = path

        self.treeview_sort_column(self.files_tree, "words", True)
        self.files_tree.yview_moveto(0)

    def _populate_leaf_dirs(self, result: ScanResult):
        self._dir_iid_to_path.clear()
        for row in self.dirs_tree.get_children():
            self.dirs_tree.delete(row)

        sorted_dirs = sorted(result.leaf_dirs, key=lambda p: result.dir_word_counts.get(p, 0), reverse=True)
        for idx, d in enumerate(sorted_dirs, start=1):
            total = result.dir_word_counts.get(d, 0)
            count = result.dir_file_counts.get(d, 0) or 1
            avg = total / count if count else 0
            iid = self.dirs_tree.insert("", "end", values=(idx, d.name, total, f"{avg:.0f}", count))
            self._dir_iid_to_path[iid] = d

        self.treeview_sort_column(self.dirs_tree, "total", True)


    # --------- Länk-graf (anteckningar) ---------

    def on_links_mode_changed(self):
        if self.result and self.last_root:
            self._populate_links(self.last_root, self.result)

    def _populate_links(self, root: Path, result: ScanResult):
        if not hasattr(self, "links_tree"):
            return
        for row in self.links_tree.get_children():
            self.links_tree.delete(row)
        self._link_iid_to_path.clear()

        g = result.link_graph
        if not g or not g.id_to_path:
            if hasattr(self, "links_summary_label"):
                self.links_summary_label.config(text="Ingen länkdata (skanna först).")
            return

        n_nodes = len(g.id_to_path)
        n_edges = g.total_unique_edges
        orphans = sum(1 for x in g.in_degree if x == 0)
        dead_ends = sum(1 for x in g.out_degree if x == 0)

        if hasattr(self, "links_summary_label"):
            self.links_summary_label.config(
                text=(
                    f"Anteckningar: {n_nodes:,}   "
                    f"Länkar: {n_edges:,}   "
                    f"Orphans (in=0): {orphans:,}   "
                    f"Dead-ends (ut=0): {dead_ends:,}   "
                    f"Unresolved: {g.unresolved_links:,}   "
                    f"Ambiguous: {g.ambiguous_links:,}"
                ).replace(",", " ")
            )

        mode = (self.links_mode_var.get() or "In-länkar").strip()
        show_all = (mode == "Alla")
        top_n = max(1, int(self.top_n_var.get()))

        ids = list(range(n_nodes))
        if not show_all:
            if mode == "Ut-länkar":
                ids.sort(key=lambda i: (g.out_degree[i], g.in_degree[i]), reverse=True)
            elif mode == "PageRank":
                pr = g.pagerank or [0.0] * n_nodes
                ids.sort(key=lambda i: (pr[i], g.in_degree[i]), reverse=True)
            else:  # In-länkar
                ids.sort(key=lambda i: (g.in_degree[i], g.out_degree[i]), reverse=True)
            ids = ids[:top_n]

        pr = g.pagerank or [0.0] * n_nodes

        for idx, node_id in enumerate(ids, start=1):
            path = g.id_to_path[node_id]
            rel = str(path.relative_to(root)) if is_relative_to(path, root) else str(path)
            iid = self.links_tree.insert(
                "",
                "end",
                values=(
                    idx,
                    rel,
                    g.in_degree[node_id],
                    g.out_degree[node_id],
                    f"{pr[node_id]:.6f}"
                )
            )
            self._link_iid_to_path[iid] = path

        # Default-sort i tabben följer mode
        if mode == "Ut-länkar":
            self.treeview_sort_column(self.links_tree, "out", True)
        elif mode == "PageRank":
            self.treeview_sort_column(self.links_tree, "pr", True)
        else:
            self.treeview_sort_column(self.links_tree, "in", True)

        self.links_tree.yview_moveto(0)

    def _get_selected_link_file_path(self) -> Optional[Path]:
        if not hasattr(self, "links_tree"):
            return None
        sel = self.links_tree.focus()
        if not sel:
            return None
        return self._link_iid_to_path.get(sel)

    def on_links_selection_changed(self, event):
        p = self._get_selected_link_file_path()
        if p:
            rel = str(p.relative_to(self.last_root)) if self.last_root and is_relative_to(p, self.last_root) else str(p)
            self.heading_path_label.config(text=rel)
            self._populate_headings_panel(p)

    def on_open_selected_link_file(self, event):
        p = self._get_selected_link_file_path()
        if not p:
            messagebox.showinfo("Öppna fil", "Markera en fil i Länkar först.")
            return
        try:
            self._open_in_default_editor(p)
            self.status_var.set(f"Öppnade: {p.name}")
        except Exception as e:
            messagebox.showerror("Öppna fil", f"Kunde inte öppna filen:\n{e!r}")

    def on_copy_selected_link_file(self):
        p = self._get_selected_link_file_path()
        if not p:
            messagebox.showinfo("Kopiera sökväg", "Markera en fil i Länkar först.")
            return
        rel = str(p.relative_to(self.last_root)) if self.last_root and is_relative_to(p, self.last_root) else str(p)
        self.master.clipboard_clear()
        self.master.clipboard_append(rel)
        self.status_var.set(f"Kopierade: {rel}")


    # --------- Länk-graf (mappnivå) ---------

    def _populate_link_dirs(self, root: Path, result: ScanResult):
        if not hasattr(self, "link_dirs_tree"):
            return

        for t in (self.link_dirs_tree, self.link_pairs_tree):
            for row in t.get_children():
                t.delete(row)

        stats = result.folder_link_stats
        g = result.link_graph
        if not stats or not g:
            if hasattr(self, "link_dirs_summary_label"):
                self.link_dirs_summary_label.config(text="Ingen länkdata (skanna först).")
            return

        folders = list(stats.folder_to_ids.keys())
        if hasattr(self, "link_dirs_summary_label"):
            self.link_dirs_summary_label.config(
                text=f"Mappar: {len(folders):,}   (räknar endast mappar som innehåller .md direkt)".replace(",", " ")
            )

        # Per folder
        rows = []
        for folder, ids in stats.folder_to_ids.items():
            rel = str(folder.relative_to(root)) if is_relative_to(folder, root) else str(folder)
            notes = len(ids)
            internal = stats.internal_edges.get(folder, 0)
            dens = stats.density.get(folder, 0.0)
            outb = stats.outbound_edges.get(folder, 0)
            inb = stats.inbound_edges.get(folder, 0)
            rows.append((rel, notes, internal, dens, outb, inb))

        # sortera initialt på densitet
        rows.sort(key=lambda t: t[3], reverse=True)
        for idx, (rel, notes, internal, dens, outb, inb) in enumerate(rows, start=1):
            self.link_dirs_tree.insert(
                "",
                "end",
                values=(idx, rel, notes, internal, f"{dens:.4f}", outb, inb)
            )

        self.treeview_sort_column(self.link_dirs_tree, "density", True)
        self.link_dirs_tree.yview_moveto(0)

        # Topp folder->folder edges (exkl. fu==fv)
        pairs = [((fu, fv), c) for (fu, fv), c in stats.folder_matrix.items() if fu != fv and c > 0]
        pairs.sort(key=lambda kv: kv[1], reverse=True)
        max_pairs = 200
        for idx, ((fu, fv), c) in enumerate(pairs[:max_pairs], start=1):
            fr = str(fu.relative_to(root)) if is_relative_to(fu, root) else str(fu)
            to = str(fv.relative_to(root)) if is_relative_to(fv, root) else str(fv)
            self.link_pairs_tree.insert("", "end", values=(idx, fr, to, c))

        self.treeview_sort_column(self.link_pairs_tree, "edges", True)
        self.link_pairs_tree.yview_moveto(0)


    # --------- Gemensamma helpers ---------

    def _get_selected_any_file_path(self) -> Optional[Path]:
        return self._get_selected_file_path() or self._get_selected_folder_file_path() or self._get_selected_link_file_path()

    def on_open_selected_any_file(self):
        p = self._get_selected_any_file_path()
        if not p:
            messagebox.showinfo("Öppna fil", "Markera en fil i någon tabell först (Toppfiler/Mappanalys/Länkar).")
            return
        try:
            self._open_in_default_editor(p)
            self.status_var.set(f"Öppnade: {p.name}")
        except Exception as e:
            messagebox.showerror("Öppna fil", f"Kunde inte öppna filen:\n{e!r}")

    # --------- Mappanalys ---------

    def on_analyze_selected_dir(self):
        """Bygg mappanalys av vald mapp (endast filer direkt i mappen)."""
        if not self.result or not self.last_root:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return
        sel = self.dirs_tree.focus()
        if not sel:
            messagebox.showinfo("Analysera mapp", "Markera en mapp i listan först.")
            return
        dir_path = self._dir_iid_to_path.get(sel)
        if not dir_path:
            messagebox.showerror("Analysera mapp", "Kunde inte avgöra mappens sökväg.")
            return

        # Filtrer a l l a filer som ligger DIREKT i mappen
        files_in_dir: Dict[Path, int] = {
            p: w for p, w in self.result.file_word_counts.items() if p.parent == dir_path
        }
        if not files_in_dir:
            messagebox.showinfo("Analysera mapp", "Inga .md-filer direkt i denna mapp.")
            return

        self._populate_folder_analysis(dir_path, files_in_dir)
        rel = str(dir_path.relative_to(self.last_root))
        self.folder_header_label.config(text=rel)
        self.nb.select(self.tab_folder)
        self.status_var.set(f"Mappanalys: {rel} – {len(files_in_dir)} filer.")

    def _populate_folder_analysis(self, dir_path: Path, file_map: Dict[Path, int]):
        self._folder_iid_to_path.clear()
        self._link_iid_to_path.clear()
        for row in self.folder_tree.get_children():
            self.folder_tree.delete(row)

        # sortera fallande på ord
        items = sorted(file_map.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (path, words) in enumerate(items, start=1):
            # visa filnamn (lokalt inom mappen)
            iid = self.folder_tree.insert("", "end", values=(idx, path.name, words))
            self._folder_iid_to_path[iid] = path

        self.treeview_sort_column(self.folder_tree, "words", True)
        self.folder_tree.yview_moveto(0)

    # ---------------- Headings panel ----------------

    def _populate_headings_panel(self, path: Optional[Path]):
        for row in self.headings_tree.get_children():
            self.headings_tree.delete(row)
        if not path:
            return
        hs = extract_headings(path)
        for level, text in hs:
            self.headings_tree.insert("", "end", values=(level, text))

    def on_files_selection_changed(self, event):
        p = self._get_selected_file_path()
        if p:
            rel = str(p.relative_to(self.last_root)) if self.last_root else str(p)
            self.heading_path_label.config(text=rel)
            self._populate_headings_panel(p)

    def on_folder_selection_changed(self, event):
        p = self._get_selected_folder_file_path()
        if p:
            rel = str(p.relative_to(self.last_root)) if self.last_root else str(p)
            self.heading_path_label.config(text=rel)
            self._populate_headings_panel(p)

    # ---------------- Enable/disable ----------------

    def _set_controls_state(self, state: str):
        for frame in self.winfo_children():
            if isinstance(frame, ttk.Panedwindow):
                for pane in frame.winfo_children():
                    for child in pane.winfo_children():
                        for c in child.winfo_children():
                            if isinstance(c, (ttk.Entry, ttk.Spinbox, ttk.Button, ttk.Combobox, ttk.Checkbutton)):
                                try:
                                    c.configure(state=state)
                                except tk.TclError:
                                    pass

    # ---------------- File helpers ----------------

    def _get_selected_file_path(self) -> Optional[Path]:
        sel = self.files_tree.focus()
        if not sel:
            return None
        return self._file_iid_to_path.get(sel)

    def _get_selected_folder_file_path(self) -> Optional[Path]:
        sel = self.folder_tree.focus()
        if not sel:
            return None
        return self._folder_iid_to_path.get(sel)

    def on_copy_selected_file(self):
        p = self._get_selected_file_path()
        if not p:
            messagebox.showinfo("Kopiera sökväg", "Markera en fil i Toppfiler först.")
            return
        rel = str(p.relative_to(self.last_root)) if self.last_root else str(p)
        self.master.clipboard_clear()
        self.master.clipboard_append(rel)
        self.status_var.set(f"Kopierade: {rel}")

    def on_copy_selected_folder_file(self):
        p = self._get_selected_folder_file_path()
        if not p:
            messagebox.showinfo("Kopiera sökväg", "Markera en fil i Mappanalys först.")
            return
        rel = str(p.relative_to(self.last_root)) if self.last_root else str(p)
        self.master.clipboard_clear()
        self.master.clipboard_append(rel)
        self.status_var.set(f"Kopierade: {rel}")

    def _open_in_default_editor(self, path: Path):
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def on_open_selected_file(self, event):
        p = self._get_selected_file_path()
        if not p:
            messagebox.showinfo("Öppna fil", "Markera en fil i Toppfiler först.")
            return
        try:
            self._open_in_default_editor(p)
            self.status_var.set(f"Öppnade: {p.name}")
        except Exception as e:
            messagebox.showerror("Öppna fil", f"Kunde inte öppna filen:\n{e!r}")

    def on_open_selected_folder_file(self, event):
        p = self._get_selected_folder_file_path()
        if not p:
            messagebox.showinfo("Öppna fil", "Markera en fil i Mappanalys först.")
            return
        try:
            self._open_in_default_editor(p)
            self.status_var.set(f"Öppnade: {p.name}")
        except Exception as e:
            messagebox.showerror("Öppna fil", f"Kunde inte öppna filen:\n{e!r}")

    # ---------------- Rubriker copy ----------------

    def on_copy_headings(self):
        p = self._get_selected_file_path() or self._get_selected_folder_file_path() or self._get_selected_link_file_path()
        if not p:
            messagebox.showinfo("Kopiera rubriker", "Markera en fil i Toppfiler eller Mappanalys först.")
            return
        hs = extract_headings(p)
        if not hs:
            messagebox.showinfo("Kopiera rubriker", "Inga rubriker hittades i filen.")
            return
        lines = []
        for level, text in hs:
            text = text.strip()
            if text:
                lines.append("#" * max(1, min(6, level)) + " " + text)
        payload = "\n".join(lines)
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(payload)
            self.status_var.set(f"Kopierade {len(lines)} rubriker från: {p.name}")
        except Exception as e:
            messagebox.showerror("Kopiera rubriker", f"Kunde inte kopiera rubriker:\n{e!r}")

    # ---------------- Sortering ----------------

    def treeview_sort_column(self, tv: ttk.Treeview, col: str, reverse: bool):
        rows = [(tv.set(k, col), k) for k in tv.get_children('')]

        def to_number(s: str):
            try:
                return float(str(s).replace(" ", ""))
            except Exception:
                return None

        if rows and all(to_number(v) is not None for v, _ in rows):
            rows.sort(key=lambda t: to_number(t[0]), reverse=reverse)
        else:
            rows.sort(key=lambda t: t[0], reverse=reverse)

        for index, (_, k) in enumerate(rows):
            tv.move(k, '', index)

        cols = tv["columns"]
        if "rank" in cols:
            rank_idx = list(cols).index("rank")
            for new_rank, (_, k) in enumerate(rows, start=1):
                vals = list(tv.item(k, "values"))
                if len(vals) > rank_idx:
                    vals[rank_idx] = new_rank
                    tv.item(k, values=vals)

        tv.heading(col, command=lambda: self.treeview_sort_column(tv, col, not reverse))

    # ---------------- Plotting ----------------

    def _require_mpl(self) -> bool:
        if not _MPL_OK:
            messagebox.showwarning(
                "Grafer kräver matplotlib",
                "Installera matplotlib för att visa grafer:\n\npip install matplotlib"
            )
            return False
        return True

    def on_plot_top_files(self):
        if not self._require_mpl():
            return
        if not self.result or not self.result.file_word_counts:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return

        n = max(1, int(self.top_n_var.get()))
        top_items = Counter(self.result.file_word_counts).most_common(n)
        labels = [p.name for p, _ in top_items]
        values = [w for _, w in top_items]

        self._open_plot_window(
            title=f"Topp {n} filer – ord",
            plot_fn=lambda fig: self._bar_plot(fig, labels, values, xlabel="Filer", ylabel="Antal ord", rotation=45)
        )

    def on_plot_top_dirs(self):
        if not self._require_mpl():
            return
        if not self.result or not self.result.leaf_dirs:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return

        dirs_sorted = sorted(self.result.leaf_dirs, key=lambda p: self.result.dir_word_counts.get(p, 0), reverse=True)[:15]
        labels = [d.name for d in dirs_sorted]
        values = [self.result.dir_word_counts.get(d, 0) for d in dirs_sorted]

        self._open_plot_window(
            title="Mappar (med filer) – ord",
            plot_fn=lambda fig: self._bar_plot(fig, labels, values, xlabel="Mappar", ylabel="Totalt ord", rotation=30)
        )

    def on_plot_pie_dirs(self):
        if not self._require_mpl():
            return
        if not self.result or not self.result.leaf_dirs:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return

        dirs_sorted = sorted(self.result.leaf_dirs, key=lambda p: self.result.dir_word_counts.get(p, 0), reverse=True)
        top = dirs_sorted[:20]
        labels = [d.name for d in top]
        values = [self.result.dir_word_counts.get(d, 0) for d in top]
        rest = sum(self.result.dir_word_counts.get(d, 0) for d in dirs_sorted[20:])
        if rest > 0:
            labels.append("Övrigt")
            values.append(rest)

        self._open_plot_window(
            title="Mappar – fördelning (sektordiagram)",
            plot_fn=lambda fig: self._pie_plot(fig, labels, values)
        )



    def on_plot_top_in_links(self):
        if not self._require_mpl():
            return
        if not self.result or not self.result.link_graph or not self.result.link_graph.id_to_path:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return

        g = self.result.link_graph
        n = min(25, len(g.id_to_path))
        ids = list(range(len(g.id_to_path)))
        ids.sort(key=lambda i: (g.in_degree[i], g.out_degree[i]), reverse=True)
        ids = ids[:n]

        labels = [g.id_to_path[i].name for i in ids]
        values = [g.in_degree[i] for i in ids]

        self._open_plot_window(
            title=f"Topp {n} anteckningar – in-länkar",
            plot_fn=lambda fig: self._bar_plot(fig, labels, values, xlabel="Anteckningar", ylabel="In-länkar", rotation=45)
        )

    def on_plot_top_pagerank(self):
        if not self._require_mpl():
            return
        if not self.result or not self.result.link_graph or not self.result.link_graph.id_to_path:
            messagebox.showinfo("Ingen data", "Skanna först.")
            return

        g = self.result.link_graph
        pr = g.pagerank or [0.0] * len(g.id_to_path)
        n = min(25, len(g.id_to_path))
        ids = list(range(len(g.id_to_path)))
        ids.sort(key=lambda i: (pr[i], g.in_degree[i]), reverse=True)
        ids = ids[:n]

        labels = [g.id_to_path[i].name for i in ids]
        values = [pr[i] for i in ids]

        self._open_plot_window(
            title=f"Topp {n} anteckningar – PageRank",
            plot_fn=lambda fig: self._bar_plot(fig, labels, values, xlabel="Anteckningar", ylabel="PageRank", rotation=45)
        )

    # ---------------- Plot helpers ----------------

    def _open_plot_window(self, title: str, plot_fn):
        win = tk.Toplevel(self.master)
        win.title(title)
        win.geometry("900x640")
        fig = Figure(figsize=(8.5, 6.0), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plot_fn(fig)
        canvas.draw()

    def _bar_plot(self, fig: 'Figure', labels: List[str], values: List[int], xlabel: str, ylabel: str, rotation: int = 0):
        ax = fig.add_subplot(111)
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rotation, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()

    def _pie_plot(self, fig: 'Figure', labels: List[str], values: List[int]):
        ax = fig.add_subplot(111)
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        ax.set_title("Andel ord per mapp")
        fig.tight_layout()


# ===================== main =====================

def main():
    root = tk.Tk()
    try:
        root.call("source", "sun-valley.tcl")
        ttk.Style().theme_use("sun-valley-light")
    except Exception:
        pass

    # Lila bakgrund/tema
    root.configure(bg="#E6E6FA")
    style = ttk.Style()
    style.configure("Purple.TFrame", background="#E6E6FA")
    style.configure("TLabel", background="#E6E6FA", foreground="black")

    style.configure("Treeview",
                    background="#E6E6FA",
                    fieldbackground="#E6E6FA",
                    foreground="black")
    style.map("Treeview",
              background=[("selected", "#C8A2C8")])
    style.configure("Treeview.Heading",
                    background="#D8BFD8",
                    foreground="black")

    style.configure("TButton",
                    background="#DDA0DD",
                    foreground="black",
                    padding=6)
    style.map("TButton", background=[("active", "#BA55D3")])

    style.configure("TCheckbutton", background="#E6E6FA")
    style.configure("TCombobox",
                    fieldbackground="#E6E6FA",
                    background="#E6E6FA")
    style.map("TCombobox",
              fieldbackground=[("readonly", "#E6E6FA")],
              background=[("readonly", "#E6E6FA")])

    app = ObsidianSummaryApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()
