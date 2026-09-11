#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paritetstest: kör Python-versionens analysfunktioner (Obsidianalys.py) och C++-kärnan
(obsidianalys --json) på samma vault och jämför allt:
- ordräkning per fil för alla fyra kombinationer av ekvations-/kodväxlarna
- länkgrafen (kanter, in/ut-grad, olösta/tvetydiga länkar)
- PageRank
- mappstatistik och mapp → mapp-länkar
- rubriker (för filer utan kodblock – C++ hoppar medvetet över "# ..." i kodblock)

Kör: python3 tools/parity_check.py VAULT [--bin build/obsidianalys]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def import_legacy():
    # Python-versionen importerar tkinter på modulnivå – stubba bort det så att
    # analysfunktionerna kan köras utan Tk och utan skärm.
    class Stub(types.ModuleType):
        def __getattr__(self, name):
            return object

    tk = Stub("tkinter")
    for sub in ("ttk", "filedialog", "messagebox"):
        mod = Stub(f"tkinter.{sub}")
        setattr(tk, sub, mod)
        sys.modules[f"tkinter.{sub}"] = mod
    sys.modules["tkinter"] = tk
    sys.path.insert(0, str(REPO))
    import Obsidianalys  # noqa: E402

    return Obsidianalys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vault")
    ap.add_argument("--bin", default=str(REPO / "build" / "obsidianalys"))
    args = ap.parse_args()
    vault = Path(args.vault).expanduser().resolve()
    O = import_legacy()

    t = time.perf_counter()
    proc = subprocess.run([args.bin, "--json", str(vault)], capture_output=True, check=True)
    t_cpp_process = time.perf_counter() - t
    cpp = json.loads(proc.stdout)

    # En "Skanna" i Python-appen = analyze_vault + länkgraf + PageRank + mappstatistik
    t = time.perf_counter()
    r = O.analyze_vault(vault, include_equations=False, include_code=False)
    g = O.build_link_graph(list(r.file_word_counts))
    g.pagerank = O.compute_pagerank(g) if g.total_unique_edges else [0.0] * len(g.id_to_path)
    fstats = O.compute_folder_link_stats(g)
    t_py = time.perf_counter() - t

    words = {}
    for eq in (False, True):
        for code in (False, True):
            res = r if not eq and not code else O.analyze_vault(vault, include_equations=eq, include_code=code)
            words[(eq, code)] = {p.relative_to(vault).as_posix(): w for p, w in res.file_word_counts.items()}

    errors: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    notes = cpp["notes"]
    rels = [n["rel"] for n in notes]
    check(set(rels) == set(words[(False, False)]), f"filmängd skiljer: {set(rels) ^ set(words[(False, False)])}")

    for (eq, code), ref in words.items():
        k = (2 if eq else 0) + (1 if code else 0)
        bad = [(n["rel"], ref.get(n["rel"]), n["w"][k]) for n in notes if ref.get(n["rel"]) != n["w"][k]]
        check(not bad, f"ordräkning (ekv={eq}, kod={code}) skiljer i {len(bad)} filer, t.ex. {bad[:3]}")

    py_rel = [p.relative_to(vault).as_posix() for p in g.id_to_path]
    py_id = {rel: i for i, rel in enumerate(py_rel)}
    stats = cpp["stats"]
    check(stats["edges"] == g.total_unique_edges, f"kanter: C++ {stats['edges']} vs Python {g.total_unique_edges}")
    check(stats["unresolved"] == g.unresolved_links, f"olösta: C++ {stats['unresolved']} vs Python {g.unresolved_links}")
    check(stats["ambiguous"] == g.ambiguous_links, f"tvetydiga: C++ {stats['ambiguous']} vs Python {g.ambiguous_links}")

    bad_links, max_pr_diff = [], 0.0
    for n in notes:
        u = py_id.get(n["rel"])
        if u is None:
            continue
        py_out = {py_rel[v] for v in g.out_neighbors[u]}
        cpp_out = {rels[j] for j in n["links"]}
        if py_out != cpp_out or n["in"] != g.in_degree[u] or n["out"] != g.out_degree[u]:
            bad_links.append((n["rel"], sorted(py_out ^ cpp_out)[:3]))
        max_pr_diff = max(max_pr_diff, abs(n["pr"] - g.pagerank[u]))
    check(not bad_links, f"länkar skiljer för {len(bad_links)} filer, t.ex. {bad_links[:3]}")
    check(max_pr_diff < 1e-12, f"PageRank skiljer (max diff {max_pr_diff:.3e})")

    def frel(folder: Path) -> str:
        rel = folder.relative_to(vault).as_posix()
        return "" if rel == "." else rel

    cpp_folders = {f["rel"]: f for f in cpp["folders"]}
    check(set(cpp_folders) == {frel(f) for f in fstats.folder_to_ids}, "mappmängd skiljer")
    for folder, ids in fstats.folder_to_ids.items():
        c = cpp_folders.get(frel(folder))
        if c is None:
            continue
        py = (len(ids), fstats.internal_edges.get(folder, 0), fstats.outbound_edges.get(folder, 0), fstats.inbound_edges.get(folder, 0))
        cc = (c["n"], c["internal"], c["out"], c["in"])
        check(py == cc and abs(c["density"] - fstats.density[folder]) < 1e-12, f"mapp {frel(folder)!r}: Python {py} vs C++ {cc}")

    folder_rel = [f["rel"] for f in cpp["folders"]]
    cpp_pairs = {(folder_rel[a], folder_rel[b]): e for a, b, e in cpp["pairs"]}
    py_pairs = {(frel(a), frel(b)): e for (a, b), e in fstats.folder_matrix.items() if a != b and e > 0}
    check(cpp_pairs == py_pairs, f"mapp → mapp-länkar skiljer ({len(set(cpp_pairs.items()) ^ set(py_pairs.items()))} par)")

    compared = skipped = 0
    bad_headings = []
    for n in notes:
        path = vault / n["rel"]
        txt = O.read_text_safe(path)
        if "```" in txt or "~~~" in txt:
            skipped += 1
            continue
        compared += 1
        if O.extract_headings(path) != [tuple(h) for h in n["headings"]]:
            bad_headings.append(n["rel"])
    check(not bad_headings, f"rubriker skiljer i {len(bad_headings)} filer, t.ex. {bad_headings[:3]}")

    print(f"Vault: {vault}")
    print(f"  {len(notes)} filer, {len(cpp['folders'])} mappar, {stats['edges']} länkar, "
          f"{stats['unresolved']} olösta, {stats['ambiguous']} tvetydiga")
    print(f"  rubriker jämförda i {compared} filer ({skipped} med kodblock hoppades över)")
    print(f"  PageRank max avvikelse: {max_pr_diff:.2e}")
    print(f"Tid för en skanning:")
    print(f"  Python:          {t_py * 1000:8.1f} ms")
    print(f"  C++ (kärna):     {cpp['timings']['total']:8.1f} ms")
    print(f"  C++ (process):   {t_cpp_process * 1000:8.1f} ms  (start + skanning + JSON med rubriker)")
    if errors:
        print(f"\nAVVIKELSER ({len(errors)}):")
        for e in errors:
            print("  - " + e)
        return 1
    print("\nOK – identiska resultat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
