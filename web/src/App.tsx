import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import { Segmented, Switch, type ViewMode } from './components/controls';
import { FolderPicker } from './components/FolderPicker';
import { Icon, type IconName } from './components/Icon';
import { Inspector } from './components/Inspector';
import { fmt, fmtMs, MODE_LABEL, wordIndex } from './format';
import type { AppState, FileRow, FolderRow, Scan, SearchMode, Theme } from './types';
import { FilesView } from './views/FilesView';
import { FolderAnalysisView } from './views/FolderAnalysisView';
import { FolderLinksView } from './views/FolderLinksView';
import { FoldersView } from './views/FoldersView';
import { LinksView, type LinksMode } from './views/LinksView';

type Tab = 'files' | 'folders' | 'folder' | 'links' | 'folderLinks';

const TABS: { id: Tab; label: string; icon: IconName }[] = [
  { id: 'files', label: 'Toppfiler', icon: 'file' },
  { id: 'folders', label: 'Mappar', icon: 'folder' },
  { id: 'folder', label: 'Mappanalys', icon: 'folderOpen' },
  { id: 'links', label: 'Länkar', icon: 'link' },
  { id: 'folderLinks', label: 'Mappkoppling', icon: 'network' },
];

interface ActiveSearch {
  q: string;
  mode: SearchMode;
  ids: number[];
}

const countingText = (eq: boolean, code: boolean) =>
  `ekvationer ${eq ? 'räknas' : 'ignoreras'} · kod ${code ? 'räknas' : 'ignoreras'}`;

export default function App() {
  const [appState, setAppState] = useState<AppState | null>(null);
  const [rootInput, setRootInput] = useState('');
  const [scan, setScan] = useState<Scan | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState('Redo.');
  const [tab, setTab] = useState<Tab>('files');
  const [views, setViews] = useState<Record<Tab, ViewMode>>({
    files: 'table',
    folders: 'table',
    folder: 'table',
    links: 'table',
    folderLinks: 'table',
  });
  const [topN, setTopN] = useState(30);
  const [showAll, setShowAll] = useState(false);
  const [includeEq, setIncludeEq] = useState(false);
  const [includeCode, setIncludeCode] = useState(false);
  const [theme, setTheme] = useState<Theme>('system');
  const [query, setQuery] = useState('');
  const [searchMode, setSearchMode] = useState<SearchMode>('name');
  const [search, setSearch] = useState<ActiveSearch | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [folderId, setFolderId] = useState<number | null>(null);
  const [linksMode, setLinksMode] = useState<LinksMode>('in');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [stopped, setStopped] = useState(false);
  const searchSeq = useRef(0);
  const searchInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = document.documentElement;
    if (theme === 'system') delete el.dataset.theme;
    else el.dataset.theme = theme;
  }, [theme]);

  const runScan = useCallback(async (root: string) => {
    const r = root.trim();
    if (!r) {
      setError('Ange en rotmapp.');
      return;
    }
    setScanning(true);
    setError(null);
    setStatus('Skannar…');
    const t0 = performance.now();
    try {
      const s = await api.scan(r);
      setScan(s);
      setRootInput(s.root);
      setSelected(null);
      setFolderId(null);
      setStatus(
        `Klar – ${fmt(s.stats.files)} filer analyserade på ${fmtMs(s.timings.total)} ` +
          `(${fmtMs(performance.now() - t0)} inkl. överföring till gränssnittet).`,
      );
    } catch (e) {
      setError((e as Error).message);
      setStatus('Fel vid skanning.');
    } finally {
      setScanning(false);
    }
  }, []);

  useEffect(() => {
    api
      .state()
      .then((s) => {
        const c = s.config;
        setAppState(s);
        setTopN(c.topN);
        setIncludeEq(c.includeEquations);
        setIncludeCode(c.includeCode);
        setTheme(c.theme);
        setRootInput(c.root);
        if (c.root) runScan(c.root);
      })
      .catch((e: Error) => setError(`Kunde inte nå Obsidianalys-servern: ${e.message}`));
  }, [runScan]);

  // Direktsökning i C++-kärnan (debounce; äldre svar ignoreras)
  useEffect(() => {
    const q = query.trim();
    const seq = ++searchSeq.current;
    if (!scan || !q) {
      setSearch(null);
      return;
    }
    const timer = window.setTimeout(async () => {
      try {
        const r = await api.search(q, searchMode);
        if (seq !== searchSeq.current) return;
        setSearch({ q, mode: searchMode, ids: r.ids });
        setStatus(
          `Hittade ${fmt(r.ids.length)} träff${r.ids.length === 1 ? '' : 'ar'} för ”${q}” ` +
            `(${MODE_LABEL[searchMode]}) på ${fmtMs(r.ms)}.`,
        );
      } catch (e) {
        if (seq === searchSeq.current) setStatus(`Sökfel: ${(e as Error).message}`);
      }
    }, 120);
    return () => window.clearTimeout(timer);
  }, [query, searchMode, scan]);

  // "/" eller Ctrl+K fokuserar sökfältet
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement;
      if ((e.key === 'k' && (e.ctrlKey || e.metaKey)) || (e.key === '/' && !typing)) {
        e.preventDefault();
        searchInput.current?.focus();
        searchInput.current?.select();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const saveConfig = (patch: Parameters<typeof api.config>[0]) => {
    api.config(patch).catch(() => undefined);
  };

  const wi = wordIndex(includeEq, includeCode);
  const files = useMemo<FileRow[]>(
    () =>
      scan
        ? scan.notes.map((n, id) => ({
            id,
            rel: n.rel,
            name: n.name,
            folder: n.f,
            words: n.w[wi],
            in: n.in,
            out: n.out,
            pr: n.pr,
            headings: n.h,
          }))
        : [],
    [scan, wi],
  );

  const folders = useMemo<FolderRow[]>(() => {
    if (!scan) return [];
    const words = new Array<number>(scan.folders.length).fill(0);
    for (const f of files) words[f.folder] += f.words;
    return scan.folders.map((f, id) => ({
      id,
      rel: f.rel,
      name: f.name,
      words: words[id],
      count: f.n,
      avg: f.n ? words[id] / f.n : 0,
      internal: f.internal,
      density: f.density,
      out: f.out,
      in: f.in,
    }));
  }, [scan, files]);

  const totalWords = useMemo(() => files.reduce((s, f) => s + f.words, 0), [files]);
  const limit = showAll ? Number.POSITIVE_INFINITY : topN;
  const limitText = showAll ? 'Alla filer' : `Topp ${fmt(topN)}`;

  const fileRows = useMemo(
    () =>
      search
        ? search.ids.map((i) => files[i]).filter(Boolean)
        : [...files].sort((a, b) => b.words - a.words || a.id - b.id).slice(0, limit),
    [search, files, limit],
  );

  const openFile = useCallback(
    (id: number) => {
      if (!scan) return;
      api
        .open(id, scan.generation)
        .then(() => setStatus(`Öppnade: ${files[id]?.name}`))
        .catch((e: Error) => setStatus(`Kunde inte öppna filen: ${e.message}`));
    },
    [scan, files],
  );

  const analyzeFolder = useCallback(
    (id: number) => {
      const f = folders[id];
      setFolderId(id);
      setTab('folder');
      if (f) setStatus(`Mappanalys: ${f.rel || scan?.rootName} – ${fmt(f.count)} filer.`);
    },
    [folders, scan],
  );

  const copy = useCallback(async (text: string, what: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setStatus(`Kopierade ${what}`);
    } catch {
      setStatus('Kunde inte kopiera till urklipp.');
    }
  }, []);

  const setView = (v: ViewMode) => setViews((s) => ({ ...s, [tab]: v }));

  if (stopped) {
    return (
      <div className="stopped">
        <Icon name="power" size={32} />
        <h1>Obsidianalys är avslutat</h1>
        <p>Du kan stänga den här fliken. Starta programmet igen för att fortsätta.</p>
      </div>
    );
  }

  const counts: Record<Tab, string> = scan
    ? {
        files: fmt(files.length),
        folders: fmt(folders.length),
        folder: folderId != null && folders[folderId] ? fmt(folders[folderId].count) : '',
        links: fmt(scan.stats.edges),
        folderLinks: fmt(scan.pairs.length),
      }
    : { files: '', folders: '', folder: '', links: '', folderLinks: '' };

  let content = null;
  if (scan) {
    switch (tab) {
      case 'files':
        content = (
          <FilesView
            rows={fileRows}
            search={search ? { q: search.q, mode: search.mode, count: search.ids.length } : null}
            onClearSearch={() => setQuery('')}
            limitText={limitText}
            selected={selected}
            onSelect={setSelected}
            onOpen={openFile}
            view={views.files}
            onView={setView}
          />
        );
        break;
      case 'folders':
        content = <FoldersView folders={folders} rootName={scan.rootName} onAnalyze={analyzeFolder} view={views.folders} onView={setView} />;
        break;
      case 'folder':
        content = (
          <FolderAnalysisView
            folders={folders}
            files={files}
            rootName={scan.rootName}
            folderId={folderId}
            onFolder={analyzeFolder}
            selected={selected}
            onSelect={setSelected}
            onOpen={openFile}
            view={views.folder}
            onView={setView}
          />
        );
        break;
      case 'links':
        content = (
          <LinksView
            files={files}
            stats={scan.stats}
            limit={limit}
            limitText={limitText}
            mode={linksMode}
            onMode={setLinksMode}
            selected={selected}
            onSelect={setSelected}
            onOpen={openFile}
            view={views.links}
            onView={setView}
          />
        );
        break;
      case 'folderLinks':
        content = (
          <FolderLinksView
            folders={folders}
            pairs={scan.pairs}
            rootName={scan.rootName}
            onAnalyze={analyzeFolder}
            view={views.folderLinks}
            onView={setView}
          />
        );
        break;
    }
  }

  return (
    <div className={`app${selected != null ? ' has-selection' : ''}`}>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Icon name="gem" size={19} />
          </span>
          <div>
            <strong>Obsidianalys</strong>
            <span>version {appState?.version ?? '2'}</span>
          </div>
        </div>

        <nav className="nav" aria-label="Vyer">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={tab === t.id ? 'is-active' : undefined}
              aria-current={tab === t.id ? 'page' : undefined}
              onClick={() => setTab(t.id)}
              disabled={!scan}
            >
              <Icon name={t.icon} />
              <span>{t.label}</span>
              <span className="nav-count">{counts[t.id]}</span>
            </button>
          ))}
        </nav>

        <div className="side-section">
          <div className="side-title">Ordräkning</div>
          <Switch
            label="Räkna ekvationer"
            hint="$…$ och $$…$$"
            checked={includeEq}
            onChange={(v) => {
              setIncludeEq(v);
              saveConfig({ eq: v });
              setStatus(`Ordräkning: ${countingText(v, includeCode)}.`);
            }}
          />
          <Switch
            label="Räkna kod"
            hint="inline-kod och kodblock"
            checked={includeCode}
            onChange={(v) => {
              setIncludeCode(v);
              saveConfig({ code: v });
              setStatus(`Ordräkning: ${countingText(includeEq, v)}.`);
            }}
          />
        </div>

        <div className="side-section">
          <div className="side-title">Urval</div>
          <TopNInput
            value={topN}
            disabled={showAll}
            onChange={(n) => {
              setTopN(n);
              saveConfig({ topN: n });
            }}
          />
          <Switch label="Visa alla" checked={showAll} onChange={setShowAll} />
        </div>

        <div className="side-footer">
          <Segmented
            label="Tema"
            size="sm"
            value={theme}
            onChange={(t) => {
              setTheme(t);
              saveConfig({ theme: t });
            }}
            options={[
              { value: 'system', label: 'System', icon: 'monitor' },
              { value: 'light', label: 'Ljust', icon: 'sun' },
              { value: 'dark', label: 'Mörkt', icon: 'moon' },
            ]}
          />
          <button type="button" className="btn ghost sm" onClick={() => api.quit().finally(() => setStopped(true))}>
            <Icon name="power" size={15} />
            Avsluta Obsidianalys
          </button>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <form
            className="vault-form"
            onSubmit={(e) => {
              e.preventDefault();
              runScan(rootInput);
            }}
          >
            <label className="vault-input">
              <Icon name="folder" />
              <input
                value={rootInput}
                onChange={(e) => setRootInput(e.target.value)}
                placeholder="Sökväg till din vault…"
                spellCheck={false}
                aria-label="Rotmapp"
              />
            </label>
            <button type="button" className="btn" onClick={() => setPickerOpen(true)} disabled={!appState}>
              Välj…
            </button>
            <button type="submit" className="btn primary" disabled={scanning || !appState}>
              <Icon name="refresh" size={16} className={scanning ? 'spin' : undefined} />
              {scanning ? 'Skannar…' : 'Skanna'}
            </button>
          </form>
        </header>

        {error && (
          <div className="banner" role="alert">
            <span>{error}</span>
            <button type="button" className="icon-btn" onClick={() => setError(null)} aria-label="Stäng">
              <Icon name="x" size={16} />
            </button>
          </div>
        )}

        {scan ? (
          <div className={`content${scanning ? ' is-busy' : ''}`}>
            <div className="kpis">
              <div className="kpi kpi-hero">
                <div className="kpi-label">Totalt antal ord</div>
                <div className="kpi-value">{fmt(totalWords)}</div>
                <div className="kpi-sub">{countingText(includeEq, includeCode)}</div>
              </div>
              <div className="kpi">
                <div className="kpi-label">Filer</div>
                <div className="kpi-value">{fmt(files.length)}</div>
                <div className="kpi-sub">snitt {fmt(files.length ? totalWords / files.length : 0)} ord/fil</div>
              </div>
              <div className="kpi">
                <div className="kpi-label">Mappar</div>
                <div className="kpi-value">{fmt(folders.length)}</div>
                <div className="kpi-sub">med .md-filer direkt i sig</div>
              </div>
              <div className="kpi">
                <div className="kpi-label">Länkar</div>
                <div className="kpi-value">{fmt(scan.stats.edges)}</div>
                <div className="kpi-sub">
                  {fmt(scan.stats.unresolved)} olösta · {fmt(scan.stats.ambiguous)} tvetydiga
                </div>
              </div>
            </div>

            <div className="searchbar">
              <label className="search-input">
                <Icon name="search" />
                <input
                  ref={searchInput}
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    if (e.target.value.trim()) setTab('files');
                  }}
                  onKeyDown={(e) => e.key === 'Escape' && setQuery('')}
                  placeholder={`Sök i ${MODE_LABEL[searchMode].toLowerCase()}…`}
                  aria-label="Sök"
                />
                {query ? (
                  <button type="button" className="icon-btn sm" onClick={() => setQuery('')} aria-label="Rensa sökningen">
                    <Icon name="x" size={15} />
                  </button>
                ) : (
                  <kbd>/</kbd>
                )}
              </label>
              <Segmented
                label="Sök i"
                value={searchMode}
                onChange={setSearchMode}
                options={[
                  { value: 'name', label: 'Filnamn' },
                  { value: 'content', label: 'Innehåll' },
                  { value: 'headings', label: 'Rubriker' },
                ]}
              />
            </div>

            {content}
          </div>
        ) : (
          <div className="welcome">
            <div className="welcome-card">
              <span className="brand-mark lg">
                <Icon name="gem" size={30} />
              </span>
              <h1>Analysera din Obsidian-vault</h1>
              <p>
                Ordstatistik, länkgraf, PageRank och mappkoppling – beräknat i C++ på några millisekunder. Välj mappen som
                innehåller din vault för att börja.
              </p>
              <div className="welcome-actions">
                <button type="button" className="btn primary" onClick={() => setPickerOpen(true)} disabled={!appState}>
                  <Icon name="folderOpen" size={16} />
                  Välj vault…
                </button>
                {rootInput && (
                  <button type="button" className="btn" onClick={() => runScan(rootInput)} disabled={scanning}>
                    Skanna {rootInput}
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        <footer className="statusbar" role="status">
          <span className="status-text">{status}</span>
          {scan && (
            <span className="status-root" title={scan.root}>
              <bdi>{scan.root}</bdi>
            </span>
          )}
        </footer>
      </main>

      {scan && (
        <Inspector
          scan={scan}
          files={files}
          id={selected}
          onSelect={setSelected}
          onOpen={openFile}
          onCopy={copy}
          onClose={() => setSelected(null)}
        />
      )}

      {pickerOpen && appState && (
        <FolderPicker
          initial={rootInput}
          home={appState.home}
          roots={appState.roots}
          onClose={() => setPickerOpen(false)}
          onPick={(p) => {
            setPickerOpen(false);
            setRootInput(p);
            runScan(p);
          }}
        />
      )}
    </div>
  );
}

function TopNInput({ value, disabled, onChange }: { value: number; disabled: boolean; onChange: (n: number) => void }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const n = Number.parseInt(text, 10);
    if (Number.isFinite(n) && n >= 1) {
      if (n !== value) onChange(n);
    } else setText(String(value));
  };
  return (
    <div className="field-row">
      <label htmlFor="topn">Top N filer</label>
      <input
        id="topn"
        className="num-input"
        type="number"
        min={1}
        value={text}
        disabled={disabled}
        onChange={(e) => {
          setText(e.target.value);
          const n = Number.parseInt(e.target.value, 10);
          if (Number.isFinite(n) && n >= 1) onChange(n);
        }}
        onBlur={commit}
      />
    </div>
  );
}
