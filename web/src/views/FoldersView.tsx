import { useMemo, useState } from 'react';
import { BarChart, Donut, type Slice } from '../components/charts';
import { ViewHeader, ViewToggle, type ViewMode } from '../components/controls';
import { DataTable, type Column, type Sort } from '../components/DataTable';
import { Icon } from '../components/Icon';
import { fmt } from '../format';
import type { FolderRow } from '../types';

const WORDS_DESC: Sort = { key: 'words', dir: 'desc' };
// Kategoriska platser 1–5 i validerad ordning + neutral grå för "Övrigt"
const SLICE_COLORS = ['var(--cat-1)', 'var(--cat-2)', 'var(--cat-3)', 'var(--cat-4)', 'var(--cat-5)'];

interface Props {
  folders: FolderRow[];
  rootName: string;
  onAnalyze: (id: number) => void;
  view: ViewMode;
  onView: (v: ViewMode) => void;
}

export function FoldersView({ folders, rootName, onAnalyze, view, onView }: Props) {
  const [selected, setSelected] = useState<number | null>(null);

  const columns = useMemo<Column<FolderRow>[]>(
    () => [
      {
        key: 'name',
        label: 'Mappnamn',
        width: 'minmax(0, 1fr)',
        value: (r) => r.name,
        render: (r) => (
          <span className="path-cell" title={r.rel || rootName}>
            <span className="path-name">{r.name}</span>
            {r.rel.includes('/') && <span className="path-dir"> · {r.rel.slice(0, r.rel.lastIndexOf('/'))}</span>}
            {!r.rel && <span className="path-dir"> · rotmappen</span>}
          </span>
        ),
      },
      { key: 'words', label: 'Totalt ord', width: '10rem', numeric: true, bar: true, value: (r) => r.words, render: (r) => fmt(r.words) },
      { key: 'avg', label: 'Snitt ord/fil', width: '8rem', numeric: true, value: (r) => r.avg, render: (r) => fmt(r.avg) },
      { key: 'count', label: 'Antal filer', width: '7rem', numeric: true, value: (r) => r.count, render: (r) => fmt(r.count) },
    ],
    [rootName],
  );

  const byWords = useMemo(() => [...folders].sort((a, b) => b.words - a.words), [folders]);
  const bars = byWords.slice(0, 15).map((f) => ({ key: f.id, label: f.name, hint: f.rel || rootName, value: f.words }));
  const slices = useMemo<Slice[]>(() => {
    const top = byWords.slice(0, 5).map((f, i) => ({ key: String(f.id), label: f.name, value: f.words, color: SLICE_COLORS[i] }));
    const rest = byWords.slice(5).reduce((s, f) => s + f.words, 0);
    return rest > 0 ? [...top, { key: 'rest', label: `Övrigt (${byWords.length - 5} mappar)`, value: rest, color: 'var(--other)' }] : top;
  }, [byWords]);

  return (
    <section className="view">
      <ViewHeader title="Mappar (med filer)" description="Varje mapp som direkt innehåller minst en .md-fil. Dubbelklicka för mappanalys.">
        {view === 'table' && (
          <button type="button" className="btn" disabled={selected == null} onClick={() => selected != null && onAnalyze(selected)}>
            <Icon name="folderOpen" size={16} />
            Analysera mapp
          </button>
        )}
        <ViewToggle value={view} onChange={onView} />
      </ViewHeader>
      {view === 'table' ? (
        <DataTable
          label="Mappar"
          rows={folders}
          columns={columns}
          rowKey={(r) => r.id}
          defaultSort={WORDS_DESC}
          selectedKey={selected}
          onSelect={(r) => setSelected(r.id)}
          onActivate={(r) => onAnalyze(r.id)}
        />
      ) : (
        <div className="chart-grid">
          <BarChart title="Största mapparna – antal ord" subtitle="Topp 15. Klicka för mappanalys." data={bars} onSelect={onAnalyze} />
          <Donut title="Andel ord per mapp" subtitle="Topp 5 mappar, resten samlat som Övrigt." slices={slices} unit="ord" />
        </div>
      )}
    </section>
  );
}
