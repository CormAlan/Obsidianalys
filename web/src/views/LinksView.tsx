import { useMemo } from 'react';
import { BarChart } from '../components/charts';
import { PathCell, Segmented, ViewHeader, ViewToggle, type ViewMode } from '../components/controls';
import { DataTable, type Column, type Sort } from '../components/DataTable';
import { fmt, fmtDec, stripMd } from '../format';
import type { FileRow, ScanStats } from '../types';

export type LinksMode = 'in' | 'out' | 'pr' | 'all';

// Samma sortering och sekundärnyckel som _populate_links() i Python-versionen
const ORDER: Record<Exclude<LinksMode, 'all'>, (a: FileRow, b: FileRow) => number> = {
  in: (a, b) => b.in - a.in || b.out - a.out,
  out: (a, b) => b.out - a.out || b.in - a.in,
  pr: (a, b) => b.pr - a.pr || b.in - a.in,
};
const DEFAULT_SORT: Record<LinksMode, Sort> = {
  in: { key: 'in', dir: 'desc' },
  out: { key: 'out', dir: 'desc' },
  pr: { key: 'pr', dir: 'desc' },
  all: { key: 'in', dir: 'desc' },
};
const CHART_N = 25;

interface Props {
  files: FileRow[];
  stats: ScanStats;
  limit: number;
  limitText: string;
  mode: LinksMode;
  onMode: (m: LinksMode) => void;
  selected: number | null;
  onSelect: (id: number) => void;
  onOpen: (id: number) => void;
  view: ViewMode;
  onView: (v: ViewMode) => void;
}

export function LinksView({ files, stats, limit, limitText, mode, onMode, selected, onSelect, onOpen, view, onView }: Props) {
  const rows = useMemo(() => (mode === 'all' ? files : [...files].sort(ORDER[mode]).slice(0, limit)), [files, mode, limit]);

  const columns = useMemo<Column<FileRow>[]>(
    () => [
      { key: 'path', label: 'Fil (relativ)', width: 'minmax(0, 1fr)', value: (r) => r.rel, render: (r) => <PathCell rel={r.rel} /> },
      { key: 'in', label: 'In', width: '6.5rem', numeric: true, bar: true, value: (r) => r.in, render: (r) => fmt(r.in) },
      { key: 'out', label: 'Ut', width: '6.5rem', numeric: true, value: (r) => r.out, render: (r) => fmt(r.out) },
      { key: 'pr', label: 'PageRank', width: '8rem', numeric: true, value: (r) => r.pr, render: (r) => fmtDec(r.pr, 6) },
    ],
    [],
  );

  const topIn = useMemo(
    () => [...files].sort(ORDER.in).slice(0, CHART_N).map((r) => ({ key: r.id, label: stripMd(r.name), hint: r.rel, value: r.in })),
    [files],
  );
  const topPr = useMemo(
    () => [...files].sort(ORDER.pr).slice(0, CHART_N).map((r) => ({ key: r.id, label: stripMd(r.name), hint: r.rel, value: r.pr })),
    [files],
  );

  return (
    <section className="view">
      <ViewHeader
        title="Länkar"
        description={
          mode === 'all'
            ? 'Alla anteckningar i länkgrafen.'
            : `${limitText} efter ${mode === 'in' ? 'inlänkar' : mode === 'out' ? 'utlänkar' : 'PageRank'}.`
        }
      >
        {view === 'table' && (
          <Segmented
            label="Visa"
            value={mode}
            onChange={onMode}
            options={[
              { value: 'in', label: 'In-länkar' },
              { value: 'out', label: 'Ut-länkar' },
              { value: 'pr', label: 'PageRank' },
              { value: 'all', label: 'Alla' },
            ]}
          />
        )}
        <ViewToggle value={view} onChange={onView} />
      </ViewHeader>

      <div className="chips">
        <span className="chip">
          Anteckningar <strong>{fmt(stats.files)}</strong>
        </span>
        <span className="chip">
          Länkar <strong>{fmt(stats.edges)}</strong>
        </span>
        <span className="chip" title="Anteckningar som ingen annan länkar till">
          Föräldralösa (in = 0) <strong>{fmt(stats.orphans)}</strong>
        </span>
        <span className="chip" title="Anteckningar utan utgående länkar">
          Återvändsgränder (ut = 0) <strong>{fmt(stats.deadEnds)}</strong>
        </span>
        <span className="chip" title="Länkmål som inte matchar någon fil">
          Olösta <strong>{fmt(stats.unresolved)}</strong>
        </span>
        <span className="chip" title="Länkmål som matchar flera filer med samma namn">
          Tvetydiga <strong>{fmt(stats.ambiguous)}</strong>
        </span>
      </div>

      {view === 'table' ? (
        <DataTable
          key={mode}
          label="Länkar"
          rows={rows}
          columns={columns}
          rowKey={(r) => r.id}
          defaultSort={DEFAULT_SORT[mode]}
          selectedKey={selected}
          onSelect={(r) => onSelect(r.id)}
          onActivate={(r) => onOpen(r.id)}
        />
      ) : (
        <div className="chart-grid">
          <BarChart title={`Topp ${CHART_N} – in-länkar`} subtitle="Mest refererade anteckningarna." data={topIn} selectedKey={selected} onSelect={onSelect} />
          <BarChart
            title={`Topp ${CHART_N} – PageRank`}
            subtitle="Strukturell betydelse i hela vaulten."
            data={topPr}
            format={(v) => fmtDec(v, 5)}
            selectedKey={selected}
            onSelect={onSelect}
          />
        </div>
      )}
    </section>
  );
}
