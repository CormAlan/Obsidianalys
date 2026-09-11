import { useMemo } from 'react';
import { Heatmap } from '../components/charts';
import { ViewHeader, ViewToggle, type ViewMode } from '../components/controls';
import { DataTable, type Column, type Sort } from '../components/DataTable';
import { fmt, fmtDec } from '../format';
import type { FolderRow } from '../types';

const DENSITY_DESC: Sort = { key: 'density', dir: 'desc' };
const EDGES_DESC: Sort = { key: 'edges', dir: 'desc' };
const MAX_PAIRS = 200;
const HEATMAP_K = 12;

interface PairRow {
  id: number;
  from: string;
  to: string;
  edges: number;
}

interface Props {
  folders: FolderRow[];
  pairs: [number, number, number][];
  rootName: string;
  onAnalyze: (id: number) => void;
  view: ViewMode;
  onView: (v: ViewMode) => void;
}

export function FolderLinksView({ folders, pairs, rootName, onAnalyze, view, onView }: Props) {
  const label = (f: FolderRow) => f.rel || `${rootName} (rot)`;

  const folderColumns = useMemo<Column<FolderRow>[]>(
    () => [
      { key: 'dir', label: 'Mapp (relativ)', width: 'minmax(0, 1fr)', value: (r) => r.rel, render: (r) => <span className="path-name">{r.rel || `${rootName} (rot)`}</span> },
      { key: 'count', label: 'Anteckn.', width: '6.5rem', numeric: true, value: (r) => r.count, render: (r) => fmt(r.count) },
      { key: 'internal', label: 'Interna länkar', width: '8rem', numeric: true, value: (r) => r.internal, render: (r) => fmt(r.internal) },
      { key: 'density', label: 'Densitet', width: '7rem', numeric: true, bar: true, value: (r) => r.density, render: (r) => fmtDec(r.density, 4) },
      { key: 'out', label: 'Ut', width: '5.5rem', numeric: true, value: (r) => r.out, render: (r) => fmt(r.out) },
      { key: 'in', label: 'In', width: '5.5rem', numeric: true, value: (r) => r.in, render: (r) => fmt(r.in) },
    ],
    [rootName],
  );

  const pairRows = useMemo<PairRow[]>(
    () =>
      pairs.slice(0, MAX_PAIRS).map(([a, b, n], id) => ({
        id,
        from: folders[a] ? label(folders[a]) : '?',
        to: folders[b] ? label(folders[b]) : '?',
        edges: n,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pairs, folders],
  );

  const pairColumns = useMemo<Column<PairRow>[]>(
    () => [
      { key: 'from', label: 'Från', width: 'minmax(0, 1fr)', value: (r) => r.from },
      { key: 'to', label: 'Till', width: 'minmax(0, 1fr)', value: (r) => r.to },
      { key: 'edges', label: 'Länkar', width: '7rem', numeric: true, bar: true, value: (r) => r.edges, render: (r) => fmt(r.edges) },
    ],
    [],
  );

  // Värmekarta över de mappar som har mest länktrafik (interna + ut + in)
  const heat = useMemo(() => {
    const top = [...folders].sort((a, b) => b.internal + b.out + b.in - (a.internal + a.out + a.in)).slice(0, HEATMAP_K);
    const index = new Map(top.map((f, i) => [f.id, i]));
    const matrix = top.map((f) => top.map((g) => (f.id === g.id ? f.internal : 0)));
    for (const [a, b, n] of pairs) {
      const i = index.get(a);
      const j = index.get(b);
      if (i !== undefined && j !== undefined) matrix[i][j] = n;
    }
    return { labels: top.map((f) => f.name), hints: top.map(label), matrix };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folders, pairs]);

  return (
    <section className="view">
      <ViewHeader
        title="Mappkoppling"
        description={`${fmt(folders.length)} mappar (räknar endast mappar som innehåller .md direkt). Densitet = interna länkar / n(n−1).`}
      >
        <ViewToggle value={view} onChange={onView} />
      </ViewHeader>
      {view === 'table' ? (
        <div className="split">
          <h3 className="split-title">Mappar</h3>
          <DataTable
            label="Mappar"
            rows={folders}
            columns={folderColumns}
            rowKey={(r) => r.id}
            defaultSort={DENSITY_DESC}
            onActivate={(r) => onAnalyze(r.id)}
          />
          <h3 className="split-title">
            Topp mapp → mapp-länkar{pairs.length > MAX_PAIRS && <span className="muted"> (visar {MAX_PAIRS} av {fmt(pairs.length)})</span>}
          </h3>
          <DataTable label="Mapp till mapp-länkar" rows={pairRows} columns={pairColumns} rowKey={(r) => r.id} defaultSort={EDGES_DESC} />
        </div>
      ) : (
        <div className="chart-grid single">
          <Heatmap
            title="Länkar mellan mappar"
            subtitle={`De ${heat.labels.length} mappar med mest länktrafik. Rad = källmapp, kolumn = målmapp.`}
            labels={heat.labels}
            hints={heat.hints}
            matrix={heat.matrix}
          />
        </div>
      )}
    </section>
  );
}
