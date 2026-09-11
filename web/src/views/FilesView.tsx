import { useMemo } from 'react';
import { BarChart } from '../components/charts';
import { PathCell, ViewHeader, ViewToggle, type ViewMode } from '../components/controls';
import { DataTable, type Column, type Sort } from '../components/DataTable';
import { fmt, MODE_LABEL, stripMd } from '../format';
import type { FileRow, SearchMode } from '../types';

const WORDS_DESC: Sort = { key: 'words', dir: 'desc' };
const CHART_MAX = 40;

interface Props {
  rows: FileRow[];
  search: { q: string; mode: SearchMode; count: number } | null;
  onClearSearch: () => void;
  limitText: string;
  selected: number | null;
  onSelect: (id: number) => void;
  onOpen: (id: number) => void;
  view: ViewMode;
  onView: (v: ViewMode) => void;
}

export function FilesView({ rows, search, onClearSearch, limitText, selected, onSelect, onOpen, view, onView }: Props) {
  const columns = useMemo<Column<FileRow>[]>(
    () => [
      { key: 'path', label: 'Fil (relativ)', width: 'minmax(0, 1fr)', value: (r) => r.rel, render: (r) => <PathCell rel={r.rel} /> },
      { key: 'words', label: 'Ord', width: '11rem', numeric: true, bar: true, value: (r) => r.words, render: (r) => fmt(r.words) },
    ],
    [],
  );

  const chart = useMemo(
    () =>
      [...rows]
        .sort((a, b) => b.words - a.words)
        .slice(0, CHART_MAX)
        .map((r) => ({ key: r.id, label: stripMd(r.name), hint: r.rel, value: r.words })),
    [rows],
  );

  return (
    <section className="view">
      <ViewHeader
        title={search ? 'Sökresultat' : 'Toppfiler'}
        description={
          search ? (
            <>
              {fmt(search.count)} träff{search.count === 1 ? '' : 'ar'} för <mark>”{search.q}”</mark> i{' '}
              {MODE_LABEL[search.mode].toLowerCase()} ·{' '}
              <button type="button" className="link-btn" onClick={onClearSearch}>
                Rensa sökningen
              </button>
            </>
          ) : (
            `${limitText} sorterade efter antal ord. Dubbelklicka eller tryck Enter för att öppna.`
          )
        }
      >
        <ViewToggle value={view} onChange={onView} />
      </ViewHeader>
      {view === 'table' ? (
        <DataTable
          label="Filer"
          rows={rows}
          columns={columns}
          rowKey={(r) => r.id}
          defaultSort={WORDS_DESC}
          selectedKey={selected}
          onSelect={(r) => onSelect(r.id)}
          onActivate={(r) => onOpen(r.id)}
          emptyText={search ? 'Inga träffar.' : 'Inga filer.'}
        />
      ) : (
        <div className="chart-grid single">
          <BarChart
            title={search ? 'Träffar – antal ord' : `${limitText} – antal ord`}
            subtitle={rows.length > CHART_MAX ? `Visar de ${CHART_MAX} största av ${fmt(rows.length)}.` : undefined}
            data={chart}
            selectedKey={selected}
            onSelect={onSelect}
          />
        </div>
      )}
    </section>
  );
}
