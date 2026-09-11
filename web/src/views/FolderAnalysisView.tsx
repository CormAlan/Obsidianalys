import { useMemo } from 'react';
import { BarChart } from '../components/charts';
import { ViewHeader, ViewToggle, type ViewMode } from '../components/controls';
import { DataTable, type Column, type Sort } from '../components/DataTable';
import { collator, fmt, fmtDec, stripMd } from '../format';
import type { FileRow, FolderRow } from '../types';

const WORDS_DESC: Sort = { key: 'words', dir: 'desc' };

interface Props {
  folders: FolderRow[];
  files: FileRow[];
  rootName: string;
  folderId: number | null;
  onFolder: (id: number) => void;
  selected: number | null;
  onSelect: (id: number) => void;
  onOpen: (id: number) => void;
  view: ViewMode;
  onView: (v: ViewMode) => void;
}

export function FolderAnalysisView({ folders, files, rootName, folderId, onFolder, selected, onSelect, onOpen, view, onView }: Props) {
  const folder = folderId != null ? folders[folderId] : undefined;
  const rows = useMemo(() => (folderId == null ? [] : files.filter((f) => f.folder === folderId)), [files, folderId]);
  const options = useMemo(() => [...folders].sort((a, b) => collator.compare(a.rel, b.rel)), [folders]);

  const columns = useMemo<Column<FileRow>[]>(
    () => [
      { key: 'name', label: 'Fil (i mappen)', width: 'minmax(0, 1fr)', value: (r) => r.name, render: (r) => <span className="path-name">{stripMd(r.name)}</span> },
      { key: 'words', label: 'Ord', width: '10rem', numeric: true, bar: true, value: (r) => r.words, render: (r) => fmt(r.words) },
      { key: 'in', label: 'In', width: '5rem', numeric: true, value: (r) => r.in, render: (r) => fmt(r.in) },
      { key: 'out', label: 'Ut', width: '5rem', numeric: true, value: (r) => r.out, render: (r) => fmt(r.out) },
    ],
    [],
  );

  const chart = useMemo(
    () => [...rows].sort((a, b) => b.words - a.words).slice(0, 40).map((r) => ({ key: r.id, label: stripMd(r.name), hint: r.rel, value: r.words })),
    [rows],
  );

  return (
    <section className="view">
      <ViewHeader title="Mappanalys" description="Filer som ligger direkt i vald mapp.">
        <select
          className="select"
          value={folderId ?? ''}
          onChange={(e) => onFolder(Number(e.target.value))}
          aria-label="Vald mapp"
        >
          {folderId == null && <option value="">Välj mapp…</option>}
          {options.map((f) => (
            <option key={f.id} value={f.id}>
              {f.rel || `${rootName} (rot)`}
            </option>
          ))}
        </select>
        <ViewToggle value={view} onChange={onView} />
      </ViewHeader>

      {!folder ? (
        <div className="placeholder">Välj en mapp ovan – eller dubbelklicka på en mapp under Mappar.</div>
      ) : (
        <>
          <div className="chips">
            <span className="chip">
              Filer <strong>{fmt(folder.count)}</strong>
            </span>
            <span className="chip">
              Ord <strong>{fmt(folder.words)}</strong>
            </span>
            <span className="chip">
              Snitt <strong>{fmt(folder.avg)}</strong> ord/fil
            </span>
            <span className="chip">
              Interna länkar <strong>{fmt(folder.internal)}</strong>
            </span>
            <span className="chip">
              Densitet <strong>{fmtDec(folder.density, 4)}</strong>
            </span>
          </div>
          {view === 'table' ? (
            <DataTable
              label={`Filer i ${folder.name}`}
              rows={rows}
              columns={columns}
              rowKey={(r) => r.id}
              defaultSort={WORDS_DESC}
              selectedKey={selected}
              onSelect={(r) => onSelect(r.id)}
              onActivate={(r) => onOpen(r.id)}
            />
          ) : (
            <div className="chart-grid single">
              <BarChart
                title={`${folder.name} – antal ord per fil`}
                subtitle={rows.length > 40 ? `Visar de 40 största av ${fmt(rows.length)}.` : undefined}
                data={chart}
                selectedKey={selected}
                onSelect={onSelect}
              />
            </div>
          )}
        </>
      )}
    </section>
  );
}
