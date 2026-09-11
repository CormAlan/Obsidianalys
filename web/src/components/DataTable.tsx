import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { collator } from '../format';

export interface Sort {
  key: string;
  dir: 'asc' | 'desc';
}

export interface Column<T> {
  key: string;
  label: string;
  width: string; // grid-spår, t.ex. "minmax(0,1fr)" eller "8rem"
  numeric?: boolean;
  value: (row: T) => number | string; // sorteringsnyckel
  render?: (row: T) => ReactNode;
  bar?: boolean; // tunn stapel i cellen, proportionell mot värdet
  title?: string;
}

interface Props<T> {
  label: string;
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => number;
  defaultSort: Sort;
  selectedKey?: number | null;
  onSelect?: (row: T) => void;
  onActivate?: (row: T) => void;
  emptyText?: string;
}

const ROW_H = 34;
const HEAD_H = 36;
const OVERSCAN = 10;

// Virtualiserad, sorterbar tabell. '#'-kolumnen är låst och numreras om efter
// sortering – precis som i Python-versionen – men bara synliga rader renderas.
export function DataTable<T>({
  label,
  rows,
  columns,
  rowKey,
  defaultSort,
  selectedKey,
  onSelect,
  onActivate,
  emptyText = 'Inga rader.',
}: Props<T>) {
  const [sort, setSort] = useState<Sort>(defaultSort);
  useEffect(() => setSort(defaultSort), [defaultSort]);

  const sorted = useMemo(() => {
    const col = columns.find((c) => c.key === sort.key);
    if (!col) return rows;
    const sign = sort.dir === 'asc' ? 1 : -1;
    return rows
      .map((row, i) => ({ row, i, v: col.value(row) }))
      .sort((a, b) => {
        const d =
          typeof a.v === 'number' && typeof b.v === 'number' ? a.v - b.v : collator.compare(String(a.v), String(b.v));
        return d !== 0 ? d * sign : a.i - b.i;
      })
      .map((x) => x.row);
  }, [rows, columns, sort]);

  const maxes = useMemo(
    () => columns.map((c) => (c.bar ? rows.reduce((m, r) => Math.max(m, Number(c.value(r))), 0) : 0)),
    [rows, columns],
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewH, setViewH] = useState(800);

  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewH(el.clientHeight));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
    setScrollTop(0);
  }, [rows]);

  const first = Math.max(0, Math.floor((scrollTop - HEAD_H) / ROW_H) - OVERSCAN);
  const last = Math.min(sorted.length, Math.ceil((scrollTop + viewH) / ROW_H) + OVERSCAN);
  const selIndex = useMemo(
    () => (selectedKey == null ? -1 : sorted.findIndex((r) => rowKey(r) === selectedKey)),
    [sorted, selectedKey, rowKey],
  );

  const reveal = (i: number) => {
    const el = scrollRef.current;
    if (!el) return;
    const top = HEAD_H + i * ROW_H;
    if (top - HEAD_H < el.scrollTop) el.scrollTop = top - HEAD_H;
    else if (top + ROW_H > el.scrollTop + el.clientHeight) el.scrollTop = top + ROW_H - el.clientHeight;
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (!sorted.length) return;
    const page = Math.max(1, Math.floor((viewH - HEAD_H) / ROW_H) - 1);
    let next: number;
    switch (e.key) {
      case 'ArrowDown':
        next = Math.min(sorted.length - 1, selIndex + 1);
        break;
      case 'ArrowUp':
        next = Math.max(0, selIndex - 1);
        break;
      case 'PageDown':
        next = Math.min(sorted.length - 1, selIndex + page);
        break;
      case 'PageUp':
        next = Math.max(0, selIndex - page);
        break;
      case 'Home':
        next = 0;
        break;
      case 'End':
        next = sorted.length - 1;
        break;
      case 'Enter':
        if (selIndex >= 0) onActivate?.(sorted[selIndex]);
        return;
      default:
        return;
    }
    e.preventDefault();
    onSelect?.(sorted[next]);
    reveal(next);
  };

  const toggleSort = (c: Column<T>) =>
    setSort((s) =>
      s.key === c.key ? { key: c.key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key: c.key, dir: c.numeric ? 'desc' : 'asc' },
    );

  const template = ['3.5rem', ...columns.map((c) => c.width)].join(' ');

  return (
    <div
      className="table"
      role="grid"
      aria-label={label}
      aria-rowcount={sorted.length + 1}
      tabIndex={0}
      ref={scrollRef}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
      onKeyDown={onKeyDown}
      style={{ '--cols': template } as CSSProperties}
    >
      <div className="table-head" role="row">
        <div className="th th-rank" role="columnheader">
          #
        </div>
        {columns.map((c) => (
          <div
            key={c.key}
            role="columnheader"
            aria-sort={sort.key === c.key ? (sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'}
          >
            <button
              type="button"
              className={`th${c.numeric ? ' is-num' : ''}${sort.key === c.key ? ' is-sorted' : ''}`}
              onClick={() => toggleSort(c)}
              title={c.title ?? `Sortera på ${c.label.toLowerCase()}`}
            >
              <span>{c.label}</span>
              <span className="sort-caret" aria-hidden="true">
                {sort.key === c.key ? (sort.dir === 'asc' ? '↑' : '↓') : ''}
              </span>
            </button>
          </div>
        ))}
      </div>
      {sorted.length === 0 ? (
        <div className="table-empty">{emptyText}</div>
      ) : (
        <div className="table-body" style={{ height: sorted.length * ROW_H }}>
          {sorted.slice(first, last).map((row, k) => {
            const i = first + k;
            const key = rowKey(row);
            const isSel = key === selectedKey;
            return (
              <div
                key={key}
                role="row"
                aria-rowindex={i + 2}
                aria-selected={isSel}
                className={`tr${isSel ? ' is-selected' : ''}`}
                style={{ transform: `translateY(${i * ROW_H}px)` }}
                onClick={() => onSelect?.(row)}
                onDoubleClick={() => onActivate?.(row)}
              >
                <div className="td td-rank" role="gridcell">
                  {i + 1}
                </div>
                {columns.map((c, ci) => (
                  <div key={c.key} role="gridcell" className={`td${c.numeric ? ' is-num' : ''}`}>
                    {c.bar && maxes[ci] > 0 && (
                      <span
                        className="cell-bar"
                        style={{ '--frac': Number(c.value(row)) / maxes[ci] } as CSSProperties}
                        aria-hidden="true"
                      />
                    )}
                    <span className="td-content">{c.render ? c.render(row) : c.value(row)}</span>
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
