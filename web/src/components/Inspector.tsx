import { useEffect, useState, type CSSProperties } from 'react';
import { api } from '../api';
import { collator, fmt, fmtDec, splitRel, stripMd } from '../format';
import type { FileRow, NoteDetail, Scan } from '../types';
import { Icon, type IconName } from './Icon';

interface Props {
  scan: Scan;
  files: FileRow[];
  id: number | null;
  onSelect: (id: number) => void;
  onOpen: (id: number) => void;
  onCopy: (text: string, what: string) => void;
  onClose: () => void;
}

// Högerpanelen: rubriker för vald fil (som i Python-versionen) plus in- och utlänkar.
export function Inspector({ scan, files, id, onSelect, onOpen, onCopy, onClose }: Props) {
  const [detail, setDetail] = useState<NoteDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const gen = scan.generation;

  useEffect(() => {
    setDetail(null);
    setError(null);
    if (id == null) return;
    let alive = true;
    api
      .note(id, gen)
      .then((d) => alive && setDetail(d))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [id, gen]);

  const f = id != null ? files[id] : undefined;
  if (!f) {
    return (
      <aside className="inspector" aria-label="Vald fil">
        <div className="inspector-empty">
          <Icon name="file" size={28} />
          <p>Markera en fil i någon tabell för att se dess rubriker och länkar.</p>
        </div>
      </aside>
    );
  }

  const { dir } = splitRel(f.rel);
  const byName = (ids: number[]) =>
    [...ids].sort((a, b) => collator.compare(files[a]?.name ?? '', files[b]?.name ?? ''));
  const headingLines =
    detail?.headings
      .filter(([, t]) => t.trim())
      .map(([l, t]) => `${'#'.repeat(Math.min(6, Math.max(1, l)))} ${t.trim()}`) ?? [];

  return (
    <aside className="inspector" aria-label="Vald fil">
      <header className="insp-head">
        <div className="insp-head-row">
          <span className="eyebrow">Vald fil</span>
          <button type="button" className="icon-btn insp-close" onClick={onClose} aria-label="Stäng panelen">
            <Icon name="x" />
          </button>
        </div>
        <h2>{stripMd(f.name)}</h2>
        <div className="insp-path">{dir || `${scan.rootName}/`}</div>
      </header>

      <div className="insp-stats">
        <div className="insp-stat">
          <b>{fmt(f.words)}</b>
          <span>ord</span>
        </div>
        <div className="insp-stat">
          <b>{fmt(f.in)}</b>
          <span>in</span>
        </div>
        <div className="insp-stat">
          <b>{fmt(f.out)}</b>
          <span>ut</span>
        </div>
        <div className="insp-stat" title={`PageRank ${f.pr} – ${fmtDec(f.pr * files.length, 2)} gånger snittet`}>
          <b>{fmtDec(f.pr * files.length, 2)}×</b>
          <span>PageRank</span>
        </div>
      </div>

      <div className="insp-actions">
        <button type="button" className="btn sm primary" onClick={() => onOpen(f.id)}>
          <Icon name="external" size={15} />
          Öppna fil
        </button>
        {detail && (
          <a className="btn sm" href={`obsidian://open?path=${encodeURIComponent(detail.abs)}`}>
            <Icon name="gem" size={15} />
            Öppna i Obsidian
          </a>
        )}
        <button type="button" className="btn sm" onClick={() => onCopy(f.rel, `sökväg: ${f.rel}`)}>
          <Icon name="copy" size={15} />
          Kopiera sökväg
        </button>
      </div>

      {error && <div className="insp-error">{error}</div>}

      <section className="insp-section">
        <h3>
          Rubriker <span className="count-badge">{fmt(f.headings)}</span>
          {headingLines.length > 0 && (
            <button
              type="button"
              className="btn ghost sm push"
              onClick={() => onCopy(headingLines.join('\n'), `${headingLines.length} rubriker från ${f.name}`)}
            >
              <Icon name="copy" size={14} />
              Kopiera alla
            </button>
          )}
        </h3>
        {!detail ? (
          <p className="muted">Laddar…</p>
        ) : detail.headings.length === 0 ? (
          <p className="muted">Inga rubriker hittades.</p>
        ) : (
          <ol className="outline">
            {detail.headings.map(([lvl, text], i) => (
              <li key={i} data-lvl={lvl} style={{ '--lvl': lvl } as CSSProperties}>
                <span className="lvl">H{lvl}</span>
                <span>{text ? prettyHeading(text) : <em className="muted">(tom)</em>}</span>
              </li>
            ))}
          </ol>
        )}
      </section>

      <LinkSection
        title="Länkar hit"
        icon="arrowIn"
        ids={detail ? byName(detail.in) : null}
        files={files}
        onSelect={onSelect}
        empty="Ingen anteckning länkar hit."
      />
      <LinkSection
        title="Länkar härifrån"
        icon="arrowOut"
        ids={detail ? byName(detail.out) : null}
        files={files}
        onSelect={onSelect}
        empty="Inga utgående länkar."
      />
    </aside>
  );
}

const PREVIEW = 60;

// [[mål|alias]] → alias, [[mål#rubrik]] → mål – bara för visning; kopiering behåller råtexten
const prettyHeading = (text: string) =>
  text.replace(/!?\[\[([^[\]|]+)(?:\|([^[\]]+))?\]\]/g, (_, target: string, alias?: string) =>
    (alias ?? target.split(/[#^]/)[0]).trim(),
  );

function LinkSection({
  title,
  icon,
  ids,
  files,
  onSelect,
  empty,
}: {
  title: string;
  icon: IconName;
  ids: number[] | null;
  files: FileRow[];
  onSelect: (id: number) => void;
  empty: string;
}) {
  const [all, setAll] = useState(false);
  const shown = ids && !all ? ids.slice(0, PREVIEW) : ids;
  return (
    <section className="insp-section">
      <h3>
        <Icon name={icon} size={15} />
        {title} {ids && <span className="count-badge">{fmt(ids.length)}</span>}
      </h3>
      {!shown ? (
        <p className="muted">Laddar…</p>
      ) : shown.length === 0 ? (
        <p className="muted">{empty}</p>
      ) : (
        <ul className="linklist">
          {shown.map((i) => (
            <li key={i}>
              <button type="button" onClick={() => onSelect(i)} title={files[i]?.rel}>
                {stripMd(files[i]?.name ?? '?')}
              </button>
            </li>
          ))}
        </ul>
      )}
      {ids && ids.length > PREVIEW && (
        <button type="button" className="link-btn more" onClick={() => setAll(!all)}>
          {all ? 'Visa färre' : `Visa alla ${fmt(ids.length)}`}
        </button>
      )}
    </section>
  );
}
