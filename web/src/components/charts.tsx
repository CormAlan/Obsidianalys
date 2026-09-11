import { Fragment, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { fmt, pct } from '../format';

// --- Tooltip: värdet först (starkt), etiketten sekundär. Endast text via React – aldrig innerHTML.

interface Tip {
  x: number;
  y: number;
  value: string;
  label: string;
  sub?: string;
  color?: string;
}

function Tooltip({ tip }: { tip: Tip | null }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: -9999, top: -9999 });
  useLayoutEffect(() => {
    if (!tip || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    let left = tip.x + 14;
    let top = tip.y + 14;
    if (left + r.width > window.innerWidth - 8) left = tip.x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = tip.y - r.height - 14;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [tip]);
  if (!tip) return null;
  return createPortal(
    <div className="tooltip" ref={ref} style={pos} role="tooltip">
      <strong>{tip.value}</strong>
      <span className="tooltip-label">
        {tip.color && <i className="tooltip-key" style={{ background: tip.color }} />}
        {tip.label}
      </span>
      {tip.sub && <span className="tooltip-sub">{tip.sub}</span>}
    </div>,
    document.body,
  );
}

function focusPoint(el: Element) {
  const r = el.getBoundingClientRect();
  return { x: r.left + r.width * 0.6, y: r.top + r.height / 2 };
}

// --- Liggande stapeldiagram (en serie): värde vid stapelns spets, tooltip med full sökväg.

export interface BarDatum {
  key: number;
  label: string;
  hint?: string;
  value: number;
}

export function BarChart({
  title,
  subtitle,
  data,
  format = fmt,
  selectedKey,
  onSelect,
}: {
  title: string;
  subtitle?: string;
  data: BarDatum[];
  format?: (v: number) => string;
  selectedKey?: number | null;
  onSelect?: (key: number) => void;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const max = data.reduce((m, d) => Math.max(m, d.value), 0) || 1;
  const show = (p: { x: number; y: number }, d: BarDatum) =>
    setTip({ ...p, value: format(d.value), label: d.label, sub: d.hint });

  return (
    <figure className="chart-card">
      <figcaption>
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </figcaption>
      {data.length === 0 ? (
        <div className="chart-empty">Ingen data.</div>
      ) : (
        <div className="bars" onPointerLeave={() => setTip(null)}>
          {data.map((d) => (
            <button
              key={d.key}
              type="button"
              className={`bar-row${d.key === selectedKey ? ' is-selected' : ''}`}
              onPointerMove={(e) => show({ x: e.clientX, y: e.clientY }, d)}
              onFocus={(e) => show(focusPoint(e.currentTarget), d)}
              onBlur={() => setTip(null)}
              onClick={() => onSelect?.(d.key)}
              aria-label={`${d.label}: ${format(d.value)}`}
            >
              <span className="bar-label">{d.label}</span>
              <span className="bar-track">
                <span className="bar" style={{ '--frac': d.value / max } as CSSProperties} />
                <span className="bar-value">{format(d.value)}</span>
              </span>
            </button>
          ))}
        </div>
      )}
      <Tooltip tip={tip} />
    </figure>
  );
}

// --- Munkdiagram för andel (högst sex segment: topp 5 + "Övrigt").

export interface Slice {
  key: string;
  label: string;
  value: number;
  color: string;
}

export function Donut({
  title,
  subtitle,
  slices,
  unit,
}: {
  title: string;
  subtitle?: string;
  slices: Slice[];
  unit: string;
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const total = slices.reduce((s, x) => s + x.value, 0);
  const C = 100;
  const R = 92;
  const r = 60;

  let angle = -Math.PI / 2;
  const arcs = slices.map((s) => {
    const sweep = total > 0 ? (s.value / total) * Math.PI * 2 : 0;
    const a0 = angle;
    angle += sweep;
    return { ...s, a0, a1: angle };
  });

  const pt = (rad: number, a: number) => `${C + rad * Math.cos(a)} ${C + rad * Math.sin(a)}`;
  const path = (a0: number, a1: number) => {
    if (a1 - a0 >= Math.PI * 2 - 1e-6) a1 = a0 + Math.PI * 2 - 1e-4; // hel ring
    const large = a1 - a0 > Math.PI ? 1 : 0;
    return `M ${pt(R, a0)} A ${R} ${R} 0 ${large} 1 ${pt(R, a1)} L ${pt(r, a1)} A ${r} ${r} 0 ${large} 0 ${pt(r, a0)} Z`;
  };

  const showSlice = (p: { x: number; y: number }, s: Slice) => {
    setActive(s.key);
    setTip({ ...p, value: `${fmt(s.value)} ${unit}`, label: s.label, sub: pct(total ? s.value / total : 0), color: s.color });
  };
  const hide = () => {
    setActive(null);
    setTip(null);
  };

  return (
    <figure className="chart-card">
      <figcaption>
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </figcaption>
      <div className="donut-wrap">
        <svg viewBox="0 0 200 200" className="donut" role="img" aria-label={title} onPointerLeave={hide}>
          {arcs.map((a) =>
            a.value > 0 ? (
              <path
                key={a.key}
                d={path(a.a0, a.a1)}
                fill={a.color}
                className={active && active !== a.key ? 'is-dim' : undefined}
                onPointerMove={(e) => showSlice({ x: e.clientX, y: e.clientY }, a)}
              />
            ) : null,
          )}
          <text x={C} y={C - 4} textAnchor="middle" className="donut-total">
            {fmt(total)}
          </text>
          <text x={C} y={C + 16} textAnchor="middle" className="donut-unit">
            {unit} totalt
          </text>
        </svg>
        <ul className="legend">
          {slices.map((s) => (
            <li
              key={s.key}
              className={active && active !== s.key ? 'is-dim' : undefined}
              tabIndex={0}
              onPointerEnter={(e) => showSlice({ x: e.clientX, y: e.clientY }, s)}
              onPointerLeave={hide}
              onFocus={(e) => showSlice(focusPoint(e.currentTarget), s)}
              onBlur={hide}
            >
              <i className="legend-swatch" style={{ background: s.color }} />
              <span className="legend-label" title={s.label}>
                {s.label}
              </span>
              <span className="legend-value">{fmt(s.value)}</span>
              <span className="legend-pct">{pct(total ? s.value / total : 0)}</span>
            </li>
          ))}
        </ul>
      </div>
      <Tooltip tip={tip} />
    </figure>
  );
}

// --- Värmekarta för mapp → mapp-länkar (en färgton, ljus → mörk; kvadratrotsskala i 7 steg).

export function Heatmap({
  title,
  subtitle,
  labels,
  hints,
  matrix,
}: {
  title: string;
  subtitle?: string;
  labels: string[];
  hints: string[];
  matrix: number[][];
}) {
  const [tip, setTip] = useState<Tip | null>(null);
  const max = Math.max(1, ...matrix.flat());
  const bin = (v: number) => (v <= 0 ? 0 : Math.min(7, Math.max(1, Math.ceil(Math.sqrt(v / max) * 7))));
  const show = (p: { x: number; y: number }, i: number, j: number) =>
    setTip({
      ...p,
      value: `${fmt(matrix[i][j])} länk${matrix[i][j] === 1 ? '' : 'ar'}`,
      label: i === j ? `Inom ${labels[i]}` : `${labels[i]} → ${labels[j]}`,
      sub: i === j ? hints[i] : `${hints[i]} → ${hints[j]}`,
    });

  return (
    <figure className="chart-card heatmap-card">
      <figcaption>
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </figcaption>
      <div className="heatmap-scroll">
        <div className="heatmap" style={{ '--k': labels.length } as CSSProperties} onPointerLeave={() => setTip(null)}>
          <div className="hm-corner">från ↓ till →</div>
          {labels.map((_, j) => (
            <div key={`c${j}`} className="hm-col" title={hints[j]}>
              {j + 1}
            </div>
          ))}
          {matrix.map((row, i) => (
            <Fragment key={i}>
              <div className="hm-row" title={hints[i]}>
                <span className="hm-num">{i + 1}</span>
                <span className="hm-name">{labels[i]}</span>
              </div>
              {row.map((v, j) => (
                <button
                  key={j}
                  type="button"
                  className={`hm-cell${i === j ? ' is-diag' : ''}`}
                  data-bin={bin(v)}
                  aria-label={`${labels[i]} till ${labels[j]}: ${fmt(v)} länkar`}
                  onPointerMove={(e) => show({ x: e.clientX, y: e.clientY }, i, j)}
                  onFocus={(e) => show(focusPoint(e.currentTarget), i, j)}
                  onBlur={() => setTip(null)}
                />
              ))}
            </Fragment>
          ))}
        </div>
      </div>
      <div className="hm-legend">
        <span>1</span>
        <span className="hm-ramp" aria-hidden="true">
          {[1, 2, 3, 4, 5, 6, 7].map((b) => (
            <i key={b} data-bin={b} />
          ))}
        </span>
        <span>{fmt(max)} länkar</span>
        <span className="hm-legend-note">Kvadratrotsskala · prick = länkar inom samma mapp</span>
      </div>
      <Tooltip tip={tip} />
    </figure>
  );
}
