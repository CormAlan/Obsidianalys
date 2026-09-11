import type { ReactNode } from 'react';
import { Icon, type IconName } from './Icon';

export function Switch({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <button type="button" role="switch" aria-checked={checked} className="switch" onClick={() => onChange(!checked)}>
      <span className="switch-track" aria-hidden="true">
        <span className="switch-thumb" />
      </span>
      <span className="switch-text">
        <span>{label}</span>
        {hint && <code className="switch-hint">{hint}</code>}
      </span>
    </button>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  size,
}: {
  value: T;
  options: { value: T; label: string; icon?: IconName }[];
  onChange: (v: T) => void;
  label: string;
  size?: 'sm';
}) {
  return (
    <div className={`segmented${size ? ` segmented-${size}` : ''}`} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={value === o.value ? 'is-active' : ''}
          onClick={() => onChange(o.value)}
          title={o.icon ? o.label : undefined}
        >
          {o.icon && <Icon name={o.icon} size={16} />}
          <span className={o.icon ? 'segmented-label' : undefined}>{o.label}</span>
        </button>
      ))}
    </div>
  );
}

export function ViewHeader({ title, description, children }: { title: string; description?: ReactNode; children?: ReactNode }) {
  return (
    <div className="view-header">
      <div className="view-heading">
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {children && <div className="view-actions">{children}</div>}
    </div>
  );
}

export type ViewMode = 'table' | 'chart';

export function ViewToggle({ value, onChange }: { value: ViewMode; onChange: (v: ViewMode) => void }) {
  return (
    <Segmented
      label="Visningsläge"
      value={value}
      onChange={onChange}
      options={[
        { value: 'table', label: 'Tabell', icon: 'table' },
        { value: 'chart', label: 'Diagram', icon: 'chart' },
      ]}
    />
  );
}

export function PathCell({ rel, strip = true }: { rel: string; strip?: boolean }) {
  const i = rel.lastIndexOf('/');
  const dir = i < 0 ? '' : rel.slice(0, i + 1);
  let name = rel.slice(i + 1);
  if (strip && name.toLowerCase().endsWith('.md')) name = name.slice(0, -3);
  return (
    <span className="path-cell" title={rel}>
      {dir && <span className="path-dir">{dir}</span>}
      <span className="path-name">{name}</span>
    </span>
  );
}
