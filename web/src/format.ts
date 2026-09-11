const integer = new Intl.NumberFormat('sv-SE');
const compactFmt = new Intl.NumberFormat('sv-SE', { notation: 'compact', maximumFractionDigits: 1 });

export const fmt = (n: number) => integer.format(Math.round(n));

export const fmtDec = (n: number, digits: number) =>
  n.toLocaleString('sv-SE', { minimumFractionDigits: digits, maximumFractionDigits: digits });

export const compact = (n: number) => compactFmt.format(n);

export const fmtMs = (ms: number) =>
  ms < 10 ? `${fmtDec(ms, 2)} ms` : ms < 1000 ? `${fmtDec(ms, 1)} ms` : `${fmtDec(ms / 1000, 2)} s`;

export const pct = (x: number) => `${fmtDec(x * 100, x < 0.1 ? 1 : 0)} %`;

// Index i NoteJson.w – samma som analysis::kWordVariants i C++.
export const wordIndex = (includeEquations: boolean, includeCode: boolean) =>
  (includeEquations ? 2 : 0) + (includeCode ? 1 : 0);

export const stripMd = (name: string) => (name.toLowerCase().endsWith('.md') ? name.slice(0, -3) : name);

export function splitRel(rel: string): { dir: string; name: string } {
  const i = rel.lastIndexOf('/');
  return { dir: i < 0 ? '' : rel.slice(0, i + 1), name: stripMd(rel.slice(i + 1)) };
}

export const collator = new Intl.Collator('sv', { numeric: true, sensitivity: 'base' });

export const MODE_LABEL = { name: 'Filnamn', content: 'Innehåll', headings: 'Rubriker' } as const;
