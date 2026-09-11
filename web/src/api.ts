import type { AppState, Config, DirListing, NoteDetail, Scan, SearchMode, SearchResult } from './types';

// Servern stoppar in en engångstoken i index.html; utan den (Vite dev) lägger proxyn till "dev".
const token = document.querySelector<HTMLMetaElement>('meta[name="obsidianalys-token"]')?.content ?? 'dev';

type Params = Record<string, string | number | boolean | undefined>;

async function call<T>(method: 'GET' | 'POST', path: string, params: Params = {}): Promise<T> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) qs.set(k, typeof v === 'boolean' ? (v ? '1' : '0') : String(v));
  }
  const query = qs.toString();
  const res = await fetch(`/api/${path}${query ? `?${query}` : ''}`, {
    method,
    headers: { 'X-Obsidianalys-Token': token },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
  return body as T;
}

export const api = {
  state: () => call<AppState>('GET', 'state'),
  scan: (root: string) => call<Scan>('POST', 'scan', { root }),
  note: (id: number, gen: number) => call<NoteDetail>('GET', 'note', { id, gen }),
  search: (q: string, mode: SearchMode) => call<SearchResult>('GET', 'search', { q, mode }),
  open: (id: number, gen: number) => call<{ ok: boolean }>('POST', 'open', { id, gen }),
  ls: (path?: string) => call<DirListing>('GET', 'ls', { path }),
  config: (patch: { topN?: number; eq?: boolean; code?: boolean; theme?: string }) =>
    call<Config>('POST', 'config', patch),
  quit: () => call<{ ok: boolean }>('POST', 'quit'),
};
