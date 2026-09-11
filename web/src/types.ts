// Speglar JSON från C++-servern (core/src/report.cpp och server.cpp).

export interface Timings {
  walk: number;
  parse: number;
  graph: number;
  pagerank: number;
  folders: number;
  total: number;
}

export interface ScanStats {
  files: number;
  folders: number;
  edges: number;
  unresolved: number;
  ambiguous: number;
  orphans: number;
  deadEnds: number;
}

export interface NoteJson {
  rel: string;
  name: string;
  f: number; // mapp-id
  w: [number, number, number, number]; // ord per variant, se wordIndex()
  in: number;
  out: number;
  pr: number;
  h: number; // antal rubriker
}

export interface FolderJson {
  rel: string;
  name: string;
  n: number;
  internal: number;
  density: number;
  out: number;
  in: number;
}

export interface Scan {
  generation: number;
  root: string;
  rootName: string;
  timings: Timings;
  stats: ScanStats;
  notes: NoteJson[];
  folders: FolderJson[];
  pairs: [number, number, number][]; // [från, till, länkar], fallande
}

export interface NoteDetail {
  id: number;
  abs: string;
  headings: [number, string][];
  out: number[];
  in: number[];
}

export type Theme = 'system' | 'light' | 'dark';

export interface Config {
  root: string;
  topN: number;
  includeEquations: boolean;
  includeCode: boolean;
  theme: Theme;
}

export interface AppState {
  version: string;
  config: Config;
  home: string;
  roots: string[];
  configPath: string;
}

export interface DirListing {
  path: string;
  parent: string | null;
  isVault: boolean;
  dirs: { name: string; path: string; vault: boolean }[];
}

export type SearchMode = 'name' | 'content' | 'headings';

export interface SearchResult {
  ids: number[];
  ms: number;
  generation: number;
}

// Härledda rader för tabellerna
export interface FileRow {
  id: number;
  rel: string;
  name: string;
  folder: number;
  words: number;
  in: number;
  out: number;
  pr: number;
  headings: number;
}

export interface FolderRow {
  id: number;
  rel: string;
  name: string;
  words: number;
  count: number;
  avg: number;
  internal: number;
  density: number;
  out: number;
  in: number;
}
