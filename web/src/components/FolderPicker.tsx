import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { DirListing } from '../types';
import { Icon } from './Icon';

// Mappväljare som bläddrar via servern (webbläsare kan inte ge oss absoluta sökvägar).
export function FolderPicker({
  initial,
  home,
  roots,
  onPick,
  onClose,
}: {
  initial: string;
  home: string;
  roots: string[];
  onPick: (path: string) => void;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<DirListing | null>(null);
  const [pathInput, setPathInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const load = (path?: string, fallback = false): Promise<void> =>
    api
      .ls(path)
      .then((l) => {
        setListing(l);
        setPathInput(l.path);
        setError(null);
        listRef.current?.scrollTo(0, 0);
      })
      .catch((e: Error) => (fallback ? load(undefined) : setError(e.message)));

  useEffect(() => {
    load(initial || undefined, true);
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="picker-title">
        <header className="modal-head">
          <h2 id="picker-title">Välj rotmapp för Obsidian-anteckningar</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Stäng">
            <Icon name="x" />
          </button>
        </header>

        <form
          className="picker-path"
          onSubmit={(e) => {
            e.preventDefault();
            load(pathInput);
          }}
        >
          <button
            type="button"
            className="icon-btn"
            onClick={() => listing?.parent && load(listing.parent)}
            disabled={!listing?.parent}
            title="Upp en nivå"
            aria-label="Upp en nivå"
          >
            <Icon name="up" />
          </button>
          <input value={pathInput} onChange={(e) => setPathInput(e.target.value)} spellCheck={false} aria-label="Sökväg" />
          <button type="submit" className="btn sm">
            Gå
          </button>
        </form>

        <div className="picker-shortcuts">
          <button type="button" className="btn ghost sm" onClick={() => load(home)}>
            <Icon name="home" size={15} />
            Hem
          </button>
          {roots.map((r) => (
            <button key={r} type="button" className="btn ghost sm" onClick={() => load(r)}>
              {r}
            </button>
          ))}
        </div>

        {error && <div className="picker-error">{error}</div>}

        <ul className="picker-list" ref={listRef}>
          {listing?.dirs.map((d) => (
            <li key={d.path}>
              <button type="button" onClick={() => load(d.path)}>
                <Icon name={d.vault ? 'gem' : 'folder'} size={17} />
                <span className="picker-name">{d.name}</span>
                {d.vault && <span className="badge">Vault</span>}
              </button>
            </li>
          ))}
          {listing && listing.dirs.length === 0 && <li className="picker-empty">Inga undermappar.</li>}
        </ul>

        <footer className="modal-foot">
          <span className={`picker-hint${listing?.isVault ? ' is-vault' : ''}`}>
            {listing?.isVault ? '✓ Det här är en Obsidian-vault' : 'Vaults känns igen på .obsidian-mappen'}
          </span>
          <button type="button" className="btn" onClick={onClose}>
            Avbryt
          </button>
          <button type="button" className="btn primary" disabled={!listing} onClick={() => listing && onPick(listing.path)}>
            Välj den här mappen
          </button>
        </footer>
      </div>
    </div>
  );
}
