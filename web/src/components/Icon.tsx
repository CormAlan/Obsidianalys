import type { ReactNode } from 'react';

export type IconName =
  | 'file'
  | 'folder'
  | 'folderOpen'
  | 'link'
  | 'network'
  | 'search'
  | 'sun'
  | 'moon'
  | 'monitor'
  | 'power'
  | 'external'
  | 'copy'
  | 'x'
  | 'up'
  | 'home'
  | 'table'
  | 'chart'
  | 'gem'
  | 'refresh'
  | 'arrowIn'
  | 'arrowOut';

const PATHS: Record<IconName, ReactNode> = {
  file: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5M9 13h6M9 17h4" />
    </>
  ),
  folder: <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h7.5A2.5 2.5 0 0 1 21 9.5v7a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5z" />,
  folderOpen: (
    <>
      <path d="M3 17V7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h6a2 2 0 0 1 2 2v1.5" />
      <path d="M3 17.5 5.6 12a2 2 0 0 1 1.8-1.2h12.4a1 1 0 0 1 .9 1.4l-2.4 5.6a2 2 0 0 1-1.8 1.2H4.5A1.5 1.5 0 0 1 3 17.5z" />
    </>
  ),
  link: (
    <>
      <path d="M10 13.5a4.5 4.5 0 0 0 6.4.4l3-3a4.5 4.5 0 0 0-6.4-6.4l-1.2 1.2" />
      <path d="M14 10.5a4.5 4.5 0 0 0-6.4-.4l-3 3a4.5 4.5 0 0 0 6.4 6.4l1.2-1.2" />
    </>
  ),
  network: (
    <>
      <circle cx="5.5" cy="6" r="2.5" />
      <circle cx="18.5" cy="6" r="2.5" />
      <circle cx="12" cy="18" r="2.5" />
      <path d="M8 6h8M7 8.2l3.7 7.6M17 8.2l-3.7 7.6" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4.5 4.5" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2.5v2M12 19.5v2M4.6 4.6 6 6M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />,
  monitor: (
    <>
      <rect x="3" y="4" width="18" height="12.5" rx="2" />
      <path d="M8.5 20.5h7M12 16.5v4" />
    </>
  ),
  power: (
    <>
      <path d="M12 3v8" />
      <path d="M6.3 7a8 8 0 1 0 11.4 0" />
    </>
  ),
  external: (
    <>
      <path d="M14 4h6v6M20 4l-8.5 8.5" />
      <path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4" />
    </>
  ),
  copy: (
    <>
      <rect x="8.5" y="8.5" width="11.5" height="11.5" rx="2" />
      <path d="M15.5 8.5V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h2.5" />
    </>
  ),
  x: <path d="M6 6l12 12M18 6 6 18" />,
  up: <path d="M12 19V5M5.5 11.5 12 5l6.5 6.5" />,
  home: (
    <>
      <path d="M4 10.5 12 4l8 6.5V19a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 19z" />
      <path d="M9.5 20.5v-6h5v6" />
    </>
  ),
  table: (
    <>
      <rect x="3.5" y="4.5" width="17" height="15" rx="2" />
      <path d="M3.5 9.5h17M3.5 14.5h17M9.5 9.5v10" />
    </>
  ),
  chart: <path d="M4 20V4M4 20h16M8 16V11M12 16V7M16 16v-3" />,
  gem: (
    <>
      <path d="M12 2.5 19.5 8l-2 11L12 21.5 6.5 19l-2-11z" />
      <path d="M12 2.5 9 12l3 9.5 3-9.5zM4.5 8 9 12h6l4.5-4" />
    </>
  ),
  refresh: (
    <>
      <path d="M20 11a8 8 0 0 0-14.6-4.5L4 8" />
      <path d="M4 3.5V8h4.5M4 13a8 8 0 0 0 14.6 4.5L20 16" />
      <path d="M20 20.5V16h-4.5" />
    </>
  ),
  arrowIn: <path d="M19 5 8 16M8 8v8h8" />,
  arrowOut: <path d="M5 19 16 8M8 8h8v8" />,
};

export function Icon({ name, size = 18, className }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      className={className ? `icon ${className}` : 'icon'}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PATHS[name]}
    </svg>
  );
}
