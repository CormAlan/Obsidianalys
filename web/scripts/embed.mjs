// Bäddar in web/dist i C++-binären: skriver core/generated/web_assets.cpp.
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, '..', 'dist');
const outFile = join(here, '..', '..', 'core', 'generated', 'web_assets.cpp');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.json': 'application/json',
};

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

const files = walk(dist).sort();
let arrays = '';
let table = '';
files.forEach((file, i) => {
  const bytes = readFileSync(file);
  const url = '/' + relative(dist, file).split(sep).join('/');
  const mime = MIME[extname(file)] ?? 'application/octet-stream';
  const rows = [];
  for (let k = 0; k < bytes.length; k += 32) {
    rows.push(Array.from(bytes.subarray(k, k + 32), (b) => '0x' + b.toString(16).padStart(2, '0')).join(','));
  }
  arrays += `static const unsigned char a${i}[] = {\n${rows.join(',\n')}\n};\n`;
  table += `    {${JSON.stringify(url)}, ${JSON.stringify(mime)}, a${i}, sizeof(a${i})},\n`;
});

const source = `// Genererad av web/scripts/embed.mjs – redigera inte.
#include "web_assets.hpp"

namespace web {
${arrays}
const Asset kAssets[] = {
${table}    {nullptr, nullptr, nullptr, 0},
};
const std::size_t kAssetCount = ${files.length};

}  // namespace web
`;

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, source);
const total = files.reduce((s, f) => s + statSync(f).size, 0);
console.log(`Bäddade in ${files.length} filer (${(total / 1024).toFixed(0)} KiB) i ${relative(process.cwd(), outFile)}`);
