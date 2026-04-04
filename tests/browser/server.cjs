const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3457;
const ROOT = path.resolve(__dirname, '../..');

const MIME = {
  '.html': 'text/html; charset=UTF-8',
  '.js': 'application/javascript; charset=UTF-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=UTF-8',
  '.bin': 'application/octet-stream',
};

/**
 * /flat/ プレフィックスで dist/wasm/ と weights/ を同一ディレクトリのように配信。
 * loadModel の baseUrl 指定テスト用。
 */
const FLAT_DIRS = [
  path.join(ROOT, 'dist', 'wasm'),
  path.join(ROOT, 'weights'),
];

function serveFlatFile(fileName, res) {
  for (const dir of FLAT_DIRS) {
    const candidate = path.join(dir, fileName);
    if (fs.existsSync(candidate)) {
      const ext = path.extname(fileName);
      const mime = MIME[ext] || 'application/octet-stream';
      fs.readFile(candidate, (err, data) => {
        if (err) {
          res.writeHead(500);
          res.end('Read error');
          return;
        }
        res.writeHead(200, { 'Content-Type': mime });
        res.end(data);
      });
      return;
    }
  }
  res.writeHead(404);
  res.end('Not found in flat dirs: ' + fileName);
}

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(req.url.split('?')[0]);

  if (urlPath.startsWith('/flat/')) {
    const fileName = urlPath.slice('/flat/'.length);
    if (fileName.includes('..') || fileName.includes('/')) {
      res.writeHead(403);
      res.end('Forbidden');
      return;
    }
    serveFlatFile(fileName, res);
    return;
  }

  const filePath = path.join(ROOT, urlPath);
  const resolvedPath = path.resolve(filePath);
  const resolvedRoot = path.resolve(ROOT) + path.sep;
  if (!resolvedPath.startsWith(resolvedRoot) && resolvedPath !== path.resolve(ROOT)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }
  const ext = path.extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found: ' + urlPath);
      return;
    }
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Static server on http://localhost:${PORT}`);
});
