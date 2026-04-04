const http = require('http');
const fs = require('fs');
const path = require('path');

const MIME = {
  '.html': 'text/html; charset=UTF-8',
  '.js': 'application/javascript; charset=UTF-8',
  '.wasm': 'application/wasm',
};

const ROOT = path.resolve(__dirname, '../..');

const server = http.createServer((req, res) => {
  const filePath = path.join(ROOT, decodeURIComponent(req.url));
  const ext = path.extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found: ' + req.url);
      return;
    }
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  });
});

server.listen(3456, () => {
  console.log('Static server on http://localhost:3456');
});
