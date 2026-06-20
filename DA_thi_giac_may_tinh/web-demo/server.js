const express = require('express');
const path = require('path');
const net = require('net');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8003';

// ── Proxy /api/* → FastAPI Python ──────────────────────────────────────────
app.use('/api', createProxyMiddleware({
  target: FASTAPI_URL,
  changeOrigin: true,
  pathRewrite: { '^/api': '' },   // /api/detect → /detect
  on: {
    error: (err, req, res) => {
      console.error('❌ FastAPI chưa chạy:', err.message);
      res.status(503).json({
        error: 'FastAPI server chưa khởi động',
        hint: 'Chạy: cd web-demo/api && start.bat (hoặc: uvicorn app:app --port 8000)',
      });
    },
  },
}));

// ── Serve static files (HTML, CSS, JS, assets) ─────────────────────────────
app.use(express.static(path.join(__dirname), {
  etag: false,
  lastModified: false,
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.js') || filePath.endsWith('.css') || filePath.endsWith('.html')) {
      res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
      res.setHeader('Pragma', 'no-cache');
    }
  }
}));

// ── Fallback to index.html ──────────────────────────────────────────────────
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// ── Tự động tìm port trống ──────────────────────────────────────────────────
function findFreePort(startPort) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.listen(startPort, () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on('error', () => resolve(findFreePort(startPort + 1)));
  });
}

findFreePort(3000).then((PORT) => {
  app.listen(PORT, () => {
    console.log(`\n✅  DefectDetect OBB đang chạy tại: http://localhost:${PORT}`);
    console.log(`🐍  FastAPI Python proxy → ${FASTAPI_URL}`);
    console.log(`   Nhấn Ctrl+C để dừng\n`);
  });
});
