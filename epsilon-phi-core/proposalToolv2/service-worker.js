'use strict';

/*
 * Proposal Tool v2 app-shell worker.
 *
 * Scenario and identity responses are deliberately network-only. They are
 * authenticated, short-lived, and must never be replayed from a browser
 * cache. Only files beneath this worker's /proposalToolv2/ scope are cached.
 */

const CACHE_PREFIX = 'proposal-tool-v2-shell-';
const CACHE_NAME = CACHE_PREFIX + 'v4';
const APP_BASE = new URL('./', self.location.href).pathname;

function appUrl(relativePath) {
  return new URL(relativePath, self.location.href).toString();
}

const ENTRY_URL = appUrl('proposalToolv2.html');
const CRITICAL_SHELL = [
  ENTRY_URL,
  appUrl('manifest.webmanifest'),
  appUrl('offline.html'),
  appUrl('static/css/proposalToolv2.css'),
  appUrl('static/js/accessGate.js'),
  appUrl('static/js/globals.js'),
  appUrl('static/js/proposalToolv2.js')
];
const OPTIONAL_SHELL = [
  appUrl('static/icons/icon.svg'),
  appUrl('static/icons/icon-192.png'),
  appUrl('static/icons/icon-512.png'),
  appUrl('static/fonts/gs-sans-variable.woff2'),
  appUrl('static/fonts/gs-sans-condensed-variable.woff2')
];

function isApiRequest(url) {
  return url.origin === self.location.origin
    && (url.pathname === '/api' || url.pathname.startsWith('/api/'));
}

function isCacheable(response) {
  return response
    && response.ok
    && response.type === 'basic'
    && !response.redirected;
}

async function fetchShellAsset(url, required) {
  try {
    const request = new Request(url, {
      cache: 'reload',
      credentials: 'same-origin'
    });
    const response = await fetch(request);
    if (!isCacheable(response)) {
      throw new Error('Unexpected response ' + response.status + ' for ' + url);
    }
    const cache = await caches.open(CACHE_NAME);
    await cache.put(url, response);
  } catch (error) {
    if (required) throw error;
  }
}

self.addEventListener('install', function (event) {
  event.waitUntil((async function () {
    await Promise.all(CRITICAL_SHELL.map(function (url) {
      return fetchShellAsset(url, true);
    }));
    await Promise.all(OPTIONAL_SHELL.map(function (url) {
      return fetchShellAsset(url, false);
    }));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', function (event) {
  event.waitUntil((async function () {
    const names = await caches.keys();
    await Promise.all(names.map(function (name) {
      if (name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME) {
        return caches.delete(name);
      }
      return Promise.resolve(false);
    }));
    await self.clients.claim();
  })());
});

function offlineResponse() {
  return new Response(
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
      + '<meta name="viewport" content="width=device-width,initial-scale=1">'
      + '<title>Proposal Tool offline</title>'
      + '<style>body{margin:0;min-height:100vh;display:grid;place-items:center;'
      + 'background:#f4f3ef;color:#12243a;font:16px/1.5 system-ui,sans-serif}'
      + 'main{box-sizing:border-box;width:min(36rem,calc(100% - 2rem));padding:2rem;'
      + 'background:#fff;border:1px solid #ccd3db;border-radius:12px}'
      + 'h1{font-size:1.45rem;margin:0 0 .5rem}p{margin:.5rem 0 1.25rem}'
      + 'button{border:0;border-radius:6px;padding:.7rem 1rem;background:#1f5fbf;'
      + 'color:#fff;font:inherit;font-weight:650;cursor:pointer}</style></head>'
      + '<body><main><h1>The Proposal Tool is offline</h1>'
      + '<p>The app shell is not available yet. Reconnect, then retry to load it securely.</p>'
      + '<button type="button" onclick="location.reload()">Retry</button></main></body></html>',
    {
      status: 503,
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
        'Cache-Control': 'no-store'
      }
    }
  );
}

async function handleNavigation(request) {
  try {
    const response = await fetch(request);
    if (isCacheable(response)) {
      const cache = await caches.open(CACHE_NAME);
      /* Cache under one canonical key: never persist a ?scenario= identifier. */
      await cache.put(ENTRY_URL, response.clone());
    }
    return response;
  } catch (error) {
    const cache = await caches.open(CACHE_NAME);
    return (await cache.match(ENTRY_URL))
      || (await cache.match(appUrl('offline.html')))
      || offlineResponse();
  }
}

async function handleScopedAsset(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (isCacheable(response)) {
    await cache.put(request, response.clone());
  }
  return response;
}

self.addEventListener('fetch', function (event) {
  const request = event.request;
  const url = new URL(request.url);

  /* Auth and scenario traffic is always live and is never placed in CacheStorage. */
  if (isApiRequest(url)) {
    event.respondWith(fetch(request));
    return;
  }

  if (request.method !== 'GET' || url.origin !== self.location.origin) return;

  const inScope = url.pathname.startsWith(APP_BASE);
  if (request.mode === 'navigate' && inScope) {
    event.respondWith(handleNavigation(request));
    return;
  }

  if (inScope) {
    event.respondWith(handleScopedAsset(request));
  }
});

self.addEventListener('message', function (event) {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
