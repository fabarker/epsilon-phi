# Proposal Tool v2

Standalone, build-free frontend for the Cyrus Portfolio Management Group Proposal Tool. Open files
are plain HTML, CSS, and JavaScript; no package install, compilation, framework, or bundler is
required.

## Run in Cyrus

The frontend must be served from the Cyrus Flask origin so `/api/*`, the Kerberos/GSSSO cookie, and
the service worker share an origin. Deploy this entire directory as `proposalToolv2` beside the
host's other page directories; do not cherry-pick individual assets. Register this one static route
in the host's `dashboardFrontend.py`:

```python
@app.route('/proposalToolv2/<path:filename>')
def serve_proposal_tool_v2(filename):
    """Serve Proposal Tool v2 static files."""
    return send_from_directory(os.path.join(DASHBOARD_DIR, 'proposalToolv2'), filename)
```

Then restart the Flask frontend and open:

```text
http://localhost:8001/proposalToolv2/proposalToolv2.html
```

Route registration is the only required host integration. The snippet is documentation for the
receiving Cyrus project; it is intentionally **not** applied in this immutable-backend workspace
change. No FastAPI route, endpoint, auth dependency, request shape, or server-side file is changed.
A `file://` page or unrelated static-server port can be used for visual review only: it cannot
exercise the authenticated same-origin API.

## Runtime contract

The browser calls the existing same-origin `/api/scenario/*` surface only. It reads every option,
threshold, capability, category, availability key, data-provenance value, and implementation route
from the schema. Portfolio resolves can run independently; scenario writes are sequenced so a basis
update reaches the store before rebuilds, sleeve snapshots cannot overtake one another, and export
waits for all mutations. No analytics or product records are embedded in this directory.

The route picker is rendered when `options.implementationVariants` is supplied by the fixed port.
Older stored scenarios and an older port without that schema field continue through the legacy
single-library sleeve request shape.

## PWA behavior

- `manifest.webmanifest` installs with scope `/proposalToolv2/` and references the 192 px and 512 px
  maskable icons in `static/icons/`. The page requests the protected manifest with same-origin
  credentials so it passes the same Cyrus gate as the page.
- `service-worker.js` pre-caches the application shell beneath that scope. Navigation is
  network-first and falls back to the cached shell, then to an explicit offline error if the shell
  has never been cached.
- `/api` and `/api/*` are always network-only. Identity, mandate, scenario, portfolio, sleeve, and
  export responses never enter CacheStorage.
- `static/js/accessGate.js` probes same-origin `/api/whoami` with credentials. Cyrus still enforces
  access before serving the page and FastAPI re-enforces it on every API request.
- Service workers require HTTPS in deployment; `localhost` is accepted as a secure development
  context.

If shell filenames or assets change, update `CRITICAL_SHELL` and increment `CACHE_NAME` in
`service-worker.js`. Browser developer tools can unregister the worker and clear site storage when
testing a clean install.
