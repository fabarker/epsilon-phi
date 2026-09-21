'use strict';

/*
 * Visible counterpart to Cyrus's server-side access gate. The Flask page
 * route remains the primary boundary; this probe reflects session expiry and
 * allowlist denial without introducing a second authentication mechanism.
 */
(function () {
  function makeElement(tag, cssText, text) {
    var node = document.createElement(tag);
    if (cssText) node.style.cssText = cssText;
    if (text) node.textContent = text;
    return node;
  }

  function mount(node) {
    var previous = document.getElementById(node.id);
    if (previous) previous.remove();
    if (document.body) {
      document.body.appendChild(node);
    } else {
      document.addEventListener('DOMContentLoaded', function () {
        document.body.appendChild(node);
      }, {once: true});
    }
  }

  function badge(text, background, color) {
    var node = makeElement('div',
      'position:fixed;top:10px;right:10px;z-index:10000;max-width:calc(100vw - 20px);'
      + 'box-sizing:border-box;padding:5px 11px;border-radius:999px;background:' + background + ';'
      + 'color:' + color + ';box-shadow:0 1px 5px rgba(0,0,0,.2);'
      + 'font:600 12px/1.4 system-ui,sans-serif;', text);
    node.id = 'accessGateBadge';
    return node;
  }

  function deniedOverlay(message) {
    var scrim = makeElement('div',
      'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;'
      + 'box-sizing:border-box;padding:20px;background:rgba(16,24,40,.58);'
      + 'font-family:system-ui,sans-serif;');
    scrim.id = 'accessGateOverlay';
    scrim.setAttribute('role', 'alertdialog');
    scrim.setAttribute('aria-modal', 'true');
    scrim.setAttribute('aria-labelledby', 'accessGateTitle');
    var card = makeElement('div',
      'width:min(32rem,100%);box-sizing:border-box;padding:26px 30px;border-radius:10px;'
      + 'background:#fff;color:#1c2733;box-shadow:0 8px 30px rgba(0,0,0,.25);');
    var title = makeElement('h1', 'margin:0 0 8px;color:#9b1c1c;font-size:20px;', 'Access denied');
    title.id = 'accessGateTitle';
    card.appendChild(title);
    card.appendChild(makeElement('p', 'margin:0;', message));
    scrim.appendChild(card);
    return scrim;
  }

  function publish(status, detail) {
    var payload = Object.assign({status: status}, detail || {});
    window.dispatchEvent(new CustomEvent('proposaltoolv2:auth', {detail: payload}));
  }

  fetch('/api/whoami', {
    credentials: 'same-origin',
    cache: 'no-store',
    headers: {'Accept': 'application/json'}
  }).then(function (response) {
    return response.json().catch(function () { return {}; }).then(function (body) {
      return {status: response.status, body: body};
    });
  }).then(function (result) {
    var body = result.body || {};
    if (result.status === 200 && body.kerberos) {
      window.accessGateUser = body.kerberos;
      mount(badge(body.kerberos, '#e7f4ec', '#176a33'));
      publish('authenticated', {kerberos: body.kerberos});
      return;
    }

    if (result.status === 401 && body.loginUrl) {
      var signIn = badge('Not signed in · sign in', '#fde8e8', '#9b1c1c');
      signIn.setAttribute('role', 'button');
      signIn.tabIndex = 0;
      signIn.style.cursor = 'pointer';
      function navigate() { window.location.assign(body.loginUrl); }
      signIn.addEventListener('click', navigate);
      signIn.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          navigate();
        }
      });
      mount(signIn);
      publish('unauthenticated', {loginUrl: body.loginUrl});
      return;
    }

    if (result.status === 403) {
      mount(deniedOverlay(body.error || 'You are not on the PMG dashboard access list.'));
      publish('denied', {error: body.error || 'Access denied.'});
      return;
    }

    publish('error', {statusCode: result.status, error: body.error || 'Identity check failed.'});
  }).catch(function () {
    /* Offline shell contains no cached scenario data; API access remains server-enforced. */
    mount(badge('Offline · access unverified', '#fff4d6', '#725600'));
    publish('offline');
  });
})();
