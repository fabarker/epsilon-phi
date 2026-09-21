'use strict';

(function () {
  var API_BASE = window.API_BASE || (window.location.origin + '/api');
  var SERIES = ['#193f67', '#2d76ae', '#58a5aa', '#bd7d43'];
  var CATEGORY = ['#214f7b', '#3f7fa2', '#6cb1b2', '#d49a55', '#ae685f', '#81658c', '#8a9f65'];

  var state = {
    phase: 'landing',
    stage: 'allocation',
    schema: null,
    schemaStatus: 'loading',
    schemaError: null,
    scenarioId: null,
    mandate: null,
    basis: {currency: 'USD', hedging: 'Hedged'},
    columns: [],
    baseDraft: {allocation: '', excludeRE: false, excludeTAA: false, riskLevel: ''},
    variant: null,
    sleeves: {},
    sleeveLib: {},
    exportStatus: 'idle',
    exportError: null,
    pendingWrites: 0,
    writeRevision: 0,
    syncError: null,
    pendingDeletes: {},
    installPrompt: null,
    busyBase: false,
    variantBusy: false
  };

  var writeTail = Promise.resolve();
  var trackedMutations = new Set();
  var modal = null;
  var modalOpener = null;
  var modalKeydown = null;
  var advisorTimer = null;
  var advisorController = null;
  var announcementTimers = {};

  function byId(id) { return document.getElementById(id); }
  function esc(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function finite(value) { return typeof value === 'number' && isFinite(value); }
  function fmt(value, digits, suffix) {
    if (!finite(value)) return '—';
    return value.toFixed(digits) + (suffix || '');
  }
  function fmtWeight(value, digits) {
    if (!finite(value)) return '';
    return value < 0.05 ? '—' : value.toFixed(digits === undefined ? 1 : digits) + '%';
  }
  function money(value) {
    if (!finite(Number(value))) return '—';
    return '$' + Math.round(Number(value)).toLocaleString('en-US');
  }
  function parseMoney(value) {
    var parsed = Number(String(value || '').replace(/[^0-9.]/g, ''));
    return isFinite(parsed) ? Math.round(parsed) : 0;
  }
  function keyString(key) {
    return [key.allocation, key.excludeRE ? 1 : 0, key.excludeTAA ? 1 : 0, key.riskLevel].join('|');
  }
  function parseKey(value) {
    var parts = String(value || '').split('|');
    if (parts.length !== 4) return null;
    return {allocation: parts[0], excludeRE: parts[1] === '1', excludeTAA: parts[2] === '1', riskLevel: parts[3]};
  }
  function cloneKey(key) {
    return {allocation: key.allocation, excludeRE: !!key.excludeRE, excludeTAA: !!key.excludeTAA, riskLevel: key.riskLevel};
  }
  function keyName(key, withCurrency) {
    if (!key) return '';
    var suffix = '';
    if (key.excludeRE && reAllowed(key.allocation)) suffix += ' ex RE';
    if (key.excludeTAA) suffix += ' ex TAA';
    return (withCurrency ? state.basis.currency + ' ' : '') + key.allocation + ' ' + key.riskLevel + suffix;
  }
  function schema(path, fallback) {
    var value = state.schema;
    path.split('.').forEach(function (part) { value = value && value[part]; });
    return value === undefined || value === null ? fallback : value;
  }
  function availability() {
    if (!state._availability || state._availabilitySource !== state.schema) {
      state._availability = {};
      schema('availability', []).forEach(function (key) { state._availability[key] = true; });
      state._availabilitySource = state.schema;
    }
    return state._availability;
  }
  function isAvailable(key) { return !!availability()[keyString(key)]; }
  function reAllowed(allocation) { return schema('options.reAllowed', []).indexOf(allocation) >= 0; }
  function canEdit() { return schema('capabilities.canEdit', true) !== false; }
  function canExport() { return schema('capabilities.canExport', true) !== false; }
  function maxPortfolios() { return Number(schema('rules.maxPortfolios', 4)) || 4; }
  function autoCategories() { return schema('rules.autoSleeveCategories', []); }
  function variantOptions() { return schema('options.implementationVariants', []); }
  function needsVariant() { return variantOptions().length > 0; }
  function baseColumn() { return state.columns.length && state.columns[0].role === 'base' ? state.columns[0] : null; }
  function readyColumns() { return state.columns.filter(function (column) { return column.status === 'ready'; }); }
  function allColumnsSettled() { return state.columns.every(function (column) { return column.status === 'ready'; }); }
  function announce(channel, message) {
    var id = channel === 'assertive' ? 'live-assertive' : 'live-polite';
    clearTimeout(announcementTimers[id]);
    var node = byId(id);
    if (!node) return;
    node.textContent = '';
    announcementTimers[id] = setTimeout(function () { node.textContent = message; }, 35);
  }
  function toast(message, kind) {
    var region = byId('toast-region');
    if (!region) return;
    var node = document.createElement('div');
    node.className = 'toast ' + (kind || '');
    node.textContent = message;
    region.replaceChildren(node);
    setTimeout(function () { if (node.parentNode) node.remove(); }, 5000);
  }
  function showAlert(message, kind) {
    var area = byId('alertArea');
    if (!area) return;
    area.innerHTML = '<div class="alert alert-' + esc(kind || 'error') + '" role="alert">'
      + esc(message) + '<button type="button" aria-label="Dismiss message" data-dismiss-alert>&times;</button></div>';
  }
  function clearAlert() { var area = byId('alertArea'); if (area) area.replaceChildren(); }
  function preserveFocus(render) {
    var active = document.activeElement;
    var id = active && active.id;
    var start = active && typeof active.selectionStart === 'number' ? active.selectionStart : null;
    var end = active && typeof active.selectionEnd === 'number' ? active.selectionEnd : null;
    render();
    if (!id) return;
    var next = byId(id);
    if (!next) return;
    try {
      next.focus({preventScroll: true});
      if (start !== null && typeof next.setSelectionRange === 'function') next.setSelectionRange(start, end);
    } catch (ignore) {}
  }

  function ApiError(message, status, body) {
    this.name = 'ApiError';
    this.message = message;
    this.status = status;
    this.body = body || null;
  }
  ApiError.prototype = Object.create(Error.prototype);

  async function apiRequest(path, options, responseType) {
    var config = Object.assign({credentials: 'same-origin', cache: 'no-store'}, options || {});
    config.headers = Object.assign({'Accept': responseType === 'blob' ? '*/*' : 'application/json'}, config.headers || {});
    var response;
    try {
      response = await fetch(API_BASE + path, config);
    } catch (error) {
      if (error && error.name === 'AbortError') throw error;
      throw new ApiError(navigator.onLine ? 'The service could not be reached. Try again.' : 'You are offline. Reconnect to continue.', 0, null);
    }
    if (!response.ok) {
      var body = null;
      var message = 'Request failed (' + response.status + ').';
      try { body = await response.json(); } catch (ignore) {}
      if (body && body.loginUrl && response.status === 401) {
        window.location.assign(body.loginUrl);
        throw new ApiError('Your session has expired. Redirecting to sign in.', response.status, body);
      }
      if (body && body.error) message = body.error;
      throw new ApiError(message, response.status, body);
    }
    if (responseType === 'blob') return {blob: await response.blob(), response: response};
    if (response.status === 204) return null;
    return response.json();
  }

  function schemaQuery() {
    var query = '?currency=' + encodeURIComponent(state.basis.currency)
      + '&hedging=' + encodeURIComponent(state.basis.hedging);
    if (state.mandate && state.mandate.mandateSize) query += '&mandateSize=' + encodeURIComponent(state.mandate.mandateSize);
    return query;
  }

  async function loadSchema(showLoading) {
    if (showLoading !== false) state.schemaStatus = 'loading';
    try {
      var payload = await apiRequest('/scenario/schema' + schemaQuery());
      state.schema = payload;
      state.schemaStatus = 'ready';
      state.schemaError = null;
      state._availabilitySource = null;
      normaliseBaseDraft();
      renderShell();
      return payload;
    } catch (error) {
      state.schemaStatus = 'error';
      state.schemaError = error.message;
      renderShell();
      throw error;
    }
  }

  function serialWrite(payload) {
    var revision = ++state.writeRevision;
    state.pendingWrites += 1;
    renderImplementationSummary();
    var run = writeTail.catch(function () {}).then(function () {
      return apiRequest('/scenario/' + encodeURIComponent(state.scenarioId), {
        method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
      });
    });
    writeTail = run.then(function (result) {
      if (revision === state.writeRevision) state.syncError = null;
      return result;
    }).catch(function (error) {
      if (revision === state.writeRevision) state.syncError = error.message;
      throw error;
    }).finally(function () {
      state.pendingWrites = Math.max(0, state.pendingWrites - 1);
      renderImplementationSummary();
    });
    writeTail.catch(function (error) {
      showAlert('A scenario change was not saved: ' + error.message, 'error');
    });
    return writeTail;
  }

  async function flushWrites() {
    try { await writeTail; } catch (ignore) {}
    if (trackedMutations.size) await Promise.allSettled(Array.from(trackedMutations));
    if (state.syncError) throw new Error(state.syncError);
  }
  function trackMutation(promise) {
    trackedMutations.add(promise);
    function settled() { trackedMutations.delete(promise); renderImplementationSummary(); }
    promise.then(settled, settled);
    renderImplementationSummary();
    return promise;
  }

  function normaliseBaseDraft() {
    var draft = state.baseDraft;
    var allocations = schema('options.allocations', []);
    if (draft.allocation && allocations.indexOf(draft.allocation) < 0) {
      draft = {allocation: '', excludeRE: false, excludeTAA: false, riskLevel: ''};
    }
    if (draft.allocation && !reAllowed(draft.allocation)) draft.excludeRE = true;
    if (draft.riskLevel && !isAvailable(draft)) draft.riskLevel = '';
    state.baseDraft = draft;
  }
  function risksFor(draft) {
    if (!draft.allocation) return [];
    return schema('options.riskLevels', []).filter(function (risk) {
      return isAvailable({allocation: draft.allocation, excludeRE: !!draft.excludeRE, excludeTAA: !!draft.excludeTAA, riskLevel: risk});
    });
  }
  function usedKey(key) {
    var string = keyString(key);
    return state.columns.some(function (column) { return keyString(column.key) === string; }) || !!state.pendingDeletes[string];
  }
  function occupiedSlots() {
    return state.columns.length + Object.keys(state.pendingDeletes).length;
  }

  function createColumn(key, role) {
    return {
      key: cloneKey(key), role: role, status: 'loading', data: null, error: null,
      skeleton: false, skeletonTimer: null, removed: false, promise: null
    };
  }

  function resolveColumn(column) {
    column.status = 'loading';
    column.error = null;
    column.skeleton = false;
    clearTimeout(column.skeletonTimer);
    column.skeletonTimer = setTimeout(function () {
      if (column.status === 'loading' && !column.removed) {
        column.skeleton = true;
        renderAllocationViews();
      }
    }, 200);
    var promise = apiRequest('/scenario/' + encodeURIComponent(state.scenarioId) + '/portfolio', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key: column.key, role: column.role})
    }).then(async function (payload) {
      clearTimeout(column.skeletonTimer);
      if (column.removed) {
        column.data = payload.portfolio;
        column.status = 'ready';
        if (column.role === 'comparison') await compensateDelete(column.key);
        return;
      }
      column.status = 'ready';
      column.skeleton = false;
      column.data = payload.portfolio;
      if (column.role === 'base') onBaseReady(column);
      announce('polite', (column.data.header || keyName(column.key)) + ' is ready.');
      renderAllocationViews();
      renderImplementation();
    }).catch(function (error) {
      clearTimeout(column.skeletonTimer);
      if (column.removed) {
        delete state.pendingDeletes[keyString(column.key)];
        renderAllocationViews();
        return;
      }
      column.status = 'error';
      column.error = error.message;
      column.skeleton = false;
      announce('assertive', keyName(column.key) + ' could not be built. Retry is available.');
      renderAllocationViews();
      renderImplementation();
    }).finally(function () {
      column.promise = null;
      renderImplementationSummary();
    });
    column.promise = trackMutation(promise);
    renderAllocationViews();
    return promise;
  }

  async function compensateDelete(key) {
    var string = keyString(key);
    try {
      await apiRequest('/scenario/' + encodeURIComponent(state.scenarioId) + '/portfolio/' + encodeURIComponent(string), {method: 'DELETE'});
    } catch (error) {
      showAlert('The comparison could not be removed, so it has been restored: ' + error.message, 'error');
      var removedColumn = state._removedColumns && state._removedColumns[string];
      delete state.pendingDeletes[string];
      if (removedColumn && !state.columns.some(function (column) { return keyString(column.key) === string; })
          && state.columns.length < maxPortfolios()) {
        removedColumn.removed = false;
        if (removedColumn.data) removedColumn.status = 'ready';
        state.columns.push(removedColumn);
      } else if (removedColumn) {
        state.syncError = 'A comparison removal was not saved.';
      }
    } finally {
      delete state.pendingDeletes[string];
      if (state._removedColumns) delete state._removedColumns[string];
      renderAllocationViews();
    }
  }

  async function setBase(key) {
    if (!canEdit() || state.busyBase || !isAvailable(key)) return;
    var old = baseColumn();
    if (old && keyString(old.key) === keyString(key) && old.status !== 'error') return;
    state.busyBase = true;
    renderBaseBuilder();
    try {
      if (old && old.promise) await old.promise;
      var duplicate = state.columns.find(function (column) {
        return column.role === 'comparison' && keyString(column.key) === keyString(key);
      });
      if (duplicate) await removeComparison(duplicate);
      if (old) old.removed = true;
      var next = createColumn(key, 'base');
      state.columns = [next].concat(state.columns.filter(function (column) { return column.role === 'comparison' && column !== duplicate; }));
      state.stage = 'allocation';
      renderWorkspace();
      await resolveColumn(next);
    } finally {
      state.busyBase = false;
      renderBaseBuilder();
    }
  }

  function addComparison(key) {
    if (!baseColumn() || !canEdit() || occupiedSlots() >= maxPortfolios() || !isAvailable(key) || usedKey(key)) return false;
    var column = createColumn(key, 'comparison');
    state.columns.push(column);
    renderAllocationViews();
    resolveColumn(column);
    announce('polite', keyName(key) + ' added for comparison.');
    return true;
  }

  async function removeComparison(column) {
    if (!column || column.role !== 'comparison' || column.removed) return;
    var string = keyString(column.key);
    column.removed = true;
    state.pendingDeletes[string] = true;
    state._removedColumns = state._removedColumns || {};
    state._removedColumns[string] = column;
    state.columns = state.columns.filter(function (item) { return item !== column; });
    renderAllocationViews();
    announce('polite', keyName(column.key) + ' removed.');
    var deletion = (async function () {
      if (column.promise) await column.promise;
      else await compensateDelete(column.key);
    })();
    trackMutation(deletion);
    return deletion;
  }

  function retryColumn(column) {
    if (column && column.status === 'error' && !column.removed) resolveColumn(column);
  }

  function onBaseReady(column) {
    if (baseColumn() !== column || !column.data) return;
    var held = {};
    (column.data.categories || []).forEach(function (category) { held[category.name] = true; });
    var changed = false;
    Object.keys(state.sleeves).forEach(function (category) {
      if (!held[category]) { delete state.sleeves[category]; changed = true; }
    });
    Object.keys(state.sleeveLib).forEach(function (category) {
      if (!held[category]) delete state.sleeveLib[category];
    });
    if (changed && state.scenarioId && (!needsVariant() || state.variant)) serialWrite({sleeves: sleeveSnapshot()});
  }

  function inflightColumns() {
    return state.columns.map(function (column) { return column.promise; }).filter(Boolean);
  }

  async function applyBasis(nextBasis) {
    if (nextBasis.currency === state.basis.currency && nextBasis.hedging === state.basis.hedging) return;
    clearAlert();
    var oldBasis = state.basis;
    var persisted = false;
    try {
      await Promise.allSettled(inflightColumns());
      await flushWrites();
      var payload = {basis: nextBasis};
      if (Object.keys(state.sleeves).length && (!needsVariant() || state.variant)) payload.sleeves = {};
      await serialWrite(payload);
      persisted = true;
      state.basis = {currency: nextBasis.currency, hedging: nextBasis.hedging};
      state.sleeves = {};
      state.sleeveLib = {};
      await loadSchema(false);
      await pruneUnavailable('basis');
      state.columns.forEach(function (column) {
        column.data = null;
        resolveColumn(column);
      });
      announce('assertive', 'Basis changed to ' + state.basis.currency + ', ' + state.basis.hedging + '. Portfolio analytics are rebuilding.');
      toast('Basis updated. Analytics are rebuilding.', 'success');
      renderWorkspace();
      return true;
    } catch (error) {
      if (!persisted) state.basis = oldBasis;
      showAlert(error.message, 'error');
      if (!persisted) {
        try { await loadSchema(false); } catch (ignore) {}
      }
    }
    renderWorkspace();
    return false;
  }

  async function pruneUnavailable(reason) {
    var base = baseColumn();
    if (base && !isAvailable(base.key)) {
      var comparisons = state.columns.filter(function (column) { return column.role === 'comparison'; });
      base.removed = true;
      comparisons.forEach(function (column) { removeComparison(column); });
      state.columns = [];
      state.sleeves = {};
      state.sleeveLib = {};
      announce('assertive', 'The prior base is unavailable under the new ' + reason + '. Choose a new base portfolio.');
      return;
    }
    var removed = state.columns.filter(function (column) { return column.role === 'comparison' && !isAvailable(column.key); });
    removed.forEach(function (column) { removeComparison(column); });
  }

  function sleeveSnapshot() {
    var snapshot = {};
    Object.keys(state.sleeves).forEach(function (category) {
      if (autoCategories().indexOf(category) < 0 && state.sleeves[category]) snapshot[category] = state.sleeves[category];
    });
    return snapshot;
  }

  function sleeveFor(category) {
    var library = state.sleeveLib[category];
    if (!library || library.status !== 'ready') return null;
    if (autoCategories().indexOf(category) >= 0) return library.sleeves[0] || null;
    var name = state.sleeves[category];
    return library.sleeves.find(function (sleeve) { return sleeve.name === name; }) || null;
  }

  function sleeveQuery(category) {
    var query = '?category=' + encodeURIComponent(category)
      + '&currency=' + encodeURIComponent(state.basis.currency)
      + '&hedging=' + encodeURIComponent(state.basis.hedging);
    if (needsVariant()) query += '&variant=' + encodeURIComponent(state.variant || '');
    return query;
  }

  function loadSleeveLibrary(category, force) {
    if (needsVariant() && !state.variant) return;
    var current = state.sleeveLib[category];
    if (current && !force) return;
    var requestState = {status: 'loading', sleeves: [], error: null};
    state.sleeveLib[category] = requestState;
    renderSleeves();
    apiRequest('/scenario/sleeves' + sleeveQuery(category)).then(function (payload) {
      if (state.sleeveLib[category] !== requestState) return;
      state.sleeveLib[category] = {status: 'ready', sleeves: payload.sleeves || [], error: null};
      if (state.sleeves[category] && !sleeveFor(category)) {
        delete state.sleeves[category];
        serialWrite({sleeves: sleeveSnapshot()});
      }
      renderImplementation();
    }).catch(function (error) {
      if (state.sleeveLib[category] !== requestState) return;
      state.sleeveLib[category] = {status: 'error', sleeves: [], error: error.message};
      renderImplementation();
    });
  }

  function ensureSleeveLibraries() {
    var base = baseColumn();
    if (state.stage !== 'implementation' || !base || base.status !== 'ready') return;
    if (needsVariant() && !state.variant) return;
    (base.data.categories || []).forEach(function (category) { loadSleeveLibrary(category.name, false); });
  }

  async function chooseVariant(value) {
    if (!canEdit() || state.variantBusy || value === state.variant) return;
    var previous = state.variant;
    var previousSleeves = Object.assign({}, state.sleeves);
    state.variant = value;
    state.variantBusy = true;
    state.sleeves = {};
    state.sleeveLib = {};
    renderImplementation();
    try {
      await serialWrite({variant: value, sleeves: {}});
      announce('polite', value + ' implementation route selected. Sleeve choices have been reset.');
      ensureSleeveLibraries();
    } catch (error) {
      state.variant = previous;
      state.sleeves = previousSleeves;
      state.sleeveLib = {};
      state.syncError = null;
    } finally {
      state.variantBusy = false;
      renderImplementation();
    }
  }

  function chooseSleeve(category, name) {
    if (!canEdit()) return;
    if (name) state.sleeves[category] = name;
    else delete state.sleeves[category];
    state.syncError = null;
    serialWrite({sleeves: sleeveSnapshot()});
    renderImplementation();
    announce('polite', name ? name + ' attached to ' + category + '.' : 'Sleeve removed from ' + category + '.');
  }

  function setBackgroundInert(inert) {
    ['app-header', 'main-content'].forEach(function (id) {
      var node = byId(id);
      if (!node) return;
      if (inert) node.setAttribute('inert', '');
      else node.removeAttribute('inert');
    });
  }

  function openModal(kind, data) {
    closeModal(false);
    modalOpener = document.activeElement;
    modal = {kind: kind, data: data || {}, busy: false};
    setBackgroundInert(true);
    document.body.style.overflow = 'hidden';
    renderModal();
    modalKeydown = function (event) {
      if (!modal) return;
      if (event.key === 'Escape' && !modal.busy) { event.preventDefault(); closeModal(); return; }
      if (event.key !== 'Tab') return;
      var root = byId('modal-root');
      var focusable = Array.from(root.querySelectorAll('button:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex]:not([tabindex="-1"])'));
      if (!focusable.length) return;
      var first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', modalKeydown);
    setTimeout(function () {
      var modalRoot = byId('modal-root');
      var first = modalRoot.querySelector('[autofocus]')
        || modalRoot.querySelector('input:not([disabled]),select:not([disabled]),button:not([disabled])');
      if (first) first.focus();
    }, 0);
  }

  function closeModal(restore) {
    if (advisorTimer) clearTimeout(advisorTimer);
    if (advisorController) advisorController.abort();
    advisorTimer = null;
    advisorController = null;
    if (modalKeydown) document.removeEventListener('keydown', modalKeydown);
    modalKeydown = null;
    modal = null;
    var root = byId('modal-root');
    if (root) root.replaceChildren();
    setBackgroundInert(false);
    document.body.style.overflow = '';
    if (restore !== false && modalOpener && document.contains(modalOpener)) modalOpener.focus();
    modalOpener = null;
  }

  function modalFrame(label, title, body, footer, wide) {
    return '<div class="modal-backdrop" data-modal-backdrop>'
      + '<section class="modal' + (wide ? ' modal-wide' : '') + '" role="dialog" aria-modal="true" aria-labelledby="modal-title" tabindex="-1">'
      + '<header class="modal-header"><div><p class="eyebrow">' + esc(label) + '</p><h2 id="modal-title">' + esc(title) + '</h2></div>'
      + '<button type="button" class="modal-close" aria-label="Close dialog" data-modal-close>&times;</button></header>'
      + '<div class="modal-body">' + body + '</div><footer class="modal-footer">' + footer + '</footer></section></div>';
  }

  function renderModal() {
    if (!modal) return;
    if (modal.kind === 'mandate') renderMandateModal();
    else if (modal.kind === 'basis') renderBasisModal();
    else if (modal.kind === 'comparison') renderComparisonModal();
  }

  function renderMandateModal() {
    var root = byId('modal-root');
    var draft = modal.data.draft;
    var errors = modal.data.errors || {};
    var editing = !!state.mandate;
    function errorFor(field) {
      return errors[field] ? '<span class="field-error" id="err-' + field + '">' + esc(errors[field]) + '</span>' : '';
    }
    function invalid(field) {
      return errors[field] ? ' aria-invalid="true" aria-describedby="err-' + field + '"' : '';
    }
    var floor = Number(schema('rules.mandateFloor', 5000000));
    var body = (modal.data.banner ? '<p class="form-banner" role="alert">' + esc(modal.data.banner) + '</p>' : '')
      + '<p class="modal-copy">The mandate anchors every availability rule and every implementation amount. Amounts are entered in US dollars.</p>'
      + '<form id="mandate-form" novalidate><div class="form-grid">'
      + '<div class="field"><label for="top-account">Top account size</label><div class="money-input"><input id="top-account" name="topAccountSize" inputmode="decimal" autocomplete="off" value="' + esc(draft.topAccountSize) + '"' + invalid('topAccountSize') + ' autofocus></div>' + errorFor('topAccountSize') + '</div>'
      + '<div class="field"><label for="mandate-size">Mandate size</label><div class="money-input"><input id="mandate-size" name="mandateSize" inputmode="decimal" autocomplete="off" value="' + esc(draft.mandateSize) + '"' + invalid('mandateSize') + '></div><span class="field-help">Minimum ' + esc(money(floor)) + '</span>' + errorFor('mandateSize') + '</div>'
      + '<div class="field span-2 combo-wrap"><label for="primary-pwa">Primary PWA</label><input id="primary-pwa" name="primaryPwa" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="pwa-options" autocomplete="off" value="' + esc(draft.primaryPwa) + '"' + invalid('primaryPwa') + '><ul id="pwa-options" class="combo-list" role="listbox" aria-label="Matching advisers" hidden></ul><span class="field-help">Enter at least two characters, then choose a directory result.</span>' + errorFor('primaryPwa') + '</div>'
      + '</div></form>';
    var footer = '<button class="button button-secondary" type="button" data-modal-close>' + (editing ? 'Cancel' : 'Back') + '</button>'
      + '<button class="button button-primary" type="submit" form="mandate-form"' + (modal.busy ? ' disabled' : '') + '>' + (modal.busy ? 'Saving…' : editing ? 'Save mandate' : 'Create proposal') + '</button>';
    root.innerHTML = modalFrame(editing ? 'Scenario context' : 'Start a proposal', editing ? 'Edit the mandate' : 'Frame the mandate', body, footer, false);
    bindMandateModal();
  }

  function bindMandateModal() {
    var input = byId('primary-pwa');
    var draft = modal.data.draft;
    ['top-account', 'mandate-size'].forEach(function (id) {
      byId(id).addEventListener('input', function (event) {
        draft[event.target.name] = event.target.value;
        delete modal.data.errors[event.target.name];
      });
      byId(id).addEventListener('blur', function (event) {
        var amount = parseMoney(event.target.value);
        if (amount) { event.target.value = amount.toLocaleString('en-US'); draft[event.target.name] = event.target.value; }
      });
    });
    input.addEventListener('input', function () {
      draft.primaryPwa = input.value;
      draft.selectedPwa = '';
      modal.data.activeAdvisor = -1;
      delete modal.data.errors.primaryPwa;
      scheduleAdvisorSearch(input.value);
    });
    input.addEventListener('keydown', function (event) {
      var matches = modal.data.matches || [];
      if (event.key === 'ArrowDown' && matches.length) {
        event.preventDefault(); modal.data.activeAdvisor = Math.min(matches.length - 1, (modal.data.activeAdvisor || -1) + 1); paintAdvisorList();
      } else if (event.key === 'ArrowUp' && matches.length) {
        event.preventDefault(); modal.data.activeAdvisor = Math.max(0, (modal.data.activeAdvisor || 0) - 1); paintAdvisorList();
      } else if (event.key === 'Enter' && modal.data.activeAdvisor >= 0 && matches[modal.data.activeAdvisor]) {
        event.preventDefault(); selectAdvisor(matches[modal.data.activeAdvisor]);
      } else if (event.key === 'Escape') {
        var list = byId('pwa-options'); if (list && !list.hidden) { event.stopPropagation(); list.hidden = true; input.setAttribute('aria-expanded', 'false'); }
      }
    });
    byId('mandate-form').addEventListener('submit', submitMandate);
  }

  function scheduleAdvisorSearch(query) {
    clearTimeout(advisorTimer);
    if (advisorController) advisorController.abort();
    var list = byId('pwa-options');
    if (String(query).trim().length < 2) {
      modal.data.matches = null;
      if (list) list.hidden = true;
      byId('primary-pwa').setAttribute('aria-expanded', 'false');
      return;
    }
    modal.data.searching = true;
    paintAdvisorList();
    advisorTimer = setTimeout(async function () {
      advisorController = 'AbortController' in window ? new AbortController() : null;
      try {
        var payload = await apiRequest('/scenario/advisors?q=' + encodeURIComponent(query), {signal: advisorController ? advisorController.signal : undefined});
        if (!modal || modal.kind !== 'mandate' || byId('primary-pwa').value !== query) return;
        modal.data.matches = payload.advisors || [];
        modal.data.searching = false;
        modal.data.activeAdvisor = modal.data.matches.length ? 0 : -1;
        paintAdvisorList();
      } catch (error) {
        if (error.name === 'AbortError' || !modal) return;
        modal.data.matches = [];
        modal.data.searching = false;
        modal.data.advisorError = error.message;
        paintAdvisorList();
      }
    }, 250);
  }

  function paintAdvisorList() {
    var list = byId('pwa-options');
    var input = byId('primary-pwa');
    if (!list || !input || !modal) return;
    var matches = modal.data.matches || [];
    if (modal.data.searching) list.innerHTML = '<li role="option" aria-disabled="true">Searching the directory…</li>';
    else if (modal.data.advisorError) list.innerHTML = '<li role="option" aria-disabled="true">' + esc(modal.data.advisorError) + '</li>';
    else if (!matches.length) list.innerHTML = '<li role="option" aria-disabled="true">No matching advisers</li>';
    else list.innerHTML = matches.map(function (advisor, index) {
      return '<li role="option" id="pwa-option-' + index + '" data-advisor-index="' + index + '" aria-selected="' + (index === modal.data.activeAdvisor) + '" class="' + (index === modal.data.activeAdvisor ? 'active' : '') + '">'
        + esc(advisor.name) + (advisor.office ? '<small>' + esc(advisor.office) + '</small>' : '') + '</li>';
    }).join('');
    list.hidden = false;
    input.setAttribute('aria-expanded', 'true');
    if (modal.data.activeAdvisor >= 0) input.setAttribute('aria-activedescendant', 'pwa-option-' + modal.data.activeAdvisor);
    else input.removeAttribute('aria-activedescendant');
    list.querySelectorAll('[data-advisor-index]').forEach(function (option) {
      option.addEventListener('pointerdown', function (event) { event.preventDefault(); });
      option.addEventListener('click', function () {
        selectAdvisor(matches[Number(option.dataset.advisorIndex)]);
      });
    });
  }

  function selectAdvisor(advisor) {
    if (!advisor || !modal) return;
    var storedValue = advisor.display || advisor.name;
    modal.data.draft.primaryPwa = storedValue;
    modal.data.draft.selectedPwa = storedValue;
    var input = byId('primary-pwa');
    input.value = storedValue;
    input.setAttribute('aria-expanded', 'false');
    byId('pwa-options').hidden = true;
  }

  function validateMandateDraft(draft) {
    var errors = {};
    var top = parseMoney(draft.topAccountSize);
    var amount = parseMoney(draft.mandateSize);
    var floor = Number(schema('rules.mandateFloor', 5000000));
    if (!top) errors.topAccountSize = 'Enter the top account size.';
    if (!amount || amount < floor) errors.mandateSize = 'Mandate size must be at least ' + money(floor) + '.';
    else if (top && amount > top) errors.mandateSize = 'Mandate size cannot exceed the top account size.';
    if (!draft.primaryPwa || draft.selectedPwa !== draft.primaryPwa) errors.primaryPwa = 'Choose a Primary PWA from the directory list.';
    return {errors: errors, mandate: {topAccountSize: top, mandateSize: amount, primaryPwa: draft.primaryPwa}};
  }

  async function submitMandate(event) {
    event.preventDefault();
    if (!modal || modal.busy) return;
    var result = validateMandateDraft(modal.data.draft);
    modal.data.errors = result.errors;
    if (Object.keys(result.errors).length) {
      renderMandateModal();
      var bad = byId('modal-root').querySelector('[aria-invalid="true"]'); if (bad) bad.focus();
      return;
    }
    modal.busy = true;
    renderMandateModal();
    clearAlert();
    var editing = !!state.mandate;
    try {
      if (editing) {
        await flushWrites();
        await serialWrite({mandate: result.mandate});
        state.mandate = result.mandate;
      } else {
        var created = await apiRequest('/scenario', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({mandate: result.mandate, basis: state.basis})
        });
        state.scenarioId = created.id;
        state.mandate = result.mandate;
        state.phase = 'workspace';
        state.stage = 'allocation';
        var url = new URL(window.location.href);
        url.searchParams.set('scenario', created.id);
        history.replaceState({}, '', url);
      }
      await loadSchema(false);
      if (editing) await pruneUnavailable('mandate');
      closeModal();
      renderWorkspace();
      if (!editing) byId('base-allocation').focus();
      toast(editing ? 'Mandate updated.' : 'Proposal created.', 'success');
    } catch (error) {
      if (!modal) return;
      modal.busy = false;
      if (error.body && error.body.field) modal.data.errors[error.body.field] = error.message;
      else modal.data.banner = error.message;
      renderMandateModal();
    }
  }

  function renderBasisModal() {
    var root = byId('modal-root');
    var draft = modal.data.draft;
    var currencies = schema('options.currencies', []);
    var policies = schema('options.hedgingPolicies', []);
    var body = '<p class="modal-copy">Currency and hedging apply to every portfolio. Confirming waits for current work to settle, saves the basis, then rebuilds each available column.</p>'
      + '<form id="basis-form"><div class="form-grid"><div class="field"><label for="basis-currency">Currency</label><select id="basis-currency">'
      + currencies.map(function (value) { return '<option' + (value === draft.currency ? ' selected' : '') + '>' + esc(value) + '</option>'; }).join('')
      + '</select></div><div class="field"><label for="basis-hedging">Hedging policy</label><select id="basis-hedging">'
      + policies.map(function (value) { return '<option' + (value === draft.hedging ? ' selected' : '') + '>' + esc(value) + '</option>'; }).join('')
      + '</select></div></div>'
      + (state.columns.length ? '<div class="confirm-summary"><div><span>Current</span><strong>' + esc(state.basis.currency + ' · ' + state.basis.hedging) + '</strong></div><span aria-hidden="true">&#8594;</span><div><span>New</span><strong id="basis-preview">' + esc(draft.currency + ' · ' + draft.hedging) + '</strong></div></div>' : '')
      + (state.columns.length ? '<p class="form-banner" style="background:var(--warning-bg);border-color:var(--warning);color:var(--warning)">Changing basis rebuilds all portfolio analytics and clears attached sleeve choices.</p>' : '')
      + '</form>';
    var footer = '<button class="button button-secondary" type="button" data-modal-close>Cancel</button><button class="button button-primary" type="submit" form="basis-form"' + (modal.busy ? ' disabled' : '') + '>' + (modal.busy ? 'Applying…' : 'Confirm & rebuild') + '</button>';
    root.innerHTML = modalFrame('Scenario basis', 'Change currency or hedging', body, footer, false);
    ['basis-currency', 'basis-hedging'].forEach(function (id) {
      byId(id).addEventListener('change', function () {
        draft[id === 'basis-currency' ? 'currency' : 'hedging'] = byId(id).value;
        var preview = byId('basis-preview'); if (preview) preview.textContent = draft.currency + ' · ' + draft.hedging;
      });
    });
    byId('basis-form').addEventListener('submit', async function (event) {
      event.preventDefault();
      if (modal.busy) return;
      modal.busy = true; renderBasisModal();
      var next = Object.assign({}, draft);
      var applied = await applyBasis(next);
      if (modal && applied) closeModal();
      else if (modal) { modal.busy = false; renderBasisModal(); }
    });
  }

  function comparisonDraft() {
    return {allocation: '', excludeRE: false, excludeTAA: false, riskLevel: ''};
  }
  function renderComparisonModal() {
    var root = byId('modal-root');
    var draft = modal.data.draft;
    var allocations = schema('options.allocations', []);
    if (draft.allocation && !reAllowed(draft.allocation)) draft.excludeRE = true;
    var risks = risksFor(draft);
    if (draft.riskLevel && risks.indexOf(draft.riskLevel) < 0) draft.riskLevel = '';
    var valid = draft.allocation && draft.riskLevel && isAvailable(draft) && !usedKey(draft) && state.columns.length < maxPortfolios();
    var reason = '';
    if (draft.allocation && draft.riskLevel && usedKey(draft)) reason = 'This portfolio is already in the line-up.';
    else if (draft.allocation && !risks.length) reason = 'No stored portfolio matches this combination.';
    var body = '<p class="modal-copy">Add one purposeful alternative at a time. Every control describes the same four-part portfolio key used by the base.</p>'
      + '<form id="comparison-form"><div class="form-grid">'
      + '<div class="field"><label for="comparison-allocation">Allocation</label><select id="comparison-allocation" autofocus><option value="">Choose allocation…</option>'
      + allocations.map(function (value) { return '<option value="' + esc(value) + '"' + (value === draft.allocation ? ' selected' : '') + '>' + esc(value) + '</option>'; }).join('') + '</select></div>'
      + '<div class="field"><label for="comparison-risk">Risk level</label><select id="comparison-risk"' + (!draft.allocation ? ' disabled' : '') + '><option value="">Choose risk…</option>'
      + risks.map(function (value) { return '<option value="' + esc(value) + '"' + (value === draft.riskLevel ? ' selected' : '') + '>' + esc(value) + '</option>'; }).join('') + '</select></div>'
      + '<fieldset class="field-group"><legend>Real estate</legend><div class="check-field' + (!reAllowed(draft.allocation) ? ' disabled' : '') + '"><input type="checkbox" id="comparison-re"' + (draft.excludeRE ? ' checked' : '') + (!reAllowed(draft.allocation) ? ' disabled' : '') + '><label for="comparison-re">Exclude real estate</label></div></fieldset>'
      + '<fieldset class="field-group"><legend>Tactical allocation</legend><div class="check-field"><input type="checkbox" id="comparison-taa"' + (draft.excludeTAA ? ' checked' : '') + '><label for="comparison-taa">Exclude TAA</label></div></fieldset>'
      + '</div><div class="comparison-preview"><small>Portfolio to add</small><strong>' + esc(draft.allocation && draft.riskLevel ? keyName(draft, true) : 'Complete the selection') + '</strong>' + (reason ? '<span class="field-error">' + esc(reason) + '</span>' : '') + '</div></form>';
    var footer = '<button class="button button-secondary" type="button" data-modal-close>Cancel</button><button class="button button-primary" type="submit" form="comparison-form"' + (!valid ? ' disabled' : '') + '>Add comparison</button>';
    root.innerHTML = modalFrame('Portfolio line-up', 'Add a comparison', body, footer, true);
    ['comparison-allocation', 'comparison-risk', 'comparison-re', 'comparison-taa'].forEach(function (id) {
      byId(id).addEventListener('change', function () {
        if (id === 'comparison-allocation') {
          draft.allocation = byId(id).value; draft.riskLevel = '';
          if (!reAllowed(draft.allocation)) draft.excludeRE = true;
        } else if (id === 'comparison-risk') draft.riskLevel = byId(id).value;
        else if (id === 'comparison-re') draft.excludeRE = byId(id).checked;
        else draft.excludeTAA = byId(id).checked;
        preserveFocus(renderComparisonModal);
      });
    });
    byId('comparison-form').addEventListener('submit', function (event) {
      event.preventDefault();
      if (valid && addComparison(cloneKey(draft))) closeModal();
    });
  }

  function renderShell() {
    var fatal = byId('schema-error');
    var landing = byId('landing');
    var workspace = byId('workspace');
    if (state.schemaStatus === 'error') {
      fatal.hidden = false;
      landing.hidden = true;
      workspace.hidden = true;
      byId('schema-error-detail').textContent = state.schemaError || 'The schema request failed.';
      return;
    }
    fatal.hidden = true;
    landing.hidden = state.phase !== 'landing';
    workspace.hidden = state.phase !== 'workspace';
    if (state.schemaStatus === 'ready') {
      byId('landing-floor').textContent = 'Mandates from ' + money(Number(schema('rules.mandateFloor', 5000000)));
    }
    if (state.phase === 'workspace') renderWorkspace();
  }

  function renderWorkspace() {
    if (state.phase !== 'workspace') return;
    renderContext();
    renderStage();
    renderBaseBuilder();
    renderAllocationViews();
    renderImplementation();
    renderDataInfo();
  }

  function renderContext() {
    if (!state.mandate) return;
    byId('context-title').textContent = state.mandate.primaryPwa + ' proposal';
    var facts = [
      ['Mandate', money(state.mandate.mandateSize)],
      ['Top account', money(state.mandate.topAccountSize)],
      ['Currency', state.basis.currency],
      ['Hedging', state.basis.hedging]
    ];
    byId('context-facts').innerHTML = facts.map(function (item) {
      return '<div><dt>' + esc(item[0]) + '</dt><dd title="' + esc(item[1]) + '">' + esc(item[1]) + '</dd></div>';
    }).join('');
    byId('edit-mandate').disabled = !canEdit();
    byId('edit-basis').disabled = !canEdit();
    var info = schema('dataInfo', null);
    var badge = byId('data-badge');
    if (info) {
      badge.hidden = false;
      badge.textContent = (info.source || 'Analytics') + (info.asOf ? ' · as of ' + info.asOf : '');
    } else badge.hidden = true;
  }

  function renderStage() {
    var allocation = state.stage === 'allocation';
    byId('allocation-stage').hidden = !allocation;
    byId('implementation-stage').hidden = allocation;
    byId('tab-allocation').toggleAttribute('aria-current', allocation);
    byId('tab-implementation').toggleAttribute('aria-current', !allocation);
    if (allocation) byId('tab-allocation').setAttribute('aria-current', 'step');
    else byId('tab-implementation').setAttribute('aria-current', 'step');
  }

  function fieldOptions(values, selected, placeholder) {
    return '<option value="">' + esc(placeholder) + '</option>' + values.map(function (value) {
      return '<option value="' + esc(value) + '"' + (selected === value ? ' selected' : '') + '>' + esc(value) + '</option>';
    }).join('');
  }

  function renderBaseBuilder() {
    var host = byId('base-fields');
    if (!host || state.phase !== 'workspace') return;
    normaliseBaseDraft();
    var draft = state.baseDraft;
    var risks = risksFor(draft);
    var editDisabled = !canEdit() || state.busyBase;
    host.innerHTML = '<div class="field"><label for="base-allocation">Allocation</label><select id="base-allocation"' + (editDisabled ? ' disabled' : '') + '>'
      + fieldOptions(schema('options.allocations', []), draft.allocation, 'Choose allocation…') + '</select></div>'
      + '<fieldset class="field-group"><legend>Real estate</legend><div class="check-field' + (!draft.allocation || !reAllowed(draft.allocation) ? ' disabled' : '') + '"><input id="base-re" type="checkbox"' + (draft.excludeRE ? ' checked' : '') + (!draft.allocation || !reAllowed(draft.allocation) || editDisabled ? ' disabled' : '') + '><label for="base-re">Exclude real estate</label></div></fieldset>'
      + '<fieldset class="field-group"><legend>Tactical allocation</legend><div class="check-field' + (!draft.allocation ? ' disabled' : '') + '"><input id="base-taa" type="checkbox"' + (draft.excludeTAA ? ' checked' : '') + (!draft.allocation || editDisabled ? ' disabled' : '') + '><label for="base-taa">Exclude TAA</label></div></fieldset>'
      + '<div class="field"><label for="base-risk">Risk level</label><select id="base-risk"' + (!draft.allocation || editDisabled ? ' disabled' : '') + '>'
      + fieldOptions(risks, draft.riskLevel, draft.allocation ? 'Choose risk…' : 'Choose allocation first') + '</select></div>';
    var valid = draft.allocation && draft.riskLevel && isAvailable(draft) && canEdit();
    var base = baseColumn();
    var same = base && keyString(base.key) === keyString(draft);
    byId('base-action').innerHTML = '<button class="button button-primary" id="build-base" type="button"' + (!valid || state.busyBase || (same && base.status !== 'error') ? ' disabled' : '') + '>'
      + (state.busyBase ? 'Waiting…' : base ? 'Update base' : 'Build base') + '</button>';
  }

  function portfolioCard(column, index) {
    var title = column.data && (column.data.header || column.data.name) || keyName(column.key);
    var body = '';
    if (column.status === 'loading') {
      body = column.skeleton
        ? '<span class="skeleton line"></span><span class="skeleton line short"></span>'
        : '<p>Resolving analytics…</p>';
    } else if (column.status === 'error') {
      body = '<p>' + esc(column.error || 'Portfolio unavailable.') + '</p><button type="button" class="button button-danger button-small retry" data-retry-column="' + index + '">Retry</button>';
    } else {
      body = '<p>' + esc(state.basis.currency + ' · ' + state.basis.hedging) + '</p>';
    }
    return '<article class="portfolio-card ' + esc(column.role) + ' ' + esc(column.status) + '">'
      + '<span class="portfolio-role">' + (column.role === 'base' ? 'Proposed base' : 'Comparison ' + index) + '</span>'
      + '<h3>' + esc(title) + '</h3>' + body
      + (column.role === 'comparison' ? '<button class="icon-button" type="button" aria-label="Remove ' + esc(title) + '" data-remove-column="' + index + '">&times;</button>' : '')
      + '</article>';
  }

  function renderLineup() {
    var host = byId('portfolio-lineup');
    if (!host) return;
    byId('slot-count').textContent = occupiedSlots() + ' of ' + maxPortfolios()
      + (Object.keys(state.pendingDeletes).length ? ' · saving removal' : '');
    var html = state.columns.map(portfolioCard).join('');
    if (!state.columns.length) {
      html += '<article class="portfolio-card"><span class="portfolio-role">Start here</span><h3>No base built yet</h3><p>Complete the proposed portfolio controls above.</p></article>';
    }
    if (occupiedSlots() < maxPortfolios()) {
      var disabled = !baseColumn() || !canEdit() || Object.keys(state.pendingDeletes).length > 0;
      var remaining = maxPortfolios() - occupiedSlots();
      var disabledCopy = !baseColumn() ? 'Build a base first' : Object.keys(state.pendingDeletes).length ? 'Saving removal…' : '';
      html += '<button class="add-card" type="button" data-add-comparison' + (disabled ? ' disabled' : '') + '><span aria-hidden="true">+</span><span>Add comparison</span><small>' + (disabled ? disabledCopy : remaining + ' slot' + (remaining === 1 ? '' : 's') + ' available') + '</small></button>';
    }
    host.innerHTML = html;
  }

  function renderAllocationViews() {
    if (state.phase !== 'workspace') return;
    renderLineup();
    var hasColumns = state.columns.length > 0;
    byId('allocation-results').hidden = !hasColumns;
    if (!hasColumns) return;
    renderMetrics();
    renderAllocationTable();
    renderCompositionChart();
    renderRiskReturnChart();
    renderRiskTable();
    renderImplementationSummary();
  }

  function columnValue(column, accessor, formatter) {
    if (column.status === 'loading') return column.skeleton ? '<span class="skeleton line"></span>' : '<span class="muted-cell">Resolving…</span>';
    if (column.status === 'error') return '<span class="muted-cell">Unavailable</span>';
    var value = accessor(column.data);
    return formatter(value);
  }

  function renderMetrics() {
    var metrics = [
      ['Estimated return', 'estimatedReturnPct', function (value) { return fmt(value, 2, '%'); }],
      ['Volatility', 'volatilityPct', function (value) { return fmt(value, 2, '%'); }],
      ['Sharpe ratio', 'sharpe', function (value) { return fmt(value, 2, ''); }]
    ];
    byId('metric-grid').innerHTML = metrics.map(function (metric) {
      return '<article class="metric-card"><span>' + esc(metric[0]) + '</span><div class="metric-values" style="--columns:' + state.columns.length + '">'
        + state.columns.map(function (column, index) {
          var value = columnValue(column, function (data) { return data.metrics[metric[1]]; }, metric[2]);
          return '<div class="metric-value" style="--series:' + SERIES[index] + '"><strong>' + value + '</strong><small title="' + esc(keyName(column.key)) + '">' + esc(keyName(column.key)) + '</small></div>';
        }).join('') + '</div></article>';
    }).join('');
  }

  function categoryFor(data, name) {
    return (data.categories || []).find(function (category) { return category.name === name; }) || null;
  }
  function assetFor(category, name) {
    return category && (category.assets || []).find(function (asset) { return asset.reportingName === name; }) || null;
  }
  function categoryUnion() {
    var ordered = schema('categories', []).slice();
    readyColumns().forEach(function (column) {
      (column.data.categories || []).forEach(function (category) { if (ordered.indexOf(category.name) < 0) ordered.push(category.name); });
    });
    return ordered;
  }

  function renderAllocationTable() {
    var table = byId('allocation-table');
    var headers = '<caption class="sr-only">Portfolio allocations and estimated outcomes</caption><thead><tr><th scope="col">Category &amp; asset class</th>'
      + state.columns.map(function (column, index) { return '<th scope="col" class="' + (index === 0 ? 'base-col' : '') + '"><span class="portfolio-role">' + (index === 0 ? 'Base' : 'Comparison') + '</span>' + esc(keyName(column.key)) + '</th>'; }).join('') + '</tr></thead>';
    var body = '<tbody>';
    categoryUnion().forEach(function (categoryName) {
      body += '<tr class="category-row"><th scope="row">' + esc(categoryName) + '</th>';
      state.columns.forEach(function (column, index) {
        body += '<td class="' + (index === 0 ? 'base-col' : '') + '">' + columnValue(column, function (data) {
          var category = categoryFor(data, categoryName); return category ? category.weightPct : null;
        }, function (value) { return value === null ? '' : fmtWeight(value, 1); }) + '</td>';
      });
      body += '</tr>';
      var assets = [];
      readyColumns().forEach(function (column) {
        var category = categoryFor(column.data, categoryName);
        (category ? category.assets : []).forEach(function (asset) { if (assets.indexOf(asset.reportingName) < 0) assets.push(asset.reportingName); });
      });
      assets.forEach(function (assetName) {
        body += '<tr class="asset-row"><th scope="row">' + esc(assetName) + '</th>';
        state.columns.forEach(function (column, index) {
          body += '<td class="' + (index === 0 ? 'base-col' : '') + '">' + columnValue(column, function (data) {
            var category = categoryFor(data, categoryName);
            if (!category) return null;
            var asset = assetFor(category, assetName); return asset ? asset.weightPct : 0;
          }, function (value) { return value === null ? '' : fmtWeight(value, 1); }) + '</td>';
        });
        body += '</tr>';
      });
    });
    body += '<tr class="category-row"><th scope="row">Estimated outcomes</th>' + state.columns.map(function (_, index) { return '<td class="' + (index === 0 ? 'base-col' : '') + '"></td>'; }).join('') + '</tr>';
    [['Estimated return', 'estimatedReturnPct', '%'], ['Volatility', 'volatilityPct', '%'], ['Sharpe ratio', 'sharpe', '']].forEach(function (row) {
      body += '<tr class="metric-row"><th scope="row">' + esc(row[0]) + '</th>' + state.columns.map(function (column, index) {
        return '<td class="' + (index === 0 ? 'base-col' : '') + '">' + columnValue(column, function (data) { return data.metrics[row[1]]; }, function (value) { return fmt(value, 2, row[2]); }) + '</td>';
      }).join('') + '</tr>';
    });
    table.innerHTML = headers + body + '</tbody>';
  }

  function categoryColour(name) {
    var index = categoryUnion().indexOf(name);
    return CATEGORY[index < 0 ? 0 : index % CATEGORY.length];
  }
  function segmentTextColour(background) {
    var channels = String(background).replace('#', '').match(/.{2}/g).map(function (part) {
      var value = parseInt(part, 16) / 255;
      return value <= .03928 ? value / 12.92 : Math.pow((value + .055) / 1.055, 2.4);
    });
    var luminance = .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
    var whiteContrast = 1.05 / (luminance + .05);
    var navyLuminance = .016;
    var navyContrast = (luminance + .05) / (navyLuminance + .05);
    return whiteContrast >= navyContrast ? '#ffffff' : '#12233f';
  }

  function renderCompositionChart() {
    var host = byId('composition-chart');
    var ready = readyColumns();
    if (!ready.length) {
      host.innerHTML = '<p class="field-help">Allocation composition appears when a portfolio is ready.</p>';
      return;
    }
    var categories = categoryUnion().filter(function (name) {
      return ready.some(function (column) { return !!categoryFor(column.data, name); });
    });
    var legend = '<div class="chart-legend">' + categories.map(function (name) {
      return '<span style="--swatch:' + categoryColour(name) + '"><i></i>' + esc(name) + '</span>';
    }).join('') + '</div>';
    var rows = '<div class="stack-chart">' + ready.map(function (column) {
      var segments = categories.map(function (name) {
        var category = categoryFor(column.data, name);
        if (!category || !finite(category.weightPct) || category.weightPct <= 0) return '';
        var label = category.weightPct >= 11 ? '<span>' + fmt(category.weightPct, 0, '%') + '</span>' : '';
        var colour = categoryColour(name);
        return '<div class="stack-segment" style="width:' + Math.max(0, category.weightPct) + '%;--segment:' + colour + ';--segment-label:' + segmentTextColour(colour) + '" title="' + esc(name + ': ' + fmt(category.weightPct, 1, '%')) + '">' + label + '</div>';
      }).join('');
      return '<div class="stack-row"><span class="stack-label" title="' + esc(column.data.name) + '">' + esc(column.data.header) + '</span><div class="stack-bar" role="img" aria-label="' + esc(column.data.header + ' allocation composition') + '">' + segments + '</div></div>';
    }).join('') + '</div>';
    host.innerHTML = legend + rows;
  }

  function renderRiskReturnChart() {
    var host = byId('risk-return-chart');
    var ready = readyColumns();
    if (!ready.length) {
      host.innerHTML = '<p class="field-help">Risk-return positions appear when a portfolio is ready.</p>';
      return;
    }
    var points = ready.map(function (column, index) {
      return {column: column, index: state.columns.indexOf(column), x: column.data.metrics.volatilityPct, y: column.data.metrics.estimatedReturnPct};
    }).filter(function (point) { return finite(point.x) && finite(point.y); });
    if (!points.length) { host.textContent = 'No finite risk-return values are available.'; return; }
    var xs = points.map(function (point) { return point.x; });
    var ys = points.map(function (point) { return point.y; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs), minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var padX = Math.max(.7, (maxX - minX) * .25), padY = Math.max(.5, (maxY - minY) * .35);
    minX = Math.max(0, minX - padX); maxX += padX; minY -= padY; maxY += padY;
    function x(value) { return 55 + (value - minX) / Math.max(.01, maxX - minX) * 430; }
    function y(value) { return 225 - (value - minY) / Math.max(.01, maxY - minY) * 175; }
    var svg = '<svg class="rr-svg" viewBox="0 0 520 260" role="img" aria-labelledby="rr-title rr-desc"><title id="rr-title">Return against volatility</title><desc id="rr-desc">' + esc(points.map(function (point) { return point.column.data.header + ': ' + fmt(point.x, 2, '%') + ' volatility and ' + fmt(point.y, 2, '%') + ' estimated return'; }).join('. ')) + '</desc>';
    for (var i = 0; i < 5; i++) {
      var gx = 55 + i * 107.5, gy = 50 + i * 43.75;
      svg += '<line class="grid" x1="' + gx + '" y1="50" x2="' + gx + '" y2="225"></line><line class="grid" x1="55" y1="' + gy + '" x2="485" y2="' + gy + '"></line>';
      svg += '<text x="' + gx + '" y="244" text-anchor="middle">' + fmt(minX + i * (maxX - minX) / 4, 1, '%') + '</text>';
      svg += '<text x="47" y="' + (gy + 3) + '" text-anchor="end">' + fmt(maxY - i * (maxY - minY) / 4, 1, '%') + '</text>';
    }
    svg += '<line class="axis" x1="55" y1="225" x2="485" y2="225"></line><line class="axis" x1="55" y1="50" x2="55" y2="225"></line><text x="270" y="258" text-anchor="middle">Volatility</text>';
    points.forEach(function (point) {
      var px = x(point.x), py = y(point.y), anchor = px > 390 ? 'end' : 'start', offset = anchor === 'end' ? -10 : 10;
      svg += '<circle class="point" cx="' + px + '" cy="' + py + '" r="7" fill="' + SERIES[point.index] + '"></circle>'
        + '<text class="label" x="' + (px + offset) + '" y="' + (py - 10) + '" text-anchor="' + anchor + '">' + esc(point.column.data.header) + '</text>';
    });
    host.innerHTML = svg + '</svg>';
  }

  function unionLabels(field, labelField) {
    var labels = [];
    readyColumns().forEach(function (column) {
      (column.data[field] || []).forEach(function (entry) {
        var label = entry[labelField]; if (labels.indexOf(label) < 0) labels.push(label);
      });
    });
    return labels;
  }
  function entryFor(column, field, labelField, label) {
    return column.status === 'ready' ? (column.data[field] || []).find(function (entry) { return entry[labelField] === label; }) : null;
  }
  function pairCells(column, entry, signed) {
    if (column.status === 'loading') {
      var loading = column.skeleton ? '<span class="skeleton line"></span>' : '<span class="muted-cell">Resolving…</span>';
      return '<td>' + loading + '</td><td>' + loading + '</td>';
    }
    if (column.status === 'error' || !entry) return '<td class="muted-cell">—</td><td class="muted-cell">—</td>';
    function cell(value) {
      var className = signed && finite(value) ? (value < 0 ? 'negative' : value > 0 ? 'positive' : '') : '';
      return '<td class="' + className + '">' + fmt(value, 2, '%') + '</td>';
    }
    return cell(entry.nominalPct) + cell(entry.realPct);
  }

  function renderRiskTable() {
    var table = byId('risk-table');
    var top = '<caption class="sr-only">Risk metrics, historical stress periods, and risk premia</caption><thead><tr><th scope="col" rowspan="2">Measure</th>'
      + state.columns.map(function (column) { return '<th scope="colgroup" colspan="2">' + esc(keyName(column.key)) + '</th>'; }).join('') + '</tr><tr class="subhead">'
      + state.columns.map(function () { return '<th scope="col">Nominal</th><th scope="col">Real</th>'; }).join('') + '</tr></thead><tbody>';
    var body = '<tr class="section-row"><th colspan="' + (1 + state.columns.length * 2) + '">Portfolio analytics</th></tr>';
    [['Estimated return', 'estimatedReturnPct', '%'], ['Volatility', 'volatilityPct', '%'], ['Sharpe ratio', 'sharpe', '']].forEach(function (metric) {
      body += '<tr class="metric-row"><th scope="row">' + esc(metric[0]) + '</th>' + state.columns.map(function (column) {
        if (column.status === 'loading') return '<td colspan="2">' + (column.skeleton ? '<span class="skeleton line"></span>' : '<span class="muted-cell">Resolving…</span>') + '</td>';
        if (column.status === 'error') return '<td colspan="2" class="muted-cell">Unavailable</td>';
        return '<td colspan="2">' + fmt(column.data.metrics[metric[1]], 2, metric[2]) + '</td>';
      }).join('') + '</tr>';
    });
    body += '<tr class="section-row"><th colspan="' + (1 + state.columns.length * 2) + '">Historical stress periods</th></tr>';
    var stress = unionLabels('stress', 'period');
    if (!stress.length) body += '<tr><th scope="row">No periods available</th><td colspan="' + state.columns.length * 2 + '" class="muted-cell">—</td></tr>';
    stress.forEach(function (period) {
      body += '<tr><th scope="row">' + esc(period) + '</th>' + state.columns.map(function (column) { return pairCells(column, entryFor(column, 'stress', 'period', period), true); }).join('') + '</tr>';
    });
    body += '<tr class="section-row"><th colspan="' + (1 + state.columns.length * 2) + '">Risk premia</th></tr>';
    var groups = unionLabels('premia', 'group');
    groups.forEach(function (group) {
      body += '<tr class="category-row"><th scope="row">' + esc(group) + '</th>' + state.columns.map(function () { return '<td colspan="2"></td>'; }).join('') + '</tr>';
      var horizons = [];
      readyColumns().forEach(function (column) {
        (column.data.premia || []).forEach(function (entry) { if (entry.group === group && horizons.indexOf(entry.horizon) < 0) horizons.push(entry.horizon); });
      });
      horizons.forEach(function (horizon) {
        body += '<tr><th scope="row" style="padding-left:28px">' + esc(horizon) + '</th>' + state.columns.map(function (column) {
          var entry = column.status === 'ready' ? (column.data.premia || []).find(function (item) { return item.group === group && item.horizon === horizon; }) : null;
          return pairCells(column, entry, entry && entry.kind === 'loss');
        }).join('') + '</tr>';
      });
    });
    table.innerHTML = top + body + '</tbody>';
  }

  function implementationCounts() {
    var base = baseColumn();
    if (!base || base.status !== 'ready') return {filled: 0, total: 0, missing: []};
    var categories = base.data.categories || [];
    var missing = categories.filter(function (category) { return !sleeveFor(category.name); }).map(function (category) { return category.name; });
    return {filled: categories.length - missing.length, total: categories.length, missing: missing};
  }

  function implementationComplete() {
    var counts = implementationCounts();
    return counts.total > 0 && counts.filled === counts.total && (!needsVariant() || !!state.variant);
  }

  function roundWeights(exact) {
    if (!exact.length) return [];
    var units = exact.map(function (weight) { return Math.floor(weight * 100 + 1e-9); });
    var shortfall = 10000 - units.reduce(function (sum, value) { return sum + value; }, 0);
    var order = exact.map(function (weight, index) { return {index: index, remainder: weight * 100 - units[index]}; })
      .sort(function (a, b) { return (b.remainder - a.remainder) || (a.index - b.index); });
    for (var i = 0; i < Math.max(0, shortfall); i++) units[order[i % order.length].index] += 1;
    return units.map(function (value) { return value / 100; });
  }

  function implementationGroups() {
    var base = baseColumn();
    if (!base || base.status !== 'ready') return [];
    var groups = [];
    var products = [];
    (base.data.categories || []).forEach(function (category) {
      var sleeve = sleeveFor(category.name);
      var items = [];
      if (sleeve) {
        (sleeve.products || []).forEach(function (product) {
          var item = {
            name: product.name, ticker: product.ticker, assetClass: product.assetClass,
            style: product.style, vehicle: product.vehicle, source: product.source,
            liquidity: product.liquidity, exposureCurrency: product.exposureCurrency,
            productCost: Number(product.productCost), managementFee: Number(product.managementFee),
            exact: category.weightPct * Number(product.weight)
          };
          items.push(item); products.push(item);
        });
      }
      groups.push({category: category.name, categoryWeight: category.weightPct, sleeve: sleeve, auto: autoCategories().indexOf(category.name) >= 0, items: items});
    });
    var printed = implementationComplete()
      ? roundWeights(products.map(function (product) { return product.exact; }))
      : products.map(function (product) { return Math.round(product.exact * 100) / 100; });
    products.forEach(function (product, index) {
      product.weight = printed[index];
      product.weightedFee = (product.productCost + product.managementFee) * product.weight;
      product.notional = Math.round(state.mandate.mandateSize * product.weight / 100 / 100) * 100;
    });
    return groups;
  }

  function renderImplementation() {
    if (state.phase !== 'workspace') return;
    var base = baseColumn();
    var empty = byId('implementation-empty');
    var content = byId('implementation-content');
    if (!base) {
      empty.hidden = false; content.hidden = true;
      empty.innerHTML = '<h2>Build a base portfolio first</h2><p>Implementation begins with the categories and weights of the proposed base. Return to Allocation to build it.</p><button class="button button-primary" type="button" data-go-allocation>Go to allocation</button>';
      return;
    }
    if (base.status === 'loading') {
      empty.hidden = false; content.hidden = true;
      empty.innerHTML = '<h2>Resolving the proposed base</h2><p>The implementation workbench will open as soon as the portfolio categories are ready.</p><span class="skeleton line" style="max-width:340px;margin:18px auto 0"></span>';
      return;
    }
    if (base.status === 'error') {
      empty.hidden = false; content.hidden = true;
      empty.innerHTML = '<h2>The proposed base is unavailable</h2><p>' + esc(base.error || 'The portfolio could not be resolved.') + '</p><button class="button button-danger" type="button" data-retry-base>Retry base</button>';
      return;
    }
    empty.hidden = true; content.hidden = false;
    renderVariant();
    renderSleeves();
    renderImplementationTable();
    renderImplementationSummary();
    if (state.stage === 'implementation') ensureSleeveLibraries();
  }

  function renderVariant() {
    var panel = byId('variant-panel');
    var options = variantOptions();
    if (!options.length) { panel.hidden = true; return; }
    panel.hidden = false;
    panel.innerHTML = '<div><p class="eyebrow">Required first</p><h2 id="variant-heading">Choose an implementation route</h2><p>The route controls which sleeves and product vehicles are available. Changing it clears prior sleeve choices.</p></div>'
      + '<fieldset class="variant-options"><legend class="sr-only">Implementation route</legend>'
      + options.map(function (option, index) {
        return '<div class="variant-option"><input type="radio" name="implementation-variant" id="variant-' + index + '" value="' + esc(option) + '"' + (state.variant === option ? ' checked' : '') + (!canEdit() || state.variantBusy ? ' disabled' : '') + '><label for="variant-' + index + '">' + esc(option) + '</label></div>';
      }).join('') + '</fieldset>';
  }

  function renderSleeves() {
    var host = byId('sleeve-grid');
    var base = baseColumn();
    if (!host || !base || base.status !== 'ready') return;
    var counts = implementationCounts();
    var pct = counts.total ? Math.round(counts.filled / counts.total * 100) : 0;
    byId('completion-summary').innerHTML = '<div class="completion-line"><span>Implementation progress</span><strong>' + counts.filled + ' of ' + counts.total + '</strong></div><div class="progress-track" aria-label="' + counts.filled + ' of ' + counts.total + ' categories implemented"><i style="--progress:' + pct + '%"></i></div>';
    if (needsVariant() && !state.variant) {
      host.innerHTML = '<article class="sleeve-card unavailable" style="grid-column:1/-1"><div class="sleeve-card-head"><h3>Sleeves unlock after an implementation route is selected</h3></div><p class="sleeve-card-status">Choose one of the routes above. No default is assumed because the route changes both product availability and composition.</p></article>';
      return;
    }
    host.innerHTML = (base.data.categories || []).map(function (category, index) {
      var library = state.sleeveLib[category.name];
      var selected = sleeveFor(category.name);
      var auto = autoCategories().indexOf(category.name) >= 0;
      var classes = 'sleeve-card' + (selected ? ' complete' : '') + (auto ? ' auto' : '');
      var control = '';
      if (!library || library.status === 'loading') {
        control = '<span class="skeleton line"></span><span class="skeleton line short"></span><p class="sleeve-card-status">Loading available sleeves…</p>';
      } else if (library.status === 'error') {
        control = '<p class="field-error">' + esc(library.error) + '</p><button class="button button-danger button-small retry" type="button" data-retry-sleeve="' + esc(category.name) + '">Retry</button>';
      } else if (!library.sleeves.length) {
        control = '<p class="sleeve-card-status">No sleeves are offered for this category under ' + esc(state.variant || 'the current basis') + '.</p>';
        classes += ' unavailable';
      } else if (auto) {
        control = '<div class="auto-name">' + esc(library.sleeves[0].name) + '</div><p class="sleeve-card-status"><span class="locked-mark">&#128274; Attached automatically</span></p>';
      } else {
        control = '<div class="field"><label class="sr-only" for="sleeve-' + index + '">Sleeve for ' + esc(category.name) + '</label><select id="sleeve-' + index + '" data-sleeve-category="' + esc(category.name) + '"' + (!canEdit() ? ' disabled' : '') + '><option value="">Choose a sleeve…</option>'
          + library.sleeves.map(function (sleeve) { return '<option value="' + esc(sleeve.name) + '"' + (state.sleeves[category.name] === sleeve.name ? ' selected' : '') + '>' + esc(sleeve.name) + '</option>'; }).join('')
          + '</select></div><p class="sleeve-card-status">' + (selected ? selected.products.length + ' product' + (selected.products.length === 1 ? '' : 's') + ' in this sleeve' : 'Selection required') + '</p>';
      }
      return '<article class="' + classes + '"><div class="sleeve-card-head"><h3>' + esc(category.name) + '</h3><span class="category-weight">' + fmt(category.weightPct, 1, '%') + '</span></div>' + control + '</article>';
    }).join('');
  }

  function productPill(value) {
    var className = String(value).toLowerCase() === 'active' ? 'pill-active'
      : String(value).toLowerCase() === 'passive' ? 'pill-passive'
      : String(value).toLowerCase() === 'internal' ? 'pill-internal'
      : String(value).toLowerCase() === 'external' ? 'pill-external' : '';
    return '<span class="pill ' + className + '">' + esc(value) + '</span>';
  }

  function renderImplementationTable() {
    var table = byId('implementation-table');
    var groups = implementationGroups();
    var totals = {weight: 0, fee: 0, notional: 0};
    var html = '<caption class="sr-only">Product-level implementation model</caption><thead><tr><th scope="col">Category &amp; asset class</th><th scope="col">Products</th><th scope="col">Allocation</th><th scope="col">Ticker</th><th scope="col">Style</th><th scope="col">Vehicle</th><th scope="col">Source</th><th scope="col">Liquidity</th><th scope="col">Exposure ccy</th><th scope="col">Cost</th><th scope="col">Mgmt fee</th><th scope="col">Wtd fee</th><th scope="col">Notional</th></tr></thead><tbody>';
    groups.forEach(function (group) {
      var groupTotals = {weight: 0, fee: 0, notional: 0};
      group.items.forEach(function (item) {
        groupTotals.weight += item.weight; groupTotals.fee += item.weightedFee; groupTotals.notional += item.notional;
        totals.weight += item.weight; totals.fee += item.weightedFee; totals.notional += item.notional;
      });
      var shownWeight = group.items.length ? groupTotals.weight : group.categoryWeight;
      var shownNotional = group.items.length ? groupTotals.notional : Math.round(state.mandate.mandateSize * group.categoryWeight / 100 / 100) * 100;
      html += '<tr class="group-row"><th scope="row">' + esc(group.category) + '</th><td>'
        + (group.sleeve ? '<span class="pill">' + esc(group.sleeve.name) + (group.auto ? ' · auto' : '') + '</span>' : '<span class="field-error">No sleeve attached</span>')
        + '</td><td>' + fmt(shownWeight, 2, '%') + '</td><td colspan="8"></td><td>' + (group.items.length ? fmt(groupTotals.fee, 1, 'bp') : '') + '</td><td>' + money(shownNotional) + '</td></tr>';
      group.items.forEach(function (item) {
        html += '<tr class="product-row"><th scope="row">' + esc(item.assetClass) + '</th><td>' + esc(item.name) + '</td><td>' + fmt(item.weight, 2, '%') + '</td><td>' + esc(item.ticker) + '</td><td>' + productPill(item.style) + '</td><td>' + productPill(item.vehicle) + '</td><td>' + productPill(item.source) + '</td><td>' + esc(item.liquidity) + '</td><td>' + esc(item.exposureCurrency) + '</td><td>' + fmt(item.productCost, 2, '%') + '</td><td>' + fmt(item.managementFee, 2, '%') + '</td><td>' + fmt(item.weightedFee, 1, 'bp') + '</td><td>' + money(item.notional) + '</td></tr>';
      });
    });
    html += '<tr class="grand-row"><th scope="row">Total</th><td></td><td>' + fmt(totals.weight, 2, '%') + '</td><td colspan="8"></td><td>' + fmt(totals.fee, 1, 'bp') + '</td><td>' + money(totals.notional) + '</td></tr></tbody>';
    table.innerHTML = html;
    var base = baseColumn();
    byId('implementation-caption').textContent = 'Implementing ' + (base.data.name || keyName(base.key, true)) + ' against ' + money(state.mandate.mandateSize) + '.';
  }

  function exportGate() {
    if (!canExport()) return 'Export is not available for your role.';
    var base = baseColumn();
    if (!base || base.status !== 'ready') return 'A resolved base portfolio is required.';
    if (needsVariant() && !state.variant) return 'Choose an implementation route first.';
    var counts = implementationCounts();
    if (!implementationComplete()) {
      if (counts.missing.length) return 'Attach a sleeve to every category — missing ' + counts.missing.join(', ') + '.';
      return 'Complete the implementation before export.';
    }
    if (!allColumnsSettled()) return 'Wait for every portfolio column to resolve.';
    if (state.pendingWrites || trackedMutations.size) return 'Saving the latest scenario changes…';
    if (state.syncError) return 'A scenario change was not saved: ' + state.syncError;
    if (!navigator.onLine) return 'Reconnect to generate the workbook.';
    return '';
  }

  function renderImplementationSummary() {
    var gate = byId('export-gate');
    var button = byId('export-workbook');
    if (!gate || !button) return;
    var reason = exportGate();
    if (state.exportStatus === 'working') {
      gate.className = 'export-gate'; gate.textContent = 'The server is generating the workbook from the persisted scenario.';
      button.disabled = true; button.textContent = 'Preparing workbook…';
      return;
    }
    if (state.exportError) {
      gate.className = 'export-gate error'; gate.textContent = state.exportError;
    } else if (reason) {
      gate.className = 'export-gate blocked'; gate.textContent = reason;
    } else {
      gate.className = 'export-gate ready'; gate.textContent = 'Ready — all categories are implemented and scenario changes are saved.';
    }
    var canRetrySync = !!state.syncError && navigator.onLine && state.exportStatus !== 'working';
    button.disabled = (!canRetrySync && !!reason) || state.exportStatus === 'working';
    button.textContent = canRetrySync ? 'Retry saving' : 'Download Excel';
    button.title = canRetrySync ? 'Retry the current implementation save' : (reason || 'Download the proposal workbook');
  }

  async function retryImplementationSync() {
    if (!state.syncError || !navigator.onLine || state.pendingWrites) return;
    var payload = {sleeves: sleeveSnapshot()};
    if (needsVariant() && state.variant) payload.variant = state.variant;
    try {
      await serialWrite(payload);
      toast('Scenario changes saved.', 'success');
    } catch (ignore) {}
    renderImplementationSummary();
  }

  async function exportWorkbook() {
    if (state.exportStatus === 'working' || exportGate()) return;
    state.exportStatus = 'working'; state.exportError = null; renderImplementationSummary();
    try {
      await flushWrites();
      var result = await apiRequest('/scenario/' + encodeURIComponent(state.scenarioId) + '/export', {method: 'POST'}, 'blob');
      var disposition = result.response.headers.get('Content-Disposition') || '';
      var match = disposition.match(/filename\*?=(?:UTF-8''|\")?([^\";]+)/i);
      var filename = match ? decodeURIComponent(match[1].replace(/\"$/g, '')) : 'EpsilonPhi_Scenario.xlsx';
      var url = URL.createObjectURL(result.blob);
      var link = document.createElement('a');
      link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 5000);
      state.exportStatus = 'idle';
      toast('Workbook downloaded.', 'success');
      announce('polite', 'Proposal workbook downloaded.');
    } catch (error) {
      state.exportStatus = 'error'; state.exportError = error.message;
      announce('assertive', 'Workbook export failed. ' + error.message);
    }
    renderImplementationSummary();
  }

  function renderDataInfo() {
    var info = schema('dataInfo', null);
    byId('data-info').textContent = info ? ['Data: ' + (info.source || 'scenario analytics'), info.dataversion, info.asOf ? 'as of ' + info.asOf : '', info.adapter ? 'adapter: ' + info.adapter : ''].filter(Boolean).join(' · ') : '';
  }

  async function hydrateScenario(stored) {
    state.scenarioId = stored.id;
    state.mandate = stored.mandate;
    state.basis = stored.basis;
    state.variant = stored.variant || null;
    state.sleeves = Object.assign({}, stored.sleeves || {});
    state.sleeveLib = {};
    state.phase = 'workspace';
    state.stage = 'allocation';
    await loadSchema(false);
    if (needsVariant() && state.variant && variantOptions().indexOf(state.variant) < 0) {
      state.variant = null;
      state.sleeves = {};
      announce('assertive', 'The stored implementation route is no longer available. Choose a new route.');
    }
    var baseKey = parseKey(stored.base);
    if (baseKey && isAvailable(baseKey)) {
      state.baseDraft = cloneKey(baseKey);
      state.columns.push(createColumn(baseKey, 'base'));
      (stored.comparisons || []).slice(0, maxPortfolios() - 1).forEach(function (value) {
        var key = parseKey(value);
        if (key && isAvailable(key) && keyString(key) !== keyString(baseKey)) state.columns.push(createColumn(key, 'comparison'));
        else if (key) {
          state.pendingDeletes[keyString(key)] = true;
          trackMutation(compensateDelete(key));
        }
      });
    } else if (stored.base) {
      state.sleeves = {};
      announce('assertive', 'The stored base is unavailable under this mandate. Choose a new base portfolio.');
    }
    renderWorkspace();
    state.columns.forEach(resolveColumn);
  }

  async function boot() {
    updateNetworkStatus();
    var start = byId('start-scenario');
    start.disabled = true;
    var wanted = new URL(window.location.href).searchParams.get('scenario');
    try {
      if (wanted) {
        var stored = await apiRequest('/scenario/' + encodeURIComponent(wanted));
        await hydrateScenario(stored);
      } else {
        await loadSchema(false);
      }
    } catch (error) {
      if (wanted && error.status === 404) {
        var clean = new URL(window.location.href); clean.searchParams.delete('scenario'); history.replaceState({}, '', clean);
        state.scenarioId = null; state.mandate = null; state.phase = 'landing'; state.columns = [];
        try { await loadSchema(false); } catch (ignore) {}
        showAlert(error.message, 'error');
      } else if (state.schemaStatus !== 'error') {
        state.schemaStatus = 'error'; state.schemaError = error.message;
      }
    } finally {
      start.disabled = state.schemaStatus !== 'ready' || !canEdit();
      renderShell();
    }
    registerPwa();
  }

  function setStage(stage, focus) {
    if (stage !== 'allocation' && stage !== 'implementation') return;
    state.stage = stage;
    renderStage();
    if (stage === 'implementation') {
      renderImplementation();
      ensureSleeveLibraries();
    }
    if (focus !== false) {
      var heading = byId(stage === 'allocation' ? 'allocation-heading' : 'implementation-heading');
      heading.setAttribute('tabindex', '-1'); heading.focus({preventScroll: true});
      heading.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  }

  function mandateDraft() {
    var current = state.mandate || {};
    return {
      topAccountSize: current.topAccountSize ? Number(current.topAccountSize).toLocaleString('en-US') : '',
      mandateSize: current.mandateSize ? Number(current.mandateSize).toLocaleString('en-US') : '',
      primaryPwa: current.primaryPwa || '', selectedPwa: current.primaryPwa || ''
    };
  }

  document.addEventListener('click', function (event) {
    var target = event.target;
    if (target.closest('[data-dismiss-alert]')) { clearAlert(); return; }
    if (target.closest('[data-modal-close]')) { if (!modal || !modal.busy) closeModal(); return; }
    if (target.matches('[data-modal-backdrop]') && modal && !modal.busy) { closeModal(); return; }
    if (target.closest('#start-scenario')) {
      if (state.schemaStatus === 'ready') openModal('mandate', {draft: mandateDraft(), errors: {}, matches: null, activeAdvisor: -1});
      return;
    }
    if (target.closest('#edit-mandate')) { openModal('mandate', {draft: mandateDraft(), errors: {}, matches: null, activeAdvisor: -1}); return; }
    if (target.closest('#edit-basis')) { openModal('basis', {draft: Object.assign({}, state.basis)}); return; }
    var stageButton = target.closest('[data-stage]');
    if (stageButton) { setStage(stageButton.dataset.stage); return; }
    if (target.closest('[data-go-implementation]')) { setStage('implementation'); return; }
    if (target.closest('[data-go-allocation]')) { setStage('allocation'); return; }
    if (target.closest('#build-base')) { setBase(cloneKey(state.baseDraft)); return; }
    if (target.closest('[data-add-comparison]')) { openModal('comparison', {draft: comparisonDraft()}); return; }
    var remove = target.closest('[data-remove-column]');
    if (remove) { removeComparison(state.columns[Number(remove.dataset.removeColumn)]); return; }
    var retry = target.closest('[data-retry-column]');
    if (retry) { retryColumn(state.columns[Number(retry.dataset.retryColumn)]); return; }
    if (target.closest('[data-retry-base]')) { retryColumn(baseColumn()); return; }
    var sleeveRetry = target.closest('[data-retry-sleeve]');
    if (sleeveRetry) { loadSleeveLibrary(sleeveRetry.dataset.retrySleeve, true); return; }
    if (target.closest('#export-workbook')) {
      if (state.syncError) retryImplementationSync();
      else exportWorkbook();
      return;
    }
    if (target.closest('#schema-retry')) {
      state.schemaStatus = 'loading'; renderShell();
      loadSchema(false).then(function () { byId('start-scenario').disabled = !canEdit(); }).catch(function () {});
      return;
    }
    if (target.closest('#install-app') && state.installPrompt) {
      state.installPrompt.prompt();
      state.installPrompt.userChoice.finally(function () { state.installPrompt = null; byId('install-app').hidden = true; });
    }
  });

  document.addEventListener('change', function (event) {
    var target = event.target;
    if (target.id === 'base-allocation') {
      state.baseDraft.allocation = target.value;
      state.baseDraft.riskLevel = '';
      if (!reAllowed(target.value)) state.baseDraft.excludeRE = true;
      else if (target.value) state.baseDraft.excludeRE = false;
      preserveFocus(renderBaseBuilder);
    } else if (target.id === 'base-re') {
      state.baseDraft.excludeRE = target.checked; state.baseDraft.riskLevel = '';
      preserveFocus(renderBaseBuilder);
    } else if (target.id === 'base-taa') {
      state.baseDraft.excludeTAA = target.checked; state.baseDraft.riskLevel = '';
      preserveFocus(renderBaseBuilder);
    } else if (target.id === 'base-risk') {
      state.baseDraft.riskLevel = target.value; preserveFocus(renderBaseBuilder);
    } else if (target.matches('[name="implementation-variant"]')) {
      chooseVariant(target.value);
    } else if (target.dataset.sleeveCategory !== undefined) {
      chooseSleeve(target.dataset.sleeveCategory, target.value || null);
    }
  });

  function updateNetworkStatus() {
    var host = byId('network-state');
    if (!host) return;
    var online = navigator.onLine;
    host.className = 'network-state ' + (online ? 'online' : 'offline');
    host.querySelector('span').textContent = online ? 'Connected' : 'Offline';
    renderImplementationSummary();
  }
  window.addEventListener('online', function () { updateNetworkStatus(); toast('Connection restored.', 'success'); });
  window.addEventListener('offline', function () { updateNetworkStatus(); toast('You are offline. Saved app screens remain available; scenario actions need a connection.', 'error'); });
  window.addEventListener('proposaltoolv2:auth', function (event) {
    if (event.detail && event.detail.status === 'authenticated') updateNetworkStatus();
  });
  window.addEventListener('beforeinstallprompt', function (event) {
    event.preventDefault(); state.installPrompt = event; byId('install-app').hidden = false;
  });
  window.addEventListener('appinstalled', function () { state.installPrompt = null; byId('install-app').hidden = true; toast('Proposal Tool installed.', 'success'); });

  function registerPwa() {
    if (!('serviceWorker' in navigator) || window.location.protocol === 'file:') return;
    navigator.serviceWorker.register('service-worker.js', {scope: './'}).catch(function (error) {
      console.warn('Proposal Tool service worker registration failed:', error.message);
    });
  }

  window.ProposalToolV2 = {
    snapshot: function () {
      return {
        phase: state.phase, stage: state.stage, schemaStatus: state.schemaStatus,
        scenarioId: state.scenarioId, mandate: state.mandate, basis: Object.assign({}, state.basis),
        columns: state.columns.map(function (column) { return {key: keyString(column.key), role: column.role, status: column.status, error: column.error}; }),
        variant: state.variant, sleeves: Object.assign({}, state.sleeves),
        implementation: implementationCounts(), pendingWrites: state.pendingWrites,
        syncError: state.syncError, exportStatus: state.exportStatus
      };
    },
    waitForSettled: async function () { await flushWrites(); await Promise.allSettled(inflightColumns()); return this.snapshot(); }
  };

  boot();
})();
