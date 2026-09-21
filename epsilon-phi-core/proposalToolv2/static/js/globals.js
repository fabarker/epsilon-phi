'use strict';

/* Cyrus pages call the Flask reverse proxy; it rewrites /api/* to /api/v1/*. */
window.API_BASE = window.location.origin + '/api';
window.PROPOSAL_TOOL_V2_BASE = '/proposalToolv2/';
