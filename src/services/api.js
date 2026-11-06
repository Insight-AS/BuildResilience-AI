// Central API layer. Change endpoints here only.
const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
export const ENDPOINTS = {
  health: '/health', summary: '/summary', footprints: '/footprints', wards: '/wards',
  density: '/density', metrics: '/metrics', predict: '/predict', analysis: (id) => `/analysis/${id}`,
}
async function request(path, opts = {}) {
  const ctrl = new AbortController(); const t = setTimeout(() => ctrl.abort(), 8000)
  try {
    const res = await fetch(BASE + path, { ...opts, signal: ctrl.signal })
    if (!res.ok) return { data: null, error: { kind: 'api', message: `Server returned ${res.status}` } }
    return { data: await res.json(), error: null }
  } catch (e) {
    return { data: null, error: { kind: 'offline', message: `Cannot reach ${BASE}. Start the FastAPI backend or check VITE_API_BASE_URL.` } }
  } finally { clearTimeout(t) }
}
export const api = {
  health: () => request(ENDPOINTS.health),
  summary: () => request(ENDPOINTS.summary),
  footprints: () => request(ENDPOINTS.footprints),
  wards: () => request(ENDPOINTS.wards),
  density: () => request(ENDPOINTS.density),
  metrics: () => request(ENDPOINTS.metrics),
  analysis: (id) => request(ENDPOINTS.analysis(id)),
  predict: (file, options) => { const f = new FormData(); f.append('file', file)
    Object.entries(options).forEach(([k, v]) => f.append(k, v)); return request(ENDPOINTS.predict, { method: 'POST', body: f }) },
}
