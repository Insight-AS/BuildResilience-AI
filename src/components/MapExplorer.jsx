import L from 'leaflet'
import { useState, useMemo, useRef, useEffect } from 'react'
import { MapContainer, TileLayer, GeoJSON, ScaleControl, ZoomControl, useMapEvents, Polyline, CircleMarker, useMap } from 'react-leaflet'
import { Layers, Ruler, MousePointer2, Download, Maximize2, ListTree, Eye, EyeOff } from 'lucide-react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { MAP_CENTER } from '../data/config'
import { useSample } from './Layout'
import { SampleBadge, StatusBadge } from './ui'

const BASEMAPS = {
  Aerial: { url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr: 'Tiles © Esri, Maxar, Earthstar Geographics' },
  Streets: { url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', attr: '© OpenStreetMap contributors' },
  Dark: { url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', attr: '© OpenStreetMap, © CARTO' },
}
const LAYERS = [
  { id: 'truth', label: 'Ground truth footprints', color: '#C8FD03' },
  { id: 'pred', label: 'Predicted footprints', color: '#7BDFF2' },
  { id: 'wards', label: 'Ward boundaries', color: '#EDB158' },
  { id: 'density', label: 'Building density', color: '#723CD6' },
  { id: 'risk', label: 'Risk / analysis overlay', color: '#E55001' },
]
// SAMPLE ONLY: synthetic squares so the layer UI can be previewed.
function sampleData() {
  const [la, lo] = MAP_CENTER, sq = (a, b, s) => ({ type: 'Polygon', coordinates: [[[b, a], [b + s, a], [b + s, a + s], [b, a + s], [b, a]]] })
  const truth = [], pred = []
  for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { if ((i * 3 + j * 5) % 4 === 0) continue
    const a = la + i * 0.0004, b = lo + j * 0.0004
    truth.push({ type: 'Feature', properties: { source: 'sample' }, geometry: sq(a, b, 0.00022) })
    pred.push({ type: 'Feature', properties: { source: 'sample' }, geometry: sq(a + 0.00002, b + 0.00002, 0.0002) }) }
  const ward = { type: 'Feature', properties: { ward: 'Sample ward' }, geometry: sq(la - 0.0004, lo - 0.0004, 0.0042) }
  return { truth: { type: 'FeatureCollection', features: truth }, pred: { type: 'FeatureCollection', features: pred }, wards: { type: 'FeatureCollection', features: [ward] } }
}
const fc = (d) => (d && d.type === 'FeatureCollection' && d.features?.length ? d : null)

function BuildingPopupText(p) { // only shows properties that exist
  const rows = [['Building ID', p.id ?? p.building_id], ['Footprint area', p.area != null ? `${p.area} m²` : null], ['Ward', p.ward], ['Confidence', p.confidence]].filter(([, v]) => v != null)
  return `<div class="popup"><b>Building</b>${rows.length ? rows.map(([k, v]) => `<div><span>${k}</span>${v}</div>`).join('') : '<div>No properties in this feature.</div>'}</div>`
}
function Measure({ on, pts, setPts }) { useMapEvents({ click: (e) => on && setPts((p) => [...p, [e.latlng.lat, e.latlng.lng]]) }); return null }
function FlyToData({ data }) { const m = useMap(); useEffect(() => { if (data) { try { m.fitBounds(data.getBounds?.() || data, { maxZoom: 18 }) } catch {} } }, [data]); return null }

export default function MapExplorer() {
  const { sample } = useSample()
  const fp = useApi(api.footprints), wd = useApi(api.wards)
  const [base, setBase] = useState('Aerial'), [vis, setVis] = useState({ truth: true, pred: true, wards: true, density: false, risk: false })
  const [panel, setPanel] = useState(true), [measure, setMeasure] = useState(false), [pts, setPts] = useState([])
  const wrap = useRef(null)
  const demo = useMemo(() => (sample ? sampleData() : null), [sample])
  const all = fc(fp.data), features = all?.features || []
  const real = {
    truth: all && { ...all, features: features.filter((f) => f.properties?.source === 'ground_truth') },
    pred: all && { ...all, features: features.filter((f) => f.properties?.source !== 'ground_truth') },
    wards: fc(wd.data),
  }
  const src = { truth: fc(real.truth) || demo?.truth, pred: fc(real.pred) || demo?.pred, wards: real.wards || demo?.wards }
  const usingSample = sample && (!fc(real.truth) && !fc(real.pred))
  const dist = pts.reduce((s, p, i) => (i ? s + L.latLng(pts[i - 1]).distanceTo(L.latLng(p)) : 0), 0)
  const style = (c, fill) => ({ color: c, weight: 1.4, fillColor: c, fillOpacity: fill })
  const onEach = (c) => (f, layer) => {
    layer.bindPopup(BuildingPopupText(f.properties || {}))
    layer.on({ mouseover: () => layer.setStyle({ weight: 3, fillOpacity: 0.7 }), mouseout: () => layer.setStyle(style(c, 0.25)) })
  }
  const status = fp.loading ? 'Loading footprints…' : fp.error ? 'Backend offline: no footprint data' : !all ? 'Backend returned no footprints' : `${features.length} features loaded`
  const fullscreen = () => (document.fullscreenElement ? document.exitFullscreen() : wrap.current?.requestFullscreen())
  return <div className="map-wrap" ref={wrap}>
    <MapContainer center={MAP_CENTER} zoom={17} zoomControl={false} className="map">
      <TileLayer key={base} url={BASEMAPS[base].url} attribution={BASEMAPS[base].attr} maxZoom={19} />
      <ZoomControl position="bottomright" /><ScaleControl position="bottomleft" />
      {vis.wards && src.wards && <GeoJSON key={'w' + !!demo} data={src.wards} style={{ color: '#EDB158', weight: 2, dashArray: '6 4', fillOpacity: 0.04 }} />}
      {vis.truth && src.truth && <GeoJSON key={'t' + !!demo} data={src.truth} style={() => style('#C8FD03', 0.25)} onEachFeature={onEach('#C8FD03')} />}
      {vis.pred && src.pred && <GeoJSON key={'p' + !!demo} data={src.pred} style={() => style('#7BDFF2', 0.25)} onEachFeature={onEach('#7BDFF2')} />}
      <Measure on={measure} pts={pts} setPts={setPts} />
      {pts.length > 0 && <><Polyline positions={pts} pathOptions={{ color: '#E55001', weight: 3 }} />{pts.map((p, i) => <CircleMarker key={i} center={p} radius={4} pathOptions={{ color: '#E55001' }} />)}</>}
    </MapContainer>

    <div className="map-toolbar glass">
      <button className={panel ? 'on' : ''} onClick={() => setPanel(!panel)} title="Toggle layer panel"><Layers size={17} /></button>
      <button disabled title="Select area: needs backend (planned)"><MousePointer2 size={17} /></button>
      <button className={measure ? 'on' : ''} onClick={() => { setMeasure(!measure); setPts([]) }} title="Measure distance: click points on the map"><Ruler size={17} /></button>
      <button disabled title="Download map output: needs backend (planned)"><Download size={17} /></button>
      <button onClick={fullscreen} title="Fullscreen"><Maximize2 size={17} /></button>
    </div>

    {panel && <aside className="map-panel glass">
      <h3><ListTree size={15} /> Layers</h3>
      <div className="seg">{Object.keys(BASEMAPS).map((b) => <button key={b} className={b === base ? 'on' : ''} onClick={() => setBase(b)}>{b}</button>)}</div>
      <ul>{LAYERS.map((l) => { const has = !!src[l.id], pending = ['density', 'risk'].includes(l.id)
        return <li key={l.id}><button onClick={() => setVis({ ...vis, [l.id]: !vis[l.id] })} className={vis[l.id] ? 'on' : ''} aria-pressed={vis[l.id]}>
          {vis[l.id] ? <Eye size={15} /> : <EyeOff size={15} />}<i style={{ background: l.color }} /><span>{l.label}</span>
          {(pending || !has) && <em>{pending ? 'awaiting data' : 'no data'}</em>}</button></li> })}</ul>
      <div className="legend"><b>Legend</b>{LAYERS.slice(0, 3).map((l) => <span key={l.id}><i style={{ background: l.color }} />{l.label}</span>)}</div>
      <div className="map-status"><StatusBadge tone={fp.error ? 'orange' : 'cyan'}>{status}</StatusBadge>{usingSample && <SampleBadge />}
        {measure && <span className="measure">Measured: {dist ? `${dist.toFixed(1)} m` : 'click points on the map'}</span>}</div>
    </aside>}
  </div>
}
