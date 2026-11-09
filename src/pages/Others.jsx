import { useState, useRef, useEffect } from 'react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts'
import { UploadCloud, Play, Download } from 'lucide-react'
import {
  PageHeader,
  ChartPanel,
  GlassPanel,
  DataTable,
  EmptyState,
  ErrorState,
  LoadingState,
  StatusBadge,
  ProjectPipeline,
} from '../components/ui'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'
import { useSample } from '../components/Layout'
import { PIPELINE, SAMPLE } from '../data/config'
import { tip } from './Overview'

const axes = (
  <>
    <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
    <XAxis stroke="#9CA3AF" fontSize={12} />
    <YAxis stroke="#9CA3AF" fontSize={12} />
    <Tooltip {...tip} />
    <Legend />
  </>
)

const stateOf = (r, ok, ready) =>
  r.loading ? (
    <LoadingState />
  ) : r.error ? (
    <ErrorState error={r.error} onRetry={r.reload} />
  ) : !ready ? (
    <EmptyState title="Awaiting model output" hint="No data returned yet." />
  ) : null

export function Density() {
  const { sample } = useSample()
  const d = useApi(api.density)

  const rows =
    Array.isArray(d.data) && d.data.length
      ? d.data
      : sample
        ? SAMPLE.wards
        : []

  const isSample = !d.data?.length && sample
  const st = rows.length ? null : stateOf(d, 0, 0)

  return (
    <>
      <PageHeader
        title="Urban density"
        subtitle="Building count, density and coverage across wards."
      />

      <div className="grid-2">
        <ChartPanel
          title="Density by ward"
          subtitle="Buildings per km²"
          sample={isSample}
          state={st}
        >
          <BarChart data={rows}>
            {axes}
            <XAxis dataKey="ward" stroke="#9CA3AF" />
            <Bar
              dataKey="density"
              name="Density"
              fill="#7BDFF2"
              radius={[6, 6, 0, 0]}
            />
          </BarChart>
        </ChartPanel>

        <ChartPanel
          title="Building coverage"
          subtitle="Footprint area as % of ward"
          sample={isSample}
          state={st}
        >
          <BarChart data={rows}>
            {axes}
            <XAxis dataKey="ward" stroke="#9CA3AF" />
            <Bar
              dataKey="coverage"
              name="Coverage %"
              fill="#C8FD03"
              radius={[6, 6, 0, 0]}
            />
          </BarChart>
        </ChartPanel>
      </div>

      <GlassPanel>
        <h3>Ward table</h3>

        {isSample && (
          <StatusBadge tone="gold">
            Sample data, not project results
          </StatusBadge>
        )}

        <DataTable
          rows={rows}
          empty="No ward data"
          columns={[
            { key: 'ward', label: 'Ward' },
            { key: 'count', label: 'Building count' },
            { key: 'density', label: 'Buildings / km²' },
            { key: 'area', label: 'Footprint area' },
            { key: 'coverage', label: 'Coverage %' },
          ]}
        />
      </GlassPanel>

      <GlassPanel>
        <h3>Planned vs informal settlements</h3>
        <p className="sub">
          Compares density, footprint distribution, count error, IoU and Dice.
        </p>

        <EmptyState
          title="Awaiting settlement classification data"
          hint="Settlement labels are not available yet. This section activates when the backend provides them."
        />
      </GlassPanel>

      <GlassPanel>
        <h3>Choropleth</h3>
        <p className="sub">
          Ward density shading is ready on the Map Explorer's “Building density”
          layer once ward and density GeoJSON are served.
        </p>
      </GlassPanel>
    </>
  )
}

export function Performance() {
  const { sample } = useSample()
  const m = useApi(api.metrics)

  const live = m.data && typeof m.data === 'object'
  const sm = sample && !live

  const comp = live ? m.data.comparison : sm ? SAMPLE.models : null
  const curves = live ? m.data.training : sm ? SAMPLE.epochs : null

  const st = (x) => (x?.length ? null : stateOf(m, 0, 0))

  const names = [
    'IoU',
    'Dice',
    'Precision',
    'Recall',
    'Building-count error',
  ]

  return (
    <>
      <PageHeader
        title="Model performance"
        subtitle="Edge-detection baseline versus CNN segmentation, evaluated on held-out data."
      />

      <div className="grid-5">
        {names.map((n) => (
          <GlassPanel key={n} className="metric">
            <span className="metric-label">{n}</span>
            <strong className="metric-wait">
              {live && m.data.summary?.[n] != null
                ? m.data.summary[n]
                : 'Awaiting model output'}
            </strong>
          </GlassPanel>
        ))}
      </div>

      <div className="grid-2">
        <ChartPanel
          title="Edge baseline vs CNN"
          subtitle="Metric comparison"
          sample={sm}
          state={st(comp)}
        >
          <BarChart data={comp || []}>
            {axes}
            <XAxis dataKey="metric" stroke="#9CA3AF" />
            <Bar
              dataKey="Edge"
              name="Edge baseline"
              fill="#EDB158"
              radius={[6, 6, 0, 0]}
            />
            <Bar
              dataKey="CNN"
              fill="#7BDFF2"
              radius={[6, 6, 0, 0]}
            />
          </BarChart>
        </ChartPanel>

        <ChartPanel
          title="Training loss and validation IoU"
          subtitle="Per epoch"
          sample={sm}
          state={st(curves)}
        >
          <LineChart data={curves || []}>
            {axes}
            <XAxis dataKey="epoch" stroke="#9CA3AF" />
            <Line
              dataKey="loss"
              name="Train loss"
              stroke="#E55001"
              dot={false}
              strokeWidth={2}
            />
            <Line
              dataKey="iou"
              name="Val IoU"
              stroke="#C8FD03"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ChartPanel>
      </div>

      <div className="grid-2">
        <GlassPanel>
          <h3>Experiment comparison</h3>
          <p className="sub">Tier 1 vs Tier 1 + Tier 2 dataset</p>

          <DataTable
            rows={live ? m.data.experiments : []}
            empty="No experiments reported"
            columns={[
              { key: 'name', label: 'Experiment' },
              { key: 'iou', label: 'IoU' },
              { key: 'dice', label: 'Dice' },
              { key: 'precision', label: 'Precision' },
              { key: 'recall', label: 'Recall' },
            ]}
          />
        </GlassPanel>

        <GlassPanel>
          <h3>Settlement comparison</h3>
          <p className="sub">Planned vs informal</p>

          <EmptyState
            title="Awaiting settlement classification data"
            hint="Needs labeled settlement types."
          />
        </GlassPanel>
      </div>
    </>
  )
}

export function Analysis() {
  const [file, setFile] = useState(null)
  const [prev, setPrev] = useState(null)
  const [prog, setProg] = useState(0)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)

  const [opt, setOpt] = useState({
    model: 'cnn',
    type: 'segmentation',
    format: 'geojson',
  })

  const inp = useRef()

  // Release browser-created preview URLs when replaced or component unmounts.
  useEffect(() => {
    return () => {
      if (prev?.startsWith('blob:')) {
        URL.revokeObjectURL(prev)
      }
    }
  }, [prev])

  const pick = (f) => {
    if (!f) return

    setFile(f)
    setRes(null)
    setProg(0)

    const isTiff = /\.(tif|tiff)$/i.test(f.name)

    if (isTiff) {
      setPrev(null)
    } else {
      setPrev(URL.createObjectURL(f))
    }
  }

  const run = async () => {
    if (!file || busy) return

    setBusy(true)
    setRes(null)
    setProg(15)

    const tick = setInterval(() => {
      setProg((p) => Math.min(p + 8, 90))
    }, 300)

    try {
      const r = await api.predict(file, opt)

      setRes(r)
      setProg(r.error ? 0 : 100)
    } catch (error) {
      setRes({
        error: error?.message || 'Analysis failed. Please try again.',
      })
      setProg(0)
    } finally {
      clearInterval(tick)
      setBusy(false)
    }
  }

  const sel = (k, label, items) => (
    <label className="field" key={k}>
      {label}

      <select
        value={opt[k]}
        onChange={(e) =>
          setOpt({
            ...opt,
            [k]: e.target.value,
          })
        }
      >
        {items.map(([v, t]) => (
          <option key={v} value={v}>
            {t}
          </option>
        ))}
      </select>
    </label>
  )

  // Use the local preview for JPG/PNG and backend preview for TIFF.
  const originalSrc = prev || res?.data?.image?.preview

  // The mask preview is generated from the same processed raster dimensions
  // as the backend image preview, allowing it to align as an image overlay.
  const maskSrc = res?.data?.mask?.data_uri

  const hasResults = Boolean(res?.data)
  const hasMask = Boolean(maskSrc)

  return (
    <>
      <PageHeader
        title="AI analysis"
        subtitle="Upload aerial imagery and run building detection."
      />

      <div className="grid-2">
        <GlassPanel>
          <div
            className="drop"
            onClick={() => inp.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              pick(e.dataTransfer.files[0])
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter') inp.current?.click()
            }}
          >
            <input
              ref={inp}
              type="file"
              hidden
              accept=".png,.jpg,.jpeg,.tif,.tiff"
              onChange={(e) => pick(e.target.files[0])}
            />

            {prev ? (
              <img src={prev} alt="Selected aerial preview" />
            ) : (
              <>
                <UploadCloud size={30} />

                <strong>
                  {file ? file.name : 'Drop an aerial image or browse'}
                </strong>

                <span>
                  PNG, JPG, JPEG and GeoTIFF supported by the backend.
                </span>
              </>
            )}
          </div>

          {(busy || prog > 0) && (
            <div className="bar">
              <i style={{ width: `${prog}%` }} />
            </div>
          )}

          <div className="grid-form">
            {sel('model', 'Model', [
              ['cnn', 'CNN segmentation'],
              ['edge', 'Edge detection baseline'],
            ])}

            {sel('type', 'Analysis type', [
              ['segmentation', 'Semantic segmentation'],
              ['footprints', 'Building footprint extraction'],
              ['density', 'Density analysis'],
            ])}

            {sel('format', 'Output format', [
              ['geojson', 'GeoJSON'],
              ['png', 'PNG mask'],
              ['csv', 'CSV summary'],
            ])}
          </div>

          <button
            className="btn btn-lime"
            disabled={!file || busy}
            onClick={run}
          >
            <Play size={15} />
            {busy ? 'Analyzing' : 'Run analysis'}
          </button>
        </GlassPanel>

        <GlassPanel>
          <h3>Results</h3>

          {!res && (
            <EmptyState
              title="No results yet"
              hint="Upload an image and run the analysis."
            />
          )}

          {res?.error && <ErrorState error={res.error} onRetry={run} />}

          {res?.data && (
            <>
              <p className="sub">Backend returned a result.</p>

              <pre className="json">
                {JSON.stringify(res.data, null, 2).slice(0, 1200)}
              </pre>

              <button
                className="btn"
                disabled
                title="Export wiring depends on the response shape"
              >
                <Download size={14} />
                Export
              </button>
            </>
          )}
        </GlassPanel>
      </div>

      <div className="grid-4">
        {/* Original image */}
        <GlassPanel>
          <h3>Original image</h3>

          {hasResults && originalSrc ? (
            <img
              src={originalSrc}
              alt="Original aerial image"
              style={{
                width: '100%',
                borderRadius: 12,
                objectFit: 'contain',
                display: 'block',
              }}
            />
          ) : (
            <EmptyState
              title="Pending"
              hint="Appears after a successful analysis."
            />
          )}
        </GlassPanel>

        {/* Segmentation mask */}
        <GlassPanel>
          <h3>Segmentation mask</h3>

          {hasMask ? (
            <img
              src={maskSrc}
              alt="Segmentation mask"
              style={{
                width: '100%',
                borderRadius: 12,
                objectFit: 'contain',
                display: 'block',
              }}
            />
          ) : (
            <EmptyState
              title="Pending"
              hint="Appears after a successful analysis."
            />
          )}
        </GlassPanel>

        {/* Footprint overlay */}
        <GlassPanel>
          <h3>Footprint overlay</h3>

          {hasResults && originalSrc && hasMask ? (
            <div
              style={{
                position: 'relative',
                width: '100%',
                overflow: 'hidden',
                borderRadius: 12,
              }}
            >
              <img
                src={originalSrc}
                alt="Original aerial image"
                style={{
                  width: '100%',
                  display: 'block',
                  objectFit: 'contain',
                }}
              />

              <img
                src={maskSrc}
                alt="Segmentation mask overlay"
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'fill',
                  opacity: 0.45,
                  mixBlendMode: 'screen',
                  pointerEvents: 'none',
                }}
              />
            </div>
          ) : (
            <EmptyState
              title="Pending"
              hint="Appears after a successful analysis."
            />
          )}

          {hasMask && (
            <p className="sub" style={{ marginTop: 8 }}>
              Overlay visualizes the predicted segmentation mask.
            </p>
          )}
        </GlassPanel>

        {/* Building statistics */}
        <GlassPanel>
          <h3>Building statistics</h3>

          {res?.data?.statistics ? (
            <div className="sub">
              <p>
                <strong>Building count: </strong>
                {res.data.statistics.building_count ?? 'N/A'}
              </p>

              <p>
                <strong>Total area (m²): </strong>
                {res.data.statistics.total_area_m2 ?? 'N/A'}
              </p>

              <p>
                <strong>Mean area (m²): </strong>
                {res.data.statistics.mean_area_m2 ?? 'N/A'}
              </p>

              <p>
                <strong>Median area (m²): </strong>
                {res.data.statistics.median_area_m2 ?? 'N/A'}
              </p>

              <p>
                <strong>Building density (per km²): </strong>
                {res.data.statistics.building_density_per_km2 ?? 'N/A'}
              </p>

              <p>
                <strong>Building coverage (%): </strong>
                {res.data.statistics.building_coverage_pct ?? 'N/A'}
              </p>
            </div>
          ) : (
            <EmptyState
              title="Pending"
              hint="Appears after a successful analysis."
            />
          )}
        </GlassPanel>
      </div>
    </>
  )
}

const T = ({ h, children }) => (
  <GlassPanel className="prose">
    <h3>{h}</h3>
    {children}
  </GlassPanel>
)

export function Project() {
  return (
    <>
      <PageHeader
        title="Project information"
        subtitle="Building footprint mapping for disaster resilience."
      />

      <div className="grid-2">
        <T h="Overview">
          <p>
            An AI system that detects building footprints in aerial imagery,
            derives ward-level building density, and presents the results on an
            interactive web map to support disaster resilience and urban risk
            planning.
          </p>
        </T>

        <T h="Objectives">
          <ul>
            <li>Segment buildings in aerial imagery.</li>
            <li>Generate footprint polygons.</li>
            <li>Compute building density per ward.</li>
            <li>Compare planned and informal settlements.</li>
          </ul>
        </T>

        <T h="Dataset">
          <p>
            Open Cities AI Challenge: aerial imagery with building-footprint
            annotations. Dataset statistics will be added from the project's
            own analysis.
          </p>
        </T>

        <T h="Methodology">
          <p>
            A mandatory edge-detection baseline is compared with a CNN-based
            segmentation model. Model architecture details will be added once
            finalized.
          </p>
        </T>

        <T h="Evaluation">
          <p>
            IoU, Dice, precision, recall and building-count error, reported
            overall and by settlement type.
          </p>
        </T>

        <GlassPanel>
          <h3>Pipeline</h3>
          <ProjectPipeline stages={PIPELINE} />
        </GlassPanel>
      </div>
    </>
  )
}

export function SettingsPage() {
  return (
    <>
      <PageHeader
        title="Settings"
        subtitle="Connection and display options."
      />

      <GlassPanel className="prose">
        <h3>API connection</h3>

        <p>
          Base URL:{' '}
          <code>
            {import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'}
          </code>
        </p>

        <p>
          Change it in <code>.env</code> (
          <code>VITE_API_BASE_URL</code>) and restart the dev server. Endpoint
          paths live in <code>src/services/api.js</code>.
        </p>

        <p>
          Sample preview is toggled from the top bar. Sample data is always
          badged.
        </p>
      </GlassPanel>
    </>
  )
}