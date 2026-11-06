import { Link } from 'react-router-dom'
import { Radar } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, AreaChart, Area } from 'recharts'
import { PageHeader, MetricCard, ChartPanel, GlassPanel, StatusBadge, ProjectPipeline, LoadingState, ErrorState, EmptyState } from '../components/ui'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'
import { useSample } from '../components/Layout'
import { PIPELINE, SAMPLE } from '../data/config'
export const tip = { contentStyle: { background: '#172033', border: '1px solid rgba(255,255,255,.1)', borderRadius: 10, color: '#FCFFFF' } }
export default function Overview() {
  const { sample } = useSample()
  const health = useApi(api.health), sum = useApi(api.summary), dens = useApi(api.density)
  const s = sum.data || {}
  const wards = Array.isArray(dens.data) && dens.data.length ? dens.data : sample ? SAMPLE.wards : null
  const isSample = !Array.isArray(dens.data) && sample
  const chartState = dens.loading ? <LoadingState /> : !wards ? (dens.error ? <ErrorState error={dens.error} onRetry={dens.reload} /> : <EmptyState title="No ward data" hint="Awaiting model output." />) : null
  return <>
    <PageHeader title="Geospatial Intelligence" subtitle="Building footprint mapping and urban density analysis for disaster resilience.">
      <StatusBadge tone={health.data ? 'lime' : 'orange'}>{health.loading ? 'Checking backend' : health.data ? 'Backend online' : 'Backend offline'}</StatusBadge>
      <StatusBadge tone="muted">Open Cities AI Challenge dataset</StatusBadge>
      <Link className="btn btn-lime" to="/analysis"><Radar size={15} /> Analyze area</Link>
    </PageHeader>
    <div className="grid-4">
      <MetricCard label="Total buildings detected" value={s.total_buildings} hint="From model inference" delay={0} />
      <MetricCard label="Total footprint area" value={s.total_footprint_area} hint="Sum of polygon areas" tone="lime" delay={70} />
      <MetricCard label="Average building density" value={s.avg_density} hint="Buildings per km²" tone="gold" delay={140} />
      <MetricCard label="Wards analyzed" value={s.wards_analyzed} hint="Ward-level aggregation" tone="purple" delay={210} />
    </div>
    <div className="grid-2">
      <ChartPanel title="Ward-wise building density" subtitle="Buildings per km²" sample={isSample} state={chartState}>
        <BarChart data={wards || []}><CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} /><XAxis dataKey="ward" stroke="#9CA3AF" fontSize={12} /><YAxis stroke="#9CA3AF" fontSize={12} /><Tooltip {...tip} cursor={{ fill: 'rgba(255,255,255,.04)' }} /><Bar dataKey="density" fill="#7BDFF2" radius={[6, 6, 0, 0]} /></BarChart>
      </ChartPanel>
      <ChartPanel title="Footprint area distribution" subtitle="Buildings by footprint size" sample={isSample} state={chartState}>
        <AreaChart data={wards || []}><CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} /><XAxis dataKey="ward" stroke="#9CA3AF" fontSize={12} /><YAxis stroke="#9CA3AF" fontSize={12} /><Tooltip {...tip} /><Area dataKey="coverage" stroke="#C8FD03" fill="rgba(200,253,3,.15)" /></AreaChart>
      </ChartPanel>
    </div>
    <div className="grid-3">
      <GlassPanel><h3>Project pipeline</h3><p className="sub">Edit stage status in <code>src/data/config.js</code>.</p><ProjectPipeline stages={PIPELINE} /></GlassPanel>
      <GlassPanel><h3>Model status</h3>{sum.data?.model ? <p>{sum.data.model}</p> : <EmptyState title="No trained model connected" hint="Model status appears once the backend reports it." />}</GlassPanel>
      <GlassPanel><h3>Recent analysis activity</h3><EmptyState title="No analyses yet" hint="Run one from AI Analysis." /></GlassPanel>
    </div>
  </>
}
