import { Loader2, AlertTriangle, Inbox, FlaskConical } from 'lucide-react'
import { ResponsiveContainer } from 'recharts'
export const GlassPanel = ({ className = '', children, ...p }) => <section className={`glass ${className}`} {...p}>{children}</section>
export const StatusBadge = ({ tone = 'muted', children }) => <span className={`badge badge-${tone}`}>{children}</span>
export const SampleBadge = () => <StatusBadge tone="gold"><FlaskConical size={12} /> Sample data, not project results</StatusBadge>
export function PageHeader({ title, subtitle, children }) {
  return <header className="page-header"><div><h1>{title}</h1><p>{subtitle}</p></div><div className="row">{children}</div></header>
}
export const LoadingState = ({ label = 'Loading' }) => <div className="state"><Loader2 className="spin" size={20} /><span>{label}…</span></div>
export const EmptyState = ({ title, hint }) => <div className="state"><Inbox size={20} /><strong>{title}</strong><span>{hint}</span></div>
export const ErrorState = ({ error, onRetry }) => <div className="state state-err"><AlertTriangle size={20} /><strong>{error?.kind === 'offline' ? 'Backend unavailable' : 'Request failed'}</strong><span>{error?.message}</span>{onRetry && <button className="btn" onClick={onRetry}>Retry</button>}</div>
export function MetricCard({ label, value, hint, tone = 'cyan', sample, delay = 0 }) {
  const waiting = value == null
  return <GlassPanel className={`metric tone-${tone}`} style={{ animationDelay: `${delay}ms` }}>
    <span className="metric-label">{label}</span>
    <strong className={waiting ? 'metric-wait' : 'metric-val'}>{waiting ? 'Awaiting model output' : value}</strong>
    <span className="metric-hint">{hint}</span>{sample && !waiting && <SampleBadge />}</GlassPanel>
}
export function ChartPanel({ title, subtitle, sample, children, height = 260, state }) {
  return <GlassPanel className="chart"><div className="chart-head"><div><h3>{title}</h3><p>{subtitle}</p></div>{sample && <SampleBadge />}</div>
    <div style={{ height }}>{state || <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>}</div></GlassPanel>
}
export function DataTable({ columns, rows, empty }) {
  if (!rows?.length) return <EmptyState title={empty || 'No rows yet'} hint="Rows appear once the backend returns data." />
  return <div className="table-wrap"><table><thead><tr>{columns.map((c) => <th key={c.key}>{c.label}</th>)}</tr></thead>
    <tbody>{rows.map((r, i) => <tr key={i}>{columns.map((c) => <td key={c.key}>{r[c.key] ?? '—'}</td>)}</tr>)}</tbody></table></div>
}
export function ProjectPipeline({ stages }) {
  return <ol className="pipeline">{stages.map((s) => <li key={s.name} className={`st-${s.status}`}><i /><span>{s.name}</span><em>{s.status === 'done' ? 'Completed' : s.status === 'active' ? 'In progress' : 'Pending'}</em></li>)}</ol>
}
