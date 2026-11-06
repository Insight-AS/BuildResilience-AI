import { useState, createContext, useContext } from 'react'
import { NavLink, Outlet, useLocation, Link } from 'react-router-dom'
import * as Icons from 'lucide-react'
import { Search, Bell, Menu, PanelLeftClose, PanelLeftOpen, FlaskConical } from 'lucide-react'
import { NAV } from '../data/config'
export const SampleCtx = createContext({ sample: false, setSample: () => {} })
export const useSample = () => useContext(SampleCtx)
export default function AppLayout() {
  const [collapsed, setCollapsed] = useState(false), [open, setOpen] = useState(false), [sample, setSample] = useState(false)
  const loc = useLocation(); const cur = NAV.find((n) => n[0] === loc.pathname)?.[1] || 'Overview'
  return <SampleCtx.Provider value={{ sample, setSample }}>
    <div className={`shell ${collapsed ? 'collapsed' : ''} ${open ? 'nav-open' : ''}`}>
      <aside className="sidebar">
        <div className="brand"><div className="logo"><Icons.Hexagon size={22} /></div><b>BuildResilience<span> AI</span></b></div>
        <nav>{NAV.map(([to, label, ic]) => { const I = Icons[ic]; return <NavLink key={to} to={to} end={to === '/'} onClick={() => setOpen(false)}><I size={18} /><span>{label}</span></NavLink> })}</nav>
        <button className="collapse" onClick={() => setCollapsed(!collapsed)} aria-label="Collapse sidebar">{collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}<span>Collapse</span></button>
      </aside>
      {open && <div className="scrim" onClick={() => setOpen(false)} />}
      <div className="main">
        <header className="topbar">
          <button className="icon-btn mobile-only" onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={18} /></button>
          <div className="crumbs"><Link to="/">BuildResilience</Link><span>/</span><strong>{cur}</strong></div>
          <label className="search"><Search size={15} /><input placeholder="Search wards, buildings (needs backend)" disabled title="Search activates once the backend is connected" /></label>
          <button className={`btn ${sample ? 'btn-gold' : ''}`} onClick={() => setSample(!sample)} title="Show clearly labeled sample data for layout preview"><FlaskConical size={14} /> Sample preview: {sample ? 'on' : 'off'}</button>
          <button className="icon-btn" disabled title="No notifications yet"><Bell size={17} /></button>
          <div className="avatar" title="Project placeholder">BR</div>
        </header>
        <main key={loc.pathname} className="page"><Outlet /></main>
      </div>
    </div></SampleCtx.Provider>
}
