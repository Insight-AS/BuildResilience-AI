import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import 'leaflet/dist/leaflet.css'
import './styles.css'
import AppLayout from './components/Layout'
import Overview from './pages/Overview'
import MapPage from './pages/MapPage'
import { Density, Performance, Analysis, Project, SettingsPage } from './pages/Others'
createRoot(document.getElementById('root')).render(<BrowserRouter><Routes><Route element={<AppLayout />}>
  <Route index element={<Overview />} /><Route path="map" element={<MapPage />} /><Route path="density" element={<Density />} />
  <Route path="performance" element={<Performance />} /><Route path="analysis" element={<Analysis />} />
  <Route path="project" element={<Project />} /><Route path="settings" element={<SettingsPage />} /></Route></Routes></BrowserRouter>)
