// Project state: edit as the project progresses. 'done' | 'active' | 'pending'
export const PIPELINE = [
  ['Dataset Preparation', 'active'], ['Image Preprocessing', 'pending'], ['Model Training', 'pending'],
  ['Building Segmentation', 'pending'], ['Polygon Generation', 'pending'], ['Density Analysis', 'pending'], ['Web Map Visualization', 'active'],
].map(([name, status]) => ({ name, status }))
export const NAV = [
  ['/', 'Overview', 'LayoutDashboard'], ['/map', 'Map Explorer', 'Map'], ['/density', 'Density Analysis', 'Grid3x3'],
  ['/performance', 'Model Performance', 'Gauge'], ['/analysis', 'AI Analysis', 'ScanSearch'],
  ['/project', 'Project Information', 'BookOpen'], ['/settings', 'Settings', 'Settings'],
]
// SAMPLE ONLY: synthetic shapes for layout preview. Not project results.
export const SAMPLE = {
  wards: ['Ward A', 'Ward B', 'Ward C', 'Ward D', 'Ward E', 'Ward F'].map((w, i) => ({ ward: w, density: [40, 75, 55, 90, 30, 65][i], coverage: [12, 28, 20, 34, 9, 24][i] })),
  epochs: Array.from({ length: 12 }, (_, i) => ({ epoch: i + 1, loss: +(1 / (1 + i * 0.5)).toFixed(3), iou: +(0.2 + 0.4 * (1 - 1 / (1 + i * 0.4))).toFixed(3) })),
  models: ['IoU', 'Dice', 'Precision', 'Recall'].map((m, i) => ({ metric: m, Edge: [0.2, 0.3, 0.35, 0.3][i], CNN: [0.5, 0.6, 0.65, 0.6][i] })),
}
export const MAP_CENTER = [-6.8235, 39.2695] // placeholder view
