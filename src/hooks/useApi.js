import { useEffect, useState, useCallback } from 'react'
export function useApi(fn) {
  const [s, set] = useState({ loading: true, data: null, error: null })
  const run = useCallback(async () => { set((p) => ({ ...p, loading: true })); const r = await fn(); set({ loading: false, ...r }) }, [fn])
  useEffect(() => { run() }, [run])
  return { ...s, reload: run }
}
