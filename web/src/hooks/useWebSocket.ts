import { useEffect, useRef, useCallback } from 'react'
import { usePipelineStore } from '@/store'
import { StreamEvent } from '@/types'
import { MOCK_EVENT_STREAM } from '@/data/mock_stream_events'

// ── Simulates a live stream by replaying mock events with configurable speed ──

interface SimulationConfig {
  speed: number // 1.0 = real-time, 2.0 = 2x, etc.
  onEvent: (event: StreamEvent) => void
  onComplete: () => void
}

function runSimulation(config: SimulationConfig): () => void {
  const { speed, onEvent, onComplete } = config
  const events = [...MOCK_EVENT_STREAM]
  let idx = 0
  let timeouts: ReturnType<typeof setTimeout>[] = []

  // Schedule events with small delays to simulate streaming
  // In a real implementation, events would have timestamps and we'd pace accordingly
  const scheduleNext = () => {
    if (idx >= events.length) {
      onComplete()
      return
    }

    const event = events[idx]
    onEvent(event)
    idx++

    // Delay between events — faster for demo
    const delay = Math.max(50, 200 / speed)
    const timeout = setTimeout(scheduleNext, delay)
    timeouts.push(timeout)
  }

  // Start immediately
  const initialTimeout = setTimeout(scheduleNext, 500)
  timeouts.push(initialTimeout)

  // Cleanup function
  return () => {
    timeouts.forEach(clearTimeout)
    timeouts = []
  }
}

export function useWebSocket() {
  const setWsConnected = usePipelineStore((s) => s.setWsConnected)
  const setSimulating = usePipelineStore((s) => s.setSimulating)
  const simulationSpeed = usePipelineStore((s) => s.streamView.simulation_speed)
  const ingestEvent = usePipelineStore((s) => s.ingestEvent)
  const resetStreamView = usePipelineStore((s) => s.resetStreamView)

  const cleanupRef = useRef<(() => void) | null>(null)

  const startSimulation = useCallback(() => {
    // Reset any previous state
    resetStreamView()
    setSimulating(true)

    cleanupRef.current = runSimulation({
      speed: simulationSpeed,
      onEvent: (event) => {
        ingestEvent(event)
      },
      onComplete: () => {
        setSimulating(false)
      },
    })
  }, [ingestEvent, resetStreamView, setSimulating, simulationSpeed])

  const stopSimulation = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current()
      cleanupRef.current = null
    }
    setSimulating(false)
  }, [setSimulating])

  useEffect(() => {
    // Try to connect to real WebSocket (not implemented yet)
    let ws: WebSocket | null = null
    try {
      ws = new WebSocket('ws://localhost:8000/ws/pipeline')
      ws.onopen = () => setWsConnected(true)
      ws.onclose = () => setWsConnected(false)
      ws.onerror = () => {
        setWsConnected(false)
        ws?.close()
      }
    } catch {
      setWsConnected(false)
    }

    return () => {
      ws?.close()
      stopSimulation()
    }
  }, [setWsConnected, stopSimulation])

  return { startSimulation, stopSimulation }
}
