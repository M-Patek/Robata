import { useEffect, useRef, useState } from 'react'
import {
  ApiProtocolError,
  type RunSnapshotResponse,
  parseRunSnapshotMessage,
  runWebSocketUrl,
} from '@/api/runs'

export type RunSocketStatus = 'idle' | 'connecting' | 'committed' | 'reconnecting' | 'disconnected'

export interface RunSocketState {
  status: RunSocketStatus
  detail: string | null
  retryCount: number
}

interface UseWebSocketOptions {
  runId: string | null
  onSnapshot: (snapshot: RunSnapshotResponse) => void
}

const INITIAL_STATE: RunSocketState = {
  status: 'idle',
  detail: null,
  retryCount: 0,
}

const MAX_RETRY_DELAY_MS = 15_000

function retryDelay(retryCount: number): number {
  return Math.min(1_000 * 2 ** Math.min(retryCount, 4), MAX_RETRY_DELAY_MS)
}

/**
 * Subscribes to committed snapshots for the selected run. The REST snapshot is
 * fetched separately by the view so the viewer remains usable when streaming
 * transport is temporarily unavailable.
 */
export function useWebSocket({ runId, onSnapshot }: UseWebSocketOptions): RunSocketState {
  const [state, setState] = useState<RunSocketState>(INITIAL_STATE)
  const snapshotHandlerRef = useRef(onSnapshot)

  useEffect(() => {
    snapshotHandlerRef.current = onSnapshot
  }, [onSnapshot])

  useEffect(() => {
    if (!runId) {
      setState(INITIAL_STATE)
      return undefined
    }

    let disposed = false
    let socket: WebSocket | null = null
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let retryCount = 0
    let closeDetail: string | null = null

    const scheduleReconnect = (detail: string | null) => {
      if (disposed) {
        return
      }

      const delay = retryDelay(retryCount)
      retryCount += 1
      setState({ status: 'reconnecting', detail, retryCount })
      retryTimer = setTimeout(connect, delay)
    }

    const connect = () => {
      if (disposed) {
        return
      }

      setState({
        status: retryCount === 0 ? 'connecting' : 'reconnecting',
        detail: null,
        retryCount,
      })

      closeDetail = null
      try {
        socket = new WebSocket(runWebSocketUrl(runId))
      } catch {
        scheduleReconnect('The committed update connection could not be opened.')
        return
      }

      socket.onopen = () => {
        if (!disposed) {
          retryCount = 0
          setState({ status: 'connecting', detail: null, retryCount })
        }
      }

      socket.onmessage = (event) => {
        if (disposed || typeof event.data !== 'string') {
          return
        }

        try {
          const message = parseRunSnapshotMessage(JSON.parse(event.data) as unknown)
          if (message.snapshot.run.run_id !== runId) {
            throw new ApiProtocolError('The committed update belongs to a different run.')
          }
          snapshotHandlerRef.current(message.snapshot)
          setState({ status: 'committed', detail: null, retryCount })
        } catch (error) {
          const detail = error instanceof Error ? error.message : 'The committed update could not be read.'
          closeDetail = detail
          setState({ status: 'disconnected', detail, retryCount })
          socket?.close()
        }
      }

      socket.onerror = () => {
        if (!disposed) {
          closeDetail = 'The committed update connection encountered an error.'
          setState({ status: 'disconnected', detail: closeDetail, retryCount })
        }
      }

      socket.onclose = () => {
        if (!disposed) {
          scheduleReconnect(closeDetail ?? 'The committed update connection is unavailable.')
        }
      }
    }

    connect()

    return () => {
      disposed = true
      if (retryTimer) {
        clearTimeout(retryTimer)
      }
      socket?.close()
    }
  }, [runId])

  return state
}
