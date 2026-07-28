import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import {
  AlertTriangle,
  CircleSlash2,
  FileCheck2,
  LoaderCircle,
  RefreshCw,
  ServerCrash,
  Wifi,
  WifiOff,
} from 'lucide-react'
import CommittedRunWorkbench from '@/CommittedRunWorkbench'
import {
  fetchRunSnapshot,
  fetchRuns,
  type RunSnapshotResponse,
  type RunSummary,
} from '@/api/runs'
import { useWebSocket, type RunSocketState } from '@/hooks/useWebSocket'
import './run-viewer.css'

type LoadState = 'idle' | 'loading' | 'ready' | 'error'

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function socketLabel(socket: RunSocketState): string {
  switch (socket.status) {
    case 'committed':
      return 'Committed updates'
    case 'connecting':
      return 'Connecting'
    case 'reconnecting':
      return `Reconnecting${socket.retryCount > 0 ? ` (${socket.retryCount})` : ''}`
    case 'disconnected':
      return 'Updates unavailable'
    default:
      return 'No run selected'
  }
}

function runOptionLabel(run: RunSummary): string {
  const identity = run.recording_identity.length > 42
    ? `${run.recording_identity.slice(0, 39)}...`
    : run.recording_identity
  return `${identity} - ${run.status}`
}

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [snapshot, setSnapshot] = useState<RunSnapshotResponse | null>(null)
  const [runsState, setRunsState] = useState<LoadState>('loading')
  const [snapshotState, setSnapshotState] = useState<LoadState>('idle')
  const [runsError, setRunsError] = useState<string | null>(null)
  const [snapshotError, setSnapshotError] = useState<string | null>(null)
  const [refreshVersion, setRefreshVersion] = useState(0)
  const selectedRunRef = useRef<string | null>(null)

  useEffect(() => {
    selectedRunRef.current = selectedRunId
  }, [selectedRunId])

  useEffect(() => {
    const controller = new AbortController()
    setRunsState('loading')
    setRunsError(null)

    void fetchRuns(controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return
        setRuns(response.runs)
        setSelectedRunId((current) => {
          if (current && response.runs.some((run) => run.run_id === current)) return current
          return response.runs.length === 1 ? response.runs[0].run_id : null
        })
        setRunsState('ready')
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || isAbortError(error)) return
        setRunsState('error')
        setRunsError(error instanceof Error ? error.message : 'The run list could not be loaded.')
      })

    return () => controller.abort()
  }, [refreshVersion])

  useEffect(() => {
    if (!selectedRunId) {
      setSnapshot(null)
      setSnapshotError(null)
      setSnapshotState('idle')
      return undefined
    }

    const controller = new AbortController()
    setSnapshot(null)
    setSnapshotState('loading')
    setSnapshotError(null)

    void fetchRunSnapshot(selectedRunId, controller.signal)
      .then((response) => {
        if (controller.signal.aborted || response.run.run_id !== selectedRunId) return
        setSnapshot(response)
        setSnapshotState('ready')
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted || isAbortError(error)) return
        setSnapshotState('error')
        setSnapshotError(error instanceof Error ? error.message : 'The selected run could not be loaded.')
      })

    return () => controller.abort()
  }, [selectedRunId, refreshVersion])

  const applyCommittedSnapshot = useCallback((nextSnapshot: RunSnapshotResponse) => {
    if (nextSnapshot.run.run_id !== selectedRunRef.current) return
    setSnapshot(nextSnapshot)
    setSnapshotState('ready')
    setSnapshotError(null)
  }, [])

  const socket = useWebSocket({ runId: selectedRunId, onSnapshot: applyCommittedSnapshot })
  const refresh = useCallback(() => setRefreshVersion((version) => version + 1), [])

  return (
    <div className="run-app">
      <header className="run-header">
        <div className="run-branding">
          <span className="run-brand">Robata</span>
          <span className="run-brand-divider" aria-hidden="true" />
          <label className="run-picker">
            <span className="sr-only">Committed run</span>
            <select
              value={selectedRunId ?? ''}
              disabled={runsState !== 'ready' || runs.length === 0}
              onChange={(event) => setSelectedRunId(event.target.value || null)}
            >
              <option value="">Select committed run</option>
              {runs.map((run) => <option key={run.run_id} value={run.run_id}>{runOptionLabel(run)}</option>)}
            </select>
          </label>
        </div>
        <div className="run-header-actions">
          <ConnectionIndicator socket={socket} />
          <button
            className="icon-button"
            type="button"
            onClick={refresh}
            disabled={runsState === 'loading'}
            aria-label="Refresh committed runs"
            title="Refresh committed runs"
          >
            <RefreshCw size={16} className={runsState === 'loading' ? 'spin' : undefined} aria-hidden="true" />
          </button>
        </div>
      </header>

      <main className="run-shell">
        {runsState === 'loading' && <ViewerState icon={<LoaderCircle className="spin" size={25} />} title="Loading committed runs" />}
        {runsState === 'error' && (
          <ViewerState
            icon={<ServerCrash size={25} />}
            title="Run service unavailable"
            detail={runsError || 'The run list could not be loaded.'}
            actionLabel="Retry"
            onAction={refresh}
          />
        )}
        {runsState === 'ready' && runs.length === 0 && <ViewerState icon={<CircleSlash2 size={25} />} title="No committed runs" />}
        {runsState === 'ready' && runs.length > 0 && !selectedRunId && <ViewerState icon={<FileCheck2 size={25} />} title="Select a committed run" />}
        {selectedRunId && snapshotState === 'loading' && <ViewerState icon={<LoaderCircle className="spin" size={25} />} title="Loading committed snapshot" />}
        {selectedRunId && snapshotState === 'error' && (
          <ViewerState
            icon={<AlertTriangle size={25} />}
            title="Snapshot unavailable"
            detail={snapshotError || 'The selected run could not be loaded.'}
            actionLabel="Retry"
            onAction={refresh}
          />
        )}
        {snapshot && snapshotState === 'ready' && <CommittedRunWorkbench snapshot={snapshot} />}
      </main>
    </div>
  )
}

function ConnectionIndicator({ socket }: { socket: RunSocketState }) {
  const connected = socket.status === 'committed'
  const connecting = socket.status === 'connecting' || socket.status === 'reconnecting'
  const Icon = connected ? Wifi : connecting ? LoaderCircle : WifiOff
  return (
    <div className={`connection-indicator ${connected ? 'committed' : connecting ? 'pending' : 'offline'}`} title={socket.detail || socketLabel(socket)}>
      <Icon size={14} className={connecting ? 'spin' : undefined} aria-hidden="true" />
      <span>{socketLabel(socket)}</span>
    </div>
  )
}

function ViewerState({
  icon,
  title,
  detail,
  actionLabel,
  onAction,
}: {
  icon: ReactNode
  title: string
  detail?: string
  actionLabel?: string
  onAction?: () => void
}) {
  return (
    <div className="viewer-state">
      <div className="viewer-state-icon">{icon}</div>
      <h1>{title}</h1>
      {detail && <p>{detail}</p>}
      {actionLabel && onAction && (
        <button className="state-action" type="button" onClick={onAction}>
          <RefreshCw size={15} aria-hidden="true" />
          {actionLabel}
        </button>
      )}
    </div>
  )
}
