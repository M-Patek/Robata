import { useMemo } from 'react'
import { usePipelineStore } from '@/store'
import { INITIAL_NODES } from '@/data/pipeline'
import { NodeStatus } from '@/types'
import { clsx } from 'clsx'

const STATUS_ORDER: NodeStatus[] = [
  'RUNNING', 'WAITING_REVIEW', 'FAILED', 'PENDING', 'BLOCKED', 'COMPLETE', 'NO_EVENTS',
]

const STATUS_LABEL: Record<NodeStatus, string> = {
  RUNNING:        '⚡ Running',
  WAITING_REVIEW: '👁 Review',
  FAILED:         '✖ Failed',
  PENDING:        '◌ Pending',
  BLOCKED:        '⛔ Blocked',
  COMPLETE:       '✔ Complete',
  NO_EVENTS:      '— No Events',
}

const STATUS_COLOR: Record<NodeStatus, string> = {
  RUNNING:        'text-blue-400',
  WAITING_REVIEW: 'text-yellow-400',
  FAILED:         'text-red-400',
  PENDING:        'text-gray-400',
  BLOCKED:        'text-purple-400',
  COMPLETE:       'text-green-400',
  NO_EVENTS:      'text-slate-400',
}

export default function RunSummaryPanel() {
  const activeRun  = usePipelineStore((s) => s.activeRun)
  const reviewTasks = usePipelineStore((s) => s.reviewTasks)

  const counts = useMemo(() => {
    const all = INITIAL_NODES.map((n) => {
      return activeRun?.node_statuses?.[n.id] ?? n.data.status
    })
    const map = {} as Record<NodeStatus, number>
    for (const s of STATUS_ORDER) map[s] = 0
    for (const s of all) map[s] = (map[s] ?? 0) + 1
    return map
  }, [activeRun])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-canvas-border">
        <h3 className="text-sm font-semibold text-white">Run Summary</h3>
        {activeRun ? (
          <p className="text-[10px] text-gray-500 font-mono mt-0.5 truncate">
            {activeRun.run_id}
          </p>
        ) : (
          <p className="text-[10px] text-gray-500 mt-0.5">No active run</p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {/* Stage counts */}
        <div>
          <p className="text-[9px] uppercase tracking-widest text-gray-500 mb-2">Stages</p>
          <div className="space-y-1">
            {STATUS_ORDER.map((s) => (
              counts[s] > 0 && (
                <div key={s} className="flex items-center justify-between">
                  <span className={clsx('text-[11px]', STATUS_COLOR[s])}>
                    {STATUS_LABEL[s]}
                  </span>
                  <span className={clsx('font-mono text-[11px]', STATUS_COLOR[s])}>
                    {counts[s]}
                  </span>
                </div>
              )
            ))}
          </div>
        </div>

        {/* Evidence class */}
        <div>
          <p className="text-[9px] uppercase tracking-widest text-gray-500 mb-2">Evidence</p>
          <div className="rounded-md bg-canvas-bg border border-canvas-border px-3 py-2 space-y-1">
            <div className="flex justify-between">
              <span className="text-[10px] text-gray-500">class</span>
              <span className="text-[10px] font-mono text-amber-300">LOCAL_CONFORMANCE</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[10px] text-gray-500">prod eligible</span>
              <span className="text-[10px] font-mono text-red-400">false</span>
            </div>
          </div>
        </div>

        {/* Review queue */}
        <div>
          <p className="text-[9px] uppercase tracking-widest text-gray-500 mb-2">
            Review Queue
            {reviewTasks.length > 0 && (
              <span className="ml-2 px-1.5 py-0.5 rounded-full bg-yellow-900 text-yellow-300 text-[9px]">
                {reviewTasks.length}
              </span>
            )}
          </p>
          {reviewTasks.length === 0 ? (
            <p className="text-[10px] text-gray-600 italic">No pending tasks</p>
          ) : (
            <div className="space-y-1">
              {reviewTasks.slice(0, 5).map((t) => (
                <div key={t.task_id}
                  className="rounded bg-canvas-bg border border-yellow-900/40 px-2 py-1.5 text-[10px]">
                  <div className="flex justify-between">
                    <span className="text-yellow-300 truncate">{t.subject_type}</span>
                    <span className="text-gray-500 font-mono ml-2">P{t.priority}</span>
                  </div>
                  <span className="text-gray-500 font-mono text-[9px]">
                    {t.task_id.slice(0, 12)}…
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Schema catalog */}
        <div>
          <p className="text-[9px] uppercase tracking-widest text-gray-500 mb-2">Schema Catalog</p>
          <div className="rounded-md bg-canvas-bg border border-canvas-border px-3 py-2 space-y-1">
            <div className="flex justify-between">
              <span className="text-[10px] text-gray-500">registered</span>
              <span className="text-[10px] font-mono text-cyan-300">44</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[10px] text-gray-500">upcasters</span>
              <span className="text-[10px] font-mono text-gray-500">0 (pending)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
