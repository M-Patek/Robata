import { useMemo } from 'react'
import { usePipelineStore } from '@/store'
import { INITIAL_NODES } from '@/data/pipeline'
import { NodeStatus, STATUS_LABEL, STATUS_STYLE } from '@/types'

const STATUS_ORDER: NodeStatus[] = [
  'RUNNING', 'WAITING_REVIEW', 'FAILED', 'PENDING', 'BLOCKED', 'COMPLETE', 'NO_EVENTS',
]

function Divider() {
  return <div style={{ height: 1, background: 'rgba(26,23,20,0.07)', margin: '12px 0' }} />
}

export default function RunSummaryPanel() {
  const activeRun   = usePipelineStore((s) => s.activeRun)
  const reviewTasks = usePipelineStore((s) => s.reviewTasks)

  const counts = useMemo(() => {
    const map = {} as Record<NodeStatus, number>
    for (const s of STATUS_ORDER) map[s] = 0
    for (const n of INITIAL_NODES) {
      const s = activeRun?.node_statuses?.[n.id] ?? n.data.status
      map[s] = (map[s] ?? 0) + 1
    }
    return map
  }, [activeRun])

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid rgba(26,23,20,0.08)' }}>
        <h3 className="text-sm font-semibold"
          style={{ fontFamily: 'Lora, serif', color: '#1A1714' }}>
          Run Summary
        </h3>
        {activeRun ? (
          <p className="text-[10px] font-mono mt-0.5 truncate" style={{ color: '#A89B93' }}>
            {activeRun.run_id}
          </p>
        ) : (
          <p className="text-[11px] mt-0.5" style={{ color: '#A89B93' }}>No active run</p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-0">
        {/* Stages */}
        <p className="label-muted mb-2">Stages</p>
        <div className="space-y-1 mb-1">
          {STATUS_ORDER.map((s) => counts[s] > 0 && (
            <div key={s} className="flex items-center justify-between py-0.5">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: STATUS_STYLE[s].dot }} />
                <span className="text-[11px]" style={{ color: '#3D3530', fontFamily: 'Inter' }}>
                  {STATUS_LABEL[s]}
                </span>
              </div>
              <span className="text-[11px] font-mono tabular-nums" style={{ color: '#6B5E55' }}>
                {counts[s]}
              </span>
            </div>
          ))}
        </div>

        <Divider />

        {/* Evidence */}
        <p className="label-muted mb-2">Evidence</p>
        <div className="rounded-lg px-3 py-1 mb-1"
          style={{ background: '#F8F3E8', border: '1px solid rgba(26,23,20,0.07)' }}>
          <div className="flex justify-between py-1.5"
            style={{ borderBottom: '1px solid rgba(26,23,20,0.06)' }}>
            <span className="text-[10px]" style={{ color: '#A89B93' }}>class</span>
            <span className="text-[10px] font-mono" style={{ color: '#A87A2A' }}>LOCAL_CONFORMANCE</span>
          </div>
          <div className="flex justify-between py-1.5">
            <span className="text-[10px]" style={{ color: '#A89B93' }}>prod eligible</span>
            <span className="text-[10px] font-mono" style={{ color: '#C96442' }}>false</span>
          </div>
        </div>

        <Divider />

        {/* Review queue */}
        <div className="flex items-center justify-between mb-2">
          <p className="label-muted">Review Queue</p>
          {reviewTasks.length > 0 && (
            <span className="status-pill"
              style={{ background: STATUS_STYLE.WAITING_REVIEW.bg, color: STATUS_STYLE.WAITING_REVIEW.text }}>
              {reviewTasks.length}
            </span>
          )}
        </div>
        {reviewTasks.length === 0 ? (
          <p className="text-[11px]" style={{ color: '#A89B93' }}>No pending tasks</p>
        ) : (
          <div className="space-y-1.5">
            {reviewTasks.slice(0, 4).map((t) => (
              <div key={t.task_id}
                className="rounded-lg px-3 py-2 text-[10px]"
                style={{ background: '#F8F3E8', border: '1px solid rgba(168,122,42,0.25)' }}>
                <div className="flex justify-between">
                  <span style={{ color: '#7A5A1A' }}>{t.subject_type}</span>
                  <span className="font-mono" style={{ color: '#A89B93' }}>P{t.priority}</span>
                </div>
                <span className="font-mono text-[9px]" style={{ color: '#A89B93' }}>
                  {t.task_id.slice(0, 14)}…
                </span>
              </div>
            ))}
          </div>
        )}

        <Divider />

        {/* Schema catalog */}
        <p className="label-muted mb-2">Schema Catalog</p>
        <div className="rounded-lg px-3 py-1"
          style={{ background: '#F8F3E8', border: '1px solid rgba(26,23,20,0.07)' }}>
          <div className="flex justify-between py-1.5"
            style={{ borderBottom: '1px solid rgba(26,23,20,0.06)' }}>
            <span className="text-[10px]" style={{ color: '#A89B93' }}>registered</span>
            <span className="text-[10px] font-mono" style={{ color: '#4A7FA8' }}>44</span>
          </div>
          <div className="flex justify-between py-1.5">
            <span className="text-[10px]" style={{ color: '#A89B93' }}>upcasters</span>
            <span className="text-[10px] font-mono" style={{ color: '#A89B93' }}>0 — pending</span>
          </div>
        </div>
      </div>
    </div>
  )
}
