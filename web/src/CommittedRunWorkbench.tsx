import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Copy,
  Database,
  FileKey2,
  Layers3,
  ShieldCheck,
  X,
} from 'lucide-react'
import type {
  CameraQuality,
  Evidence,
  NanosecondInterval,
  PipelineStage,
  RunHypothesis,
  RunPackage,
  RunPublication,
  RunSnapshot,
  RunSnapshotResponse,
  RunWindow,
} from '@/api/runs'
import './committed-run-workbench.css'

type PlaneMode = 'both' | 'evidence' | 'decision'
type Tone = 'success' | 'warning' | 'danger' | 'info' | 'neutral'

interface Fact {
  label: string
  value: string
  code?: boolean
}

interface DrawerSelection {
  category: string
  title: string
  tone: Tone
  facts: Fact[]
}

interface TimelineBounds {
  start: bigint
  end: bigint
}

interface TimelineItem {
  id: string
  label: string
  interval: NanosecondInterval
  tone: Tone
  selection: DrawerSelection
}

interface WorkbenchProps {
  snapshot: RunSnapshotResponse
}

const NANOSECONDS_PER_SECOND = 1_000_000_000n

export default function CommittedRunWorkbench({ snapshot }: WorkbenchProps) {
  const [planeMode, setPlaneMode] = useState<PlaneMode>('both')
  const [selection, setSelection] = useState<DrawerSelection | null>(null)
  const closeDrawer = useCallback(() => setSelection(null), [])
  const run = snapshot.run
  const bounds = useMemo(() => timelineBounds(run), [run])
  const cameraGroups = useMemo(() => groupCameraQuality(run.camera_quality), [run.camera_quality])

  const packageItems: TimelineItem[] = run.packages.map((item) => ({
    id: item.package_id,
    label: `Package ${item.ordinal + 1}`,
    interval: item.interval,
    tone: 'info',
    selection: packageSelection(item),
  }))
  const hypothesisItems: TimelineItem[] = run.hypotheses.map((item) => ({
    id: item.logical_key,
    label: `Hypothesis ${item.ordinal + 1}`,
    interval: item.effective_interval,
    tone: 'warning',
    selection: hypothesisSelection(item),
  }))
  const publicationItems: TimelineItem[] = run.publications.map((item) => ({
    id: item.revision_id,
    label: 'Published event',
    interval: item.effective_interval,
    tone: 'success',
    selection: publicationSelection(item),
  }))
  const rootItems: TimelineItem[] = run.window
    ? [{
        id: run.window.logical_key,
        label: run.window.purpose,
        interval: run.window.effective_interval,
        tone: 'neutral',
        selection: windowSelection(run.window),
      }]
    : []

  return (
    <div className="committed-workbench">
      <section className="workbench-context" aria-label="Committed run context">
        <div className="workbench-context-title">
          <span className="eyebrow">Committed run</span>
          <h1 title={run.recording_identity}>{abbreviate(run.recording_identity, 34)}</h1>
          <code title={run.run_id}>{abbreviate(run.run_id, 18)}</code>
        </div>
        <div className="workbench-context-metrics">
          <ToneBadge label={run.status} tone={statusTone(run.status)} />
          <ToneBadge
            label={run.production_eligible ? 'Production eligible' : run.evidence_class}
            tone={run.production_eligible ? 'success' : 'warning'}
            icon={run.production_eligible ? <CheckCircle2 size={13} /> : <ShieldCheck size={13} />}
          />
          <Metric label="Duration" value={formatNanoseconds(run.window?.recording_duration_ns ?? '0')} />
          <Metric label="Events" value={String(run.event_count)} />
          <Metric label="Evidence" value={String(run.evidence.length)} />
        </div>
      </section>

      <StageRail stages={run.stages} onSelect={(stage) => setSelection(stageSelection(stage))} />

      <section className="timeline-section" aria-label="Committed interval timeline">
        <div className="section-header">
          <div>
            <span className="eyebrow">Committed timeline</span>
            <h2>Recording intervals</h2>
          </div>
          <button
            className="integrity-trigger"
            type="button"
            onClick={() => setSelection(integritySelection(run, snapshot.cursor))}
          >
            <FileKey2 size={14} aria-hidden="true" />
            <span>Snapshot identity</span>
          </button>
        </div>
        <div className="timeline-scroll">
          <div className="timeline-canvas">
            <TimelineTicks bounds={bounds} />
            <TimelineLane label="Root window" items={rootItems} bounds={bounds} onSelect={setSelection} />
            <TimelineLane label="Packages" items={packageItems} bounds={bounds} onSelect={setSelection} />
            {cameraGroups.map(([cameraId, items]) => (
              <TimelineLane
                key={cameraId}
                label={cameraId}
                items={items.map((item) => ({
                  id: `${cameraId}:${item.interval.start_ns}:${item.interval.end_ns}`,
                  label: item.status,
                  interval: item.interval,
                  tone: statusTone(item.status),
                  selection: cameraSelection(item),
                }))}
                bounds={bounds}
                onSelect={setSelection}
              />
            ))}
            <TimelineLane label="Hypotheses" items={hypothesisItems} bounds={bounds} onSelect={setSelection} />
            <TimelineLane label="Publications" items={publicationItems} bounds={bounds} onSelect={setSelection} />
          </div>
        </div>
      </section>

      <div className="plane-switch" role="tablist" aria-label="Workspace planes">
        <PlaneButton mode="both" active={planeMode === 'both'} onSelect={setPlaneMode} />
        <PlaneButton mode="evidence" active={planeMode === 'evidence'} onSelect={setPlaneMode} />
        <PlaneButton mode="decision" active={planeMode === 'decision'} onSelect={setPlaneMode} />
      </div>

      <div className={`workbench-planes ${planeMode}`}>
        {planeMode !== 'decision' && (
          <EvidencePlane
            run={run}
            onSelect={setSelection}
          />
        )}
        {planeMode !== 'evidence' && (
          <DecisionPlane
            run={run}
            onSelect={setSelection}
          />
        )}
      </div>

      {selection && <DetailDrawer selection={selection} onClose={closeDrawer} />}
    </div>
  )
}

function StageRail({ stages, onSelect }: { stages: PipelineStage[]; onSelect: (stage: PipelineStage) => void }) {
  const completed = stages.filter((stage) => stage.state === 'COMPLETE').length
  return (
    <section className="stage-rail" aria-label="Completion stages">
      <div className="stage-rail-heading">
        <span className="eyebrow">Completion path</span>
        <span>{completed}/{stages.length}</span>
      </div>
      <div className="stage-track">
        {stages.map((stage, index) => (
          <div className="stage-sequence" key={stage.name}>
            <button
              className={`stage-node ${stageTone(stage)}`}
              type="button"
              onClick={() => onSelect(stage)}
              title={`${stage.name}: ${stage.state}`}
            >
              <span className="stage-node-dot" aria-hidden="true" />
              <span>{stage.name}</span>
            </button>
            {index < stages.length - 1 && <ChevronRight className="stage-arrow" size={14} aria-hidden="true" />}
          </div>
        ))}
      </div>
    </section>
  )
}

function TimelineTicks({ bounds }: { bounds: TimelineBounds }) {
  const span = bounds.end - bounds.start
  return (
    <div className="timeline-ticks" aria-hidden="true">
      <span />
      <div className="timeline-tick-track">
        {[0n, 1n, 2n, 3n, 4n].map((step) => (
          <span key={step.toString()} style={{ left: `${Number(step * 25n)}%` }}>
            {formatNanoseconds((bounds.start + (span * step) / 4n).toString())}
          </span>
        ))}
      </div>
    </div>
  )
}

function TimelineLane({
  label,
  items,
  bounds,
  onSelect,
}: {
  label: string
  items: TimelineItem[]
  bounds: TimelineBounds
  onSelect: (selection: DrawerSelection) => void
}) {
  const positionedItems = stackTimelineItems(items)
  const rowCount = Math.max(1, ...positionedItems.map((item) => item.row + 1))

  return (
    <div className="timeline-lane">
      <div className="timeline-lane-label" title={label}>
        <span>{label}</span>
        <small>{items.length}</small>
      </div>
      <div className="timeline-lane-track" style={{ height: `${rowCount * 21}px` }}>
        {items.length === 0 && <span className="timeline-empty" />}
        {positionedItems.map(({ item, row }) => {
          const metrics = intervalMetrics(item.interval, bounds)
          return (
            <button
              className={`timeline-bar ${item.tone}`}
              key={item.id}
              type="button"
              style={{ left: `${metrics.left}%`, top: `${3 + row * 21}px`, width: `${metrics.width}%` }}
              title={`${item.label}: ${formatInterval(item.interval)}`}
              onClick={() => onSelect(item.selection)}
            >
              <span>{item.label}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function EvidencePlane({ run, onSelect }: { run: RunSnapshot; onSelect: (selection: DrawerSelection) => void }) {
  return (
    <section className="workspace-plane evidence-plane" aria-label="Source and QA plane">
      <PlaneHeading icon={<Database size={16} />} title="Source & QA" count={run.camera_quality.length} />
      {run.window ? (
        <button className="source-window-row" type="button" onClick={() => onSelect(windowSelection(run.window!))}>
          <div>
            <span>Root window</span>
            <strong>{run.window.purpose}</strong>
          </div>
          <code>{formatInterval(run.window.effective_interval)}</code>
          <ChevronRight size={15} aria-hidden="true" />
        </button>
      ) : (
        <EmptyPlaneRow label="No root window committed" />
      )}
      <ObjectGroup
        label="Packages"
        count={run.packages.length}
        empty="No packages committed"
        items={run.packages.map((item) => ({
          key: item.package_id,
          title: `Package ${item.ordinal + 1}`,
          meta: `${item.part_count} parts - ${formatInterval(item.interval)}`,
          tone: 'info' as Tone,
          selection: packageSelection(item),
        }))}
        onSelect={onSelect}
      />
      <ObjectGroup
        label="Camera quality"
        count={run.camera_quality.length}
        empty="No camera quality committed"
        items={run.camera_quality.map((item) => ({
          key: `${item.camera_id}:${item.interval.start_ns}:${item.interval.end_ns}`,
          title: item.camera_id,
          meta: `${item.status} - ${formatInterval(item.interval)}`,
          tone: statusTone(item.status),
          selection: cameraSelection(item),
        }))}
        onSelect={onSelect}
      />
      <ObjectGroup
        label="Evidence references"
        count={run.evidence.length}
        empty="No evidence references committed"
        items={run.evidence.map((item) => ({
          key: `${item.role}:${item.exact_bytes_sha256}`,
          title: item.role,
          meta: `${item.schema_id} v${item.schema_version}`,
          tone: 'neutral' as Tone,
          selection: evidenceSelection(item),
        }))}
        onSelect={onSelect}
      />
    </section>
  )
}

function DecisionPlane({ run, onSelect }: { run: RunSnapshot; onSelect: (selection: DrawerSelection) => void }) {
  return (
    <section className="workspace-plane decision-plane" aria-label="Decision and delivery plane">
      <PlaneHeading icon={<Layers3 size={16} />} title="Decision & delivery" count={run.publications.length} />
      <button className="decision-row" type="button" onClick={() => onSelect(decisionSelection(run))}>
        <div className={`decision-symbol ${run.decision ? statusTone(run.decision.decision) : 'neutral'}`}>
          {run.decision ? <CheckCircle2 size={17} aria-hidden="true" /> : <AlertTriangle size={17} aria-hidden="true" />}
        </div>
        <div>
          <span>Output decision</span>
          <strong>{run.decision?.decision ?? 'Not recorded'}</strong>
        </div>
        <ChevronRight size={15} aria-hidden="true" />
      </button>
      <ObjectGroup
        label="Hypotheses"
        count={run.hypotheses.length}
        empty="No hypotheses committed"
        items={run.hypotheses.map((item) => ({
          key: item.logical_key,
          title: `Hypothesis ${item.ordinal + 1}`,
          meta: formatInterval(item.effective_interval),
          tone: 'warning' as Tone,
          selection: hypothesisSelection(item),
        }))}
        onSelect={onSelect}
      />
      <ObjectGroup
        label="Publications"
        count={run.publications.length}
        empty="No action publications committed"
        items={run.publications.map((item) => ({
          key: item.revision_id,
          title: abbreviate(item.event_id, 15),
          meta: formatInterval(item.effective_interval),
          tone: 'success' as Tone,
          selection: publicationSelection(item),
        }))}
        onSelect={onSelect}
      />
      <button className="integrity-row" type="button" onClick={() => onSelect(integritySelection(run, 'Current snapshot'))}>
        <FileKey2 size={15} aria-hidden="true" />
        <span>Completion integrity</span>
        <code>{abbreviate(run.integrity.completion_semantic_sha256, 10)}</code>
        <ChevronRight size={15} aria-hidden="true" />
      </button>
    </section>
  )
}

function PlaneHeading({ icon, title, count }: { icon: ReactNode; title: string; count: number }) {
  return (
    <div className="plane-heading">
      <div>
        {icon}
        <h2>{title}</h2>
      </div>
      <span>{count}</span>
    </div>
  )
}

function ObjectGroup({
  label,
  count,
  empty,
  items,
  onSelect,
}: {
  label: string
  count: number
  empty: string
  items: Array<{ key: string; title: string; meta: string; tone: Tone; selection: DrawerSelection }>
  onSelect: (selection: DrawerSelection) => void
}) {
  return (
    <div className="object-group">
      <div className="object-group-heading">
        <span>{label}</span>
        <small>{count}</small>
      </div>
      {items.length === 0 ? (
        <EmptyPlaneRow label={empty} />
      ) : (
        <div className="object-list">
          {items.map((item) => (
            <button className="object-row" key={item.key} type="button" onClick={() => onSelect(item.selection)}>
              <span className={`object-row-marker ${item.tone}`} aria-hidden="true" />
              <span className="object-row-copy">
                <strong title={item.title}>{item.title}</strong>
                <small title={item.meta}>{item.meta}</small>
              </span>
              <ChevronRight size={14} aria-hidden="true" />
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function EmptyPlaneRow({ label }: { label: string }) {
  return <div className="plane-empty">{label}</div>
}

function PlaneButton({ mode, active, onSelect }: { mode: PlaneMode; active: boolean; onSelect: (mode: PlaneMode) => void }) {
  const label = mode === 'both' ? 'Both' : mode === 'evidence' ? 'Source & QA' : 'Decision & delivery'
  return (
    <button
      className={active ? 'active' : undefined}
      type="button"
      role="tab"
      aria-selected={active}
      onClick={() => onSelect(mode)}
    >
      {label}
    </button>
  )
}

function ToneBadge({ label, tone, icon }: { label: string; tone: Tone; icon?: ReactNode }) {
  return (
    <span className={`tone-badge ${tone}`}>
      {icon ?? <span className="tone-dot" aria-hidden="true" />}
      <span>{label}</span>
    </span>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="context-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function DetailDrawer({ selection, onClose }: { selection: DrawerSelection; onClose: () => void }) {
  const drawerRef = useRef<HTMLElement>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const openerRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    const activeElement = document.activeElement
    openerRef.current = activeElement instanceof HTMLElement ? activeElement : null
    closeButtonRef.current?.focus()

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onClose()
        return
      }
      if (event.key !== 'Tab') return

      const focusable = Array.from(
        drawerRef.current?.querySelectorAll<HTMLButtonElement>('button:not([disabled])') ?? [],
      )
      if (focusable.length === 0) {
        event.preventDefault()
        return
      }

      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      openerRef.current?.focus()
    }
  }, [onClose])

  return (
    <>
      <div className="detail-backdrop" aria-hidden="true" onClick={onClose} />
      <aside ref={drawerRef} className="detail-drawer" role="dialog" aria-modal="true" aria-labelledby="detail-drawer-title">
        <header className="detail-drawer-header">
          <div>
            <span className="eyebrow">{selection.category}</span>
            <h2 id="detail-drawer-title">{selection.title}</h2>
          </div>
          <button ref={closeButtonRef} className="drawer-icon-button" type="button" aria-label="Close detail drawer" title="Close" onClick={onClose}>
            <X size={17} aria-hidden="true" />
          </button>
        </header>
        <div className="detail-facts">
          {selection.facts.map((fact) => (
            <div className="detail-fact" key={fact.label}>
              <span>{fact.label}</span>
              <div>
                {fact.code ? <code title={fact.value}>{fact.value}</code> : <strong title={fact.value}>{fact.value}</strong>}
                <button
                  className="copy-button"
                  type="button"
                  aria-label={`Copy ${fact.label}`}
                  title={`Copy ${fact.label}`}
                  onClick={() => copyValue(fact.value)}
                >
                  <Copy size={14} aria-hidden="true" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>
    </>
  )
}

function timelineBounds(run: RunSnapshot): TimelineBounds {
  const intervals: NanosecondInterval[] = []
  if (run.window) {
    intervals.push(run.window.requested_interval, run.window.effective_interval)
  }
  intervals.push(
    ...run.packages.map((item) => item.interval),
    ...run.camera_quality.map((item) => item.interval),
    ...run.hypotheses.map((item) => item.effective_interval),
    ...run.publications.map((item) => item.effective_interval),
  )

  let start: bigint | null = null
  let end: bigint | null = null
  for (const interval of intervals) {
    const intervalStart = asBigInt(interval.start_ns)
    const intervalEnd = asBigInt(interval.end_ns)
    start = start === null || intervalStart < start ? intervalStart : start
    end = end === null || intervalEnd > end ? intervalEnd : end
  }
  if (run.window) {
    const recordingEnd = asBigInt(run.window.recording_duration_ns)
    if (recordingEnd > 0n) {
      start = start === null || 0n < start ? 0n : start
      end = end === null || recordingEnd > end ? recordingEnd : end
    }
  }
  if (start === null || end === null || end <= start) {
    return { start: 0n, end: NANOSECONDS_PER_SECOND }
  }
  return { start, end }
}

function intervalMetrics(interval: NanosecondInterval, bounds: TimelineBounds): { left: number; width: number } {
  const span = bounds.end - bounds.start
  if (span <= 0n) {
    return { left: 0, width: 100 }
  }
  const start = clamp(asBigInt(interval.start_ns), bounds.start, bounds.end)
  const end = clamp(asBigInt(interval.end_ns), bounds.start, bounds.end)
  const left = Number(((start - bounds.start) * 100_000n) / span) / 1_000
  const rawWidth = Number(((end - start) * 100_000n) / span) / 1_000
  return { left, width: Math.max(0.8, rawWidth) }
}

function stackTimelineItems(items: TimelineItem[]): Array<{ item: TimelineItem; row: number }> {
  const rowEnds: bigint[] = []
  return [...items]
    .sort((left, right) => {
      const startDelta = asBigInt(left.interval.start_ns) - asBigInt(right.interval.start_ns)
      if (startDelta !== 0n) return startDelta < 0n ? -1 : 1
      const endDelta = asBigInt(left.interval.end_ns) - asBigInt(right.interval.end_ns)
      if (endDelta !== 0n) return endDelta < 0n ? -1 : 1
      return left.id.localeCompare(right.id)
    })
    .map((item) => {
      const intervalStart = asBigInt(item.interval.start_ns)
      const intervalEnd = asBigInt(item.interval.end_ns)
      const row = rowEnds.findIndex((rowEnd) => intervalStart >= rowEnd)
      if (row === -1) {
        rowEnds.push(intervalEnd)
        return { item, row: rowEnds.length - 1 }
      }
      rowEnds[row] = intervalEnd
      return { item, row }
    })
}

function groupCameraQuality(items: CameraQuality[]): Array<[string, CameraQuality[]]> {
  const grouped = new Map<string, CameraQuality[]>()
  for (const item of items) {
    const current = grouped.get(item.camera_id) ?? []
    current.push(item)
    grouped.set(item.camera_id, current)
  }
  return [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right))
}

function statusTone(value: string): Tone {
  const normalized = value.toLowerCase()
  if (/(fail|reject|invalid|quarantine|deny|error|not_run)/.test(normalized)) return 'danger'
  if (/(pending|review|wait|partial|abstain|local_conformance)/.test(normalized)) return 'warning'
  if (/(complete|success|admit|eligible|published|allow|pass|good)/.test(normalized)) return 'success'
  return 'info'
}

function stageTone(stage: PipelineStage): Tone {
  return stage.state === 'COMPLETE' ? 'success' : 'neutral'
}

function windowSelection(item: RunWindow): DrawerSelection {
  return {
    category: 'Root window',
    title: item.purpose,
    tone: 'info',
    facts: [
      { label: 'Logical key', value: item.logical_key, code: true },
      { label: 'Purpose', value: item.purpose },
      { label: 'Requested interval', value: formatInterval(item.requested_interval), code: true },
      { label: 'Effective interval', value: formatInterval(item.effective_interval), code: true },
      { label: 'Recording duration', value: formatNanoseconds(item.recording_duration_ns), code: true },
    ],
  }
}

function packageSelection(item: RunPackage): DrawerSelection {
  return {
    category: 'Package',
    title: `Package ${item.ordinal + 1}`,
    tone: 'info',
    facts: [
      { label: 'Package ID', value: item.package_id, code: true },
      { label: 'Ordinal', value: String(item.ordinal) },
      { label: 'Part count', value: String(item.part_count) },
      { label: 'Interval', value: formatInterval(item.interval), code: true },
    ],
  }
}

function cameraSelection(item: CameraQuality): DrawerSelection {
  return {
    category: 'Camera quality',
    title: item.camera_id,
    tone: statusTone(item.status),
    facts: [
      { label: 'Camera ID', value: item.camera_id, code: true },
      { label: 'Status', value: item.status },
      { label: 'Interval', value: formatInterval(item.interval), code: true },
    ],
  }
}

function stageSelection(item: PipelineStage): DrawerSelection {
  return {
    category: 'Completion stage',
    title: item.name,
    tone: stageTone(item),
    facts: [
      { label: 'State', value: item.state },
      { label: 'Semantic SHA-256', value: item.semantic_sha256 ?? 'Not recorded', code: true },
    ],
  }
}

function decisionSelection(run: RunSnapshot): DrawerSelection {
  return {
    category: 'Output decision',
    title: run.decision?.decision ?? 'Not recorded',
    tone: run.decision ? statusTone(run.decision.decision) : 'neutral',
    facts: run.decision
      ? [
          { label: 'Decision', value: run.decision.decision },
          { label: 'Reason code', value: run.decision.reason_code ?? 'Not recorded', code: true },
          { label: 'Admitted claims', value: String(run.decision.admitted_claim_count) },
        ]
      : [{ label: 'Decision', value: 'Not recorded' }],
  }
}

function hypothesisSelection(item: RunHypothesis): DrawerSelection {
  return {
    category: 'Hypothesis',
    title: `Hypothesis ${item.ordinal + 1}`,
    tone: 'warning',
    facts: [
      { label: 'Ordinal', value: String(item.ordinal) },
      { label: 'Logical key', value: item.logical_key, code: true },
      { label: 'Semantic SHA-256', value: item.semantic_sha256, code: true },
      { label: 'Effective interval', value: formatInterval(item.effective_interval), code: true },
    ],
  }
}

function publicationSelection(item: RunPublication): DrawerSelection {
  return {
    category: 'Action publication',
    title: abbreviate(item.event_id, 22),
    tone: 'success',
    facts: [
      { label: 'Event ID', value: item.event_id, code: true },
      { label: 'Revision ID', value: item.revision_id, code: true },
      { label: 'Effective interval', value: formatInterval(item.effective_interval), code: true },
    ],
  }
}

function evidenceSelection(item: Evidence): DrawerSelection {
  return {
    category: 'Evidence reference',
    title: item.role,
    tone: 'neutral',
    facts: [
      { label: 'Role', value: item.role },
      { label: 'Schema ID', value: item.schema_id, code: true },
      { label: 'Schema version', value: item.schema_version, code: true },
      { label: 'Semantic SHA-256', value: item.semantic_sha256, code: true },
      { label: 'Exact bytes SHA-256', value: item.exact_bytes_sha256, code: true },
      { label: 'Byte count', value: item.byte_count.toLocaleString() },
    ],
  }
}

function integritySelection(run: RunSnapshot, cursor: string): DrawerSelection {
  return {
    category: 'Completion integrity',
    title: 'Committed snapshot',
    tone: 'neutral',
    facts: [
      { label: 'Snapshot cursor', value: cursor, code: true },
      { label: 'Command SHA-256', value: run.integrity.command_sha256, code: true },
      { label: 'Completion semantic SHA-256', value: run.integrity.completion_semantic_sha256, code: true },
    ],
  }
}

function formatNanoseconds(value: string): string {
  const nanoseconds = asBigInt(value)
  const sign = nanoseconds < 0n ? '-' : ''
  const absolute = nanoseconds < 0n ? -nanoseconds : nanoseconds
  const seconds = absolute / NANOSECONDS_PER_SECOND
  const remainder = absolute % NANOSECONDS_PER_SECOND
  if (remainder === 0n) return `${sign}${seconds}s`
  return `${sign}${seconds}.${remainder.toString().padStart(9, '0').replace(/0+$/, '')}s`
}

function formatInterval(interval: NanosecondInterval): string {
  return `${formatNanoseconds(interval.start_ns)} to ${formatNanoseconds(interval.end_ns)}`
}

function abbreviate(value: string, length: number): string {
  return value.length <= length * 2 + 3 ? value : `${value.slice(0, length)}...${value.slice(-length)}`
}

function asBigInt(value: string): bigint {
  try {
    return BigInt(value)
  } catch {
    return 0n
  }
}

function clamp(value: bigint, minimum: bigint, maximum: bigint): bigint {
  return value < minimum ? minimum : value > maximum ? maximum : value
}

function copyValue(value: string): void {
  if (navigator.clipboard) {
    void navigator.clipboard.writeText(value)
  }
}
