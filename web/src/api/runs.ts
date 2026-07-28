export interface NanosecondInterval {
  start_ns: string
  end_ns: string
}

export interface RunSummary {
  run_id: string
  recording_identity: string
  status: string
  started_at: string | null
  completed_at: string | null
  pipeline_version: string
  output_decision: string | null
  event_count: number
}

export interface RunWindow {
  logical_key: string
  purpose: string
  requested_interval: NanosecondInterval
  effective_interval: NanosecondInterval
  recording_duration_ns: string
}

export interface RunPackage {
  package_id: string
  ordinal: number
  part_count: number
  interval: NanosecondInterval
}

export interface CameraQuality {
  camera_id: string
  status: string
  interval: NanosecondInterval
}

export interface PipelineStage {
  name: string
  state: 'COMPLETE' | 'NOT_RUN'
  semantic_sha256: string | null
}

export interface RunDecision {
  decision: string
  reason_code: string | null
  admitted_claim_count: number
}

export interface RunHypothesis {
  ordinal: number
  logical_key: string
  semantic_sha256: string
  effective_interval: NanosecondInterval
}

export interface RunPublication {
  event_id: string
  revision_id: string
  effective_interval: NanosecondInterval
}

export interface Evidence {
  role: string
  schema_id: string
  schema_version: string
  semantic_sha256: string
  exact_bytes_sha256: string
  byte_count: number
}

export interface RunIntegrity {
  command_sha256: string
  completion_semantic_sha256: string
}

export interface RunSnapshot extends RunSummary {
  evidence_class: string
  production_eligible: boolean
  window: RunWindow | null
  packages: RunPackage[]
  camera_quality: CameraQuality[]
  stages: PipelineStage[]
  decision: RunDecision | null
  hypotheses: RunHypothesis[]
  publications: RunPublication[]
  integrity: RunIntegrity
  evidence: Evidence[]
}

export interface RunsResponse {
  api_version: 'v1'
  runs: RunSummary[]
}

export interface RunSnapshotResponse {
  api_version: 'v1'
  cursor: string
  run: RunSnapshot
}

export interface RunSnapshotMessage {
  type: 'snapshot'
  snapshot: RunSnapshotResponse
}

export class ApiRequestError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiRequestError'
    this.status = status
  }
}

export class ApiProtocolError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ApiProtocolError'
  }
}

const API_VERSION = 'v1'

function apiBase(): string {
  const configured = import.meta.env.VITE_ROBATA_API_BASE?.trim()
  return (configured || '/api/v1').replace(/\/+$/, '')
}

function apiUrl(path: string): string {
  return `${apiBase()}/${path.replace(/^\/+/, '')}`
}

async function requestJson(url: string, signal?: AbortSignal): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(url, {
      headers: { Accept: 'application/json' },
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error
    }
    throw new ApiRequestError('The API is unavailable. Check the API base URL and server status.')
  }

  const body = await response.text()
  let payload: unknown
  try {
    payload = body ? JSON.parse(body) : null
  } catch {
    throw new ApiProtocolError('The API returned an invalid JSON response.')
  }

  if (!response.ok) {
    const detail = errorDetail(payload)
    throw new ApiRequestError(detail || `The API request failed with status ${response.status}.`, response.status)
  }

  return payload
}

export async function fetchRuns(signal?: AbortSignal): Promise<RunsResponse> {
  return parseRunsResponse(await requestJson(apiUrl('runs'), signal))
}

export async function fetchRunSnapshot(runId: string, signal?: AbortSignal): Promise<RunSnapshotResponse> {
  return parseRunSnapshotResponse(
    await requestJson(apiUrl(`runs/${encodeURIComponent(runId)}/snapshot`), signal),
  )
}

export function runWebSocketUrl(runId: string): string {
  const configured = import.meta.env.VITE_ROBATA_WS_BASE?.trim() || '/ws/v1'
  const endpoint = new URL(configured, window.location.origin)
  if (endpoint.protocol === 'http:' || endpoint.protocol === 'https:') {
    endpoint.protocol = endpoint.protocol === 'https:' ? 'wss:' : 'ws:'
  }
  endpoint.pathname = `${endpoint.pathname.replace(/\/+$/, '')}/runs/${encodeURIComponent(runId)}`
  return endpoint.toString()
}

export function parseRunSnapshotMessage(value: unknown): RunSnapshotMessage {
  if (!isRecord(value) || value.type !== 'snapshot') {
    throw new ApiProtocolError('The WebSocket delivered an unsupported message.')
  }

  return { type: 'snapshot', snapshot: parseRunSnapshotResponse(value.snapshot) }
}

function parseRunsResponse(value: unknown): RunsResponse {
  if (!isRecord(value) || value.api_version !== API_VERSION || !Array.isArray(value.runs)) {
    throw new ApiProtocolError('The API returned an invalid run list response.')
  }

  if (!value.runs.every(isRunSummary)) {
    throw new ApiProtocolError('The API returned a malformed run summary.')
  }

  return { api_version: API_VERSION, runs: value.runs }
}

function parseRunSnapshotResponse(value: unknown): RunSnapshotResponse {
  if (!isRecord(value) || value.api_version !== API_VERSION || typeof value.cursor !== 'string' || !isRunSnapshot(value.run)) {
    throw new ApiProtocolError('The API returned an invalid run snapshot response.')
  }

  return { api_version: API_VERSION, cursor: value.cursor, run: value.run }
}

function errorDetail(value: unknown): string | null {
  if (!isRecord(value)) {
    return null
  }
  return typeof value.detail === 'string' ? value.detail : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isStringOrNull(value: unknown): value is string | null {
  return typeof value === 'string' || value === null
}

function isRunSummary(value: unknown): value is RunSummary {
  return isRecord(value)
    && typeof value.run_id === 'string'
    && typeof value.recording_identity === 'string'
    && typeof value.status === 'string'
    && isStringOrNull(value.started_at)
    && isStringOrNull(value.completed_at)
    && typeof value.pipeline_version === 'string'
    && isStringOrNull(value.output_decision)
    && typeof value.event_count === 'number'
}

function isInterval(value: unknown): value is NanosecondInterval {
  return isRecord(value) && typeof value.start_ns === 'string' && typeof value.end_ns === 'string'
}

function isRunWindow(value: unknown): value is RunWindow {
  return isRecord(value)
    && typeof value.logical_key === 'string'
    && typeof value.purpose === 'string'
    && isInterval(value.requested_interval)
    && isInterval(value.effective_interval)
    && typeof value.recording_duration_ns === 'string'
}

function isRunPackage(value: unknown): value is RunPackage {
  return isRecord(value)
    && typeof value.package_id === 'string'
    && typeof value.ordinal === 'number'
    && typeof value.part_count === 'number'
    && isInterval(value.interval)
}

function isCameraQuality(value: unknown): value is CameraQuality {
  return isRecord(value)
    && typeof value.camera_id === 'string'
    && typeof value.status === 'string'
    && isInterval(value.interval)
}

function isPipelineStage(value: unknown): value is PipelineStage {
  return isRecord(value)
    && typeof value.name === 'string'
    && (value.state === 'COMPLETE' || value.state === 'NOT_RUN')
    && isStringOrNull(value.semantic_sha256)
}

function isRunDecision(value: unknown): value is RunDecision {
  return isRecord(value)
    && typeof value.decision === 'string'
    && isStringOrNull(value.reason_code)
    && typeof value.admitted_claim_count === 'number'
}

function isRunHypothesis(value: unknown): value is RunHypothesis {
  return isRecord(value)
    && typeof value.ordinal === 'number'
    && typeof value.logical_key === 'string'
    && typeof value.semantic_sha256 === 'string'
    && isInterval(value.effective_interval)
}

function isRunPublication(value: unknown): value is RunPublication {
  return isRecord(value)
    && typeof value.event_id === 'string'
    && typeof value.revision_id === 'string'
    && isInterval(value.effective_interval)
}

function isEvidence(value: unknown): value is Evidence {
  return isRecord(value)
    && typeof value.role === 'string'
    && typeof value.schema_id === 'string'
    && typeof value.schema_version === 'string'
    && typeof value.semantic_sha256 === 'string'
    && typeof value.exact_bytes_sha256 === 'string'
    && typeof value.byte_count === 'number'
}

function isRunIntegrity(value: unknown): value is RunIntegrity {
  return isRecord(value)
    && typeof value.command_sha256 === 'string'
    && typeof value.completion_semantic_sha256 === 'string'
}

function isRunSnapshot(value: unknown): value is RunSnapshot {
  if (!isRunSummary(value) || !isRecord(value)) {
    return false
  }

  return typeof value.evidence_class === 'string'
    && typeof value.production_eligible === 'boolean'
    && (value.window === null || isRunWindow(value.window))
    && Array.isArray(value.packages) && value.packages.every(isRunPackage)
    && Array.isArray(value.camera_quality) && value.camera_quality.every(isCameraQuality)
    && Array.isArray(value.stages) && value.stages.every(isPipelineStage)
    && (value.decision === null || isRunDecision(value.decision))
    && Array.isArray(value.hypotheses) && value.hypotheses.every(isRunHypothesis)
    && Array.isArray(value.publications) && value.publications.every(isRunPublication)
    && isRunIntegrity(value.integrity)
    && Array.isArray(value.evidence) && value.evidence.every(isEvidence)
}
