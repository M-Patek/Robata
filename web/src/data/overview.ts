import { Node, Edge } from 'reactflow'
import { PIPELINE_GROUPS } from './groups'
import { EdgeSchema } from '@/types'

const COL = (n: number) => n * 320
const ROW = 60  // single horizontal row

export const OVERVIEW_NODES: Node[] = PIPELINE_GROUPS.map((g, i) => ({
  id: g.id,
  type: 'group',
  position: { x: COL(i), y: ROW },
  data: { group: g },
}))

type OverviewEdge = { source: string; target: string; schema: EdgeSchema }
const OVERVIEW_EDGE_DEFS: OverviewEdge[] = [
  { source: 'ingestion', target: 'quality',    schema: 'QA' },
  { source: 'quality',   target: 'events',     schema: 'PROPOSAL' },
  { source: 'events',    target: 'fusion',     schema: 'EVIDENCE' },
  { source: 'fusion',    target: 'completion', schema: 'COMPLETION' },
]

export const OVERVIEW_EDGES: Edge[] = OVERVIEW_EDGE_DEFS.map((d, i) => ({
  id: `oe-${i}`,
  source: d.source,
  target: d.target,
  type: 'schema',
  data: { schema: d.schema },
}))
