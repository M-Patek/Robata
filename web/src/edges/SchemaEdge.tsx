import { memo } from 'react'
import { EdgeProps, getBezierPath, EdgeLabelRenderer, BaseEdge } from 'reactflow'
import { EDGE_COLORS, EdgeSchema } from '@/types'

interface SchemaEdgeData {
  schema: EdgeSchema
  label?: string
}

function SchemaEdge({
  id,
  sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition,
  data,
  markerEnd,
}: EdgeProps<SchemaEdgeData>) {
  const schema: EdgeSchema = data?.schema ?? 'SOURCE'
  const color = EDGE_COLORS[schema]

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX, sourceY, sourcePosition,
    targetX, targetY, targetPosition,
  })

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{ stroke: color, strokeWidth: 2, opacity: 0.75 }}
      />
      {data?.label && (
        <EdgeLabelRenderer>
          <div
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)` }}
            className="absolute pointer-events-none"
          >
            <span
              className="px-1.5 py-0.5 rounded text-[9px] font-mono font-medium bg-canvas-panel border"
              style={{ color, borderColor: color + '55' }}
            >
              {data.label}
            </span>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}

export default memo(SchemaEdge)
