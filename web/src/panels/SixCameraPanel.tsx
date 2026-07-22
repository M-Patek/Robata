import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Box, Sphere } from '@react-three/drei'
import * as THREE from 'three'
import { usePipelineStore } from '@/store'
import { NodeStatus } from '@/types'

const CAMERA_POSITIONS: [number, number, number][] = [
  [-2.5,  1.5,  2.0],  // CAM_01
  [ 2.5,  1.5,  2.0],  // CAM_02
  [-2.5,  1.5, -2.0],  // CAM_03
  [ 2.5,  1.5, -2.0],  // CAM_04
  [ 0.0,  2.5,  0.0],  // CAM_05 — overhead
  [ 0.0,  0.5,  2.8],  // CAM_06 — low front
]

const CAMERA_IDS = ['CAM_01', 'CAM_02', 'CAM_03', 'CAM_04', 'CAM_05', 'CAM_06']

const STATUS_GLOW: Record<NodeStatus, string> = {
  PENDING:        '#4b5563',
  RUNNING:        '#3b82f6',
  COMPLETE:       '#22c55e',
  FAILED:         '#ef4444',
  WAITING_REVIEW: '#eab308',
  BLOCKED:        '#a855f7',
  NO_EVENTS:      '#64748b',
}

function CameraMount({
  position,
  camId,
  status,
  active,
  onClick,
}: {
  position: [number, number, number]
  camId: string
  status: NodeStatus
  active: boolean
  onClick: () => void
}) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const glowColor = STATUS_GLOW[status]

  useFrame((_, delta) => {
    if (status === 'RUNNING' && meshRef.current) {
      meshRef.current.rotation.y += delta * 1.5
    }
  })

  return (
    <group position={position} onClick={onClick}>
      {/* Camera body */}
      <Box ref={meshRef} args={[0.35, 0.25, 0.45]} castShadow>
        <meshStandardMaterial
          color={glowColor}
          emissive={glowColor}
          emissiveIntensity={active ? 0.8 : 0.3}
          roughness={0.4}
          metalness={0.6}
        />
      </Box>
      {/* Lens */}
      <Sphere args={[0.1, 16, 16]} position={[0, 0, 0.28]}>
        <meshStandardMaterial
          color="#1e293b"
          emissive={active ? glowColor : '#000'}
          emissiveIntensity={active ? 1.2 : 0}
          roughness={0.1}
          metalness={0.9}
        />
      </Sphere>
      {/* Label */}
      <Text
        position={[0, -0.28, 0]}
        fontSize={0.15}
        color="#94a3b8"
        anchorX="center"
        anchorY="top"
      >
        {camId}
      </Text>
      {/* Active ring */}
      {active && (
        <mesh>
          <ringGeometry args={[0.28, 0.32, 32]} />
          <meshBasicMaterial color={glowColor} side={THREE.DoubleSide} transparent opacity={0.7} />
        </mesh>
      )}
    </group>
  )
}

function FloorGrid() {
  return (
    <gridHelper args={[8, 16, '#1e293b', '#1e293b']} position={[0, -0.5, 0]} />
  )
}

function SubjectVolume() {
  return (
    <group position={[0, 0.5, 0]}>
      <Box args={[1.2, 1.8, 0.6]} position={[0, 0, 0]}>
        <meshStandardMaterial
          color="#0ea5e9"
          transparent
          opacity={0.15}
          wireframe
        />
      </Box>
      <Text position={[0, 1.2, 0]} fontSize={0.12} color="#38bdf8" anchorX="center">
        subject volume
      </Text>
    </group>
  )
}

export default function SixCameraPanel() {
  const activeRun = usePipelineStore((s) => s.activeRun)
  const toggleSixCameraPanel = usePipelineStore((s) => s.toggleSixCameraPanel)

  // For now, all cameras at 'COMPLETE' unless run says otherwise
  const cameraStatuses = useMemo((): NodeStatus[] => {
    if (!activeRun) return Array(6).fill('PENDING') as NodeStatus[]
    const s = activeRun.status
    return Array(6).fill(s) as NodeStatus[]
  }, [activeRun])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-canvas-border">
        <div>
          <h3 className="text-sm font-semibold text-white">Six-Camera Layout</h3>
          <p className="text-[10px] text-gray-500 mt-0.5">
            Multi-view geometry • Section 25.1
          </p>
        </div>
        <button
          onClick={toggleSixCameraPanel}
          className="text-gray-500 hover:text-white text-lg leading-none px-1"
        >
          ✕
        </button>
      </div>

      {/* 3D Canvas */}
      <div className="flex-1 min-h-0">
        <Canvas
          camera={{ position: [0, 4, 7], fov: 50 }}
          shadows
          className="w-full h-full"
        >
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 8, 5]} intensity={0.8} castShadow />
          <pointLight position={[-3, 4, -3]} intensity={0.3} color="#6366f1" />

          <FloorGrid />
          <SubjectVolume />

          {CAMERA_IDS.map((camId, i) => (
            <CameraMount
              key={camId}
              camId={camId}
              position={CAMERA_POSITIONS[i]}
              status={cameraStatuses[i]}
              active={cameraStatuses[i] === 'RUNNING'}
              onClick={() => {}}
            />
          ))}

          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={3}
            maxDistance={15}
          />
        </Canvas>
      </div>

      {/* Camera grid legend */}
      <div className="px-4 py-3 border-t border-canvas-border">
        <div className="grid grid-cols-3 gap-1.5">
          {CAMERA_IDS.map((camId, i) => (
            <div
              key={camId}
              className="flex items-center gap-1.5 px-2 py-1.5 rounded bg-canvas-bg border border-canvas-border"
            >
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: STATUS_GLOW[cameraStatuses[i]] }}
              />
              <span className="text-[10px] font-mono text-gray-300">{camId}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
