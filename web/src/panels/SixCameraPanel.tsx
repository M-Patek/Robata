import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Box, Sphere } from '@react-three/drei'
import * as THREE from 'three'
import { usePipelineStore } from '@/store'
import { NodeStatus, STATUS_STYLE } from '@/types'

const CAMERA_POSITIONS: [number, number, number][] = [
  [-2.5,  1.5,  2.0],
  [ 2.5,  1.5,  2.0],
  [-2.5,  1.5, -2.0],
  [ 2.5,  1.5, -2.0],
  [ 0.0,  2.8,  0.0],
  [ 0.0,  0.5,  2.8],
]
const CAMERA_IDS = ['CAM_01', 'CAM_02', 'CAM_03', 'CAM_04', 'CAM_05', 'CAM_06']

function CameraMount({
  position, camId, status, active,
}: {
  position: [number, number, number]
  camId: string
  status: NodeStatus
  active: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const dot = STATUS_STYLE[status].dot

  useFrame((_, delta) => {
    if (status === 'RUNNING' && meshRef.current) {
      meshRef.current.rotation.y += delta * 1.2
    }
  })

  return (
    <group position={position}>
      <Box ref={meshRef} args={[0.32, 0.22, 0.42]} castShadow>
        <meshStandardMaterial color={dot} emissive={dot}
          emissiveIntensity={active ? 0.5 : 0.15} roughness={0.5} metalness={0.5} />
      </Box>
      <Sphere args={[0.09, 16, 16]} position={[0, 0, 0.25]}>
        <meshStandardMaterial color="#2C2420"
          emissive={active ? dot : '#000'}
          emissiveIntensity={active ? 1.0 : 0}
          roughness={0.15} metalness={0.85} />
      </Sphere>
      <Text position={[0, -0.26, 0]} fontSize={0.13} color="#8A7D74"
        anchorX="center" anchorY="top" font={undefined}>
        {camId}
      </Text>
      {active && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.26, 0.30, 32]} />
          <meshBasicMaterial color={dot} side={THREE.DoubleSide} transparent opacity={0.6} />
        </mesh>
      )}
    </group>
  )
}

export default function SixCameraPanel() {
  const activeRun = usePipelineStore((s) => s.activeRun)
  const toggleSixCameraPanel = usePipelineStore((s) => s.toggleSixCameraPanel)

  const cameraStatuses = useMemo((): NodeStatus[] => {
    if (!activeRun) return Array(6).fill('PENDING') as NodeStatus[]
    return Array(6).fill(activeRun.status) as NodeStatus[]
  }, [activeRun])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid rgba(26,23,20,0.08)' }}>
        <div>
          <h3 className="text-sm font-semibold"
            style={{ fontFamily: 'Lora, serif', color: '#1A1714' }}>
            Six-Camera Layout
          </h3>
          <p className="text-[10px] mt-0.5" style={{ color: '#A89B93' }}>
            Multi-view geometry · V1.1 §25.1
          </p>
        </div>
        <button onClick={toggleSixCameraPanel}
          className="w-5 h-5 flex items-center justify-center rounded"
          style={{ color: '#A89B93' }}>
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
      </div>

      <div className="flex-1 min-h-0" style={{ background: '#F8F3E8' }}>
        <Canvas camera={{ position: [0, 4, 7], fov: 50 }} shadows>
          <ambientLight intensity={0.6} color="#FFF8F0" />
          <directionalLight position={[5, 8, 5]} intensity={0.7} castShadow
            color="#FFF4E8" />
          <pointLight position={[-3, 4, -3]} intensity={0.2} color="#E8D4C0" />
          <gridHelper args={[8, 16, '#D9CCBA', '#D9CCBA']} position={[0, -0.5, 0]} />
          <group position={[0, 0.5, 0]}>
            <Box args={[1.2, 1.8, 0.6]}>
              <meshStandardMaterial color="#4A7FA8" transparent opacity={0.08} wireframe />
            </Box>
          </group>
          {CAMERA_IDS.map((camId, i) => (
            <CameraMount key={camId} camId={camId}
              position={CAMERA_POSITIONS[i]}
              status={cameraStatuses[i]}
              active={cameraStatuses[i] === 'RUNNING'} />
          ))}
          <OrbitControls enablePan enableZoom enableRotate minDistance={3} maxDistance={14} />
        </Canvas>
      </div>

      <div className="px-4 py-3 flex-shrink-0"
        style={{ borderTop: '1px solid rgba(26,23,20,0.08)' }}>
        <div className="grid grid-cols-3 gap-1.5">
          {CAMERA_IDS.map((camId, i) => (
            <div key={camId} className="flex items-center gap-1.5 px-2 py-1.5 rounded-md"
              style={{ background: '#F8F3E8', border: '1px solid rgba(26,23,20,0.08)' }}>
              <span className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: STATUS_STYLE[cameraStatuses[i]].dot }} />
              <span className="text-[10px] font-mono" style={{ color: '#6B5E55' }}>{camId}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
