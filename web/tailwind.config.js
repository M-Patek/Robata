/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Node status colors
        node: {
          pending:   '#374151', // gray-700
          running:   '#1d4ed8', // blue-700
          complete:  '#15803d', // green-700
          failed:    '#b91c1c', // red-700
          review:    '#a16207', // yellow-700
          blocked:   '#6b21a8', // purple-700
        },
        // Schema edge type colors
        edge: {
          source:     '#06b6d4', // cyan
          qa:         '#6366f1', // indigo
          proposal:   '#f59e0b', // amber
          candidate:  '#10b981', // emerald
          evidence:   '#f97316', // orange
          fusion:     '#ec4899', // pink
          completion: '#8b5cf6', // violet
          review:     '#14b8a6', // teal
        },
        canvas: {
          bg:     '#0f1117',
          grid:   '#1a1d27',
          panel:  '#16181f',
          border: '#2a2d3a',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-fast': 'pulse 0.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow':  'spin 3s linear infinite',
      },
    },
  },
  plugins: [],
}
