/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cream: {
          50:  '#FDFAF5',
          100: '#F8F3E8',
          200: '#EDE4D3',
          300: '#D9CCBA',
          400: '#C4B59E',
        },
        ink: {
          900: '#1A1714',
          700: '#3D3530',
          500: '#6B5E55',
          400: '#8A7D74',
          300: '#A89B93',
          200: '#C4B9B3',
        },
        coral: {
          DEFAULT: '#C96442',
          light:   '#E08A6A',
          dark:    '#A34E30',
          muted:   '#E8C4B4',
          bg:      '#F5E8E2',
        },
        status: {
          pending:  '#8A7D74',
          running:  '#4A7FA8',
          complete: '#4A7A5A',
          failed:   '#A84A4A',
          review:   '#A87A2A',
          blocked:  '#7A4A9A',
          noevents: '#6B5E55',
        },
        edge: {
          source:    '#4A7FA8',
          qa:        '#6B5EA8',
          proposal:  '#A87A2A',
          candidate: '#4A7A5A',
          evidence:  '#A85A3A',
          fusion:    '#A84A7A',
          completion:'#6A4AA8',
          review:    '#3A8A88',
        },
      },
      fontFamily: {
        serif: ['Lora', 'Georgia', 'serif'],
        sans:  ['Inter', 'system-ui', 'sans-serif'],
        mono:  ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        node: '0 1px 4px rgba(26,23,20,0.08), 0 0 0 1px rgba(26,23,20,0.06)',
        'node-hover': '0 4px 12px rgba(26,23,20,0.12), 0 0 0 1px rgba(26,23,20,0.08)',
        'node-selected': '0 4px 16px rgba(201,100,66,0.18), 0 0 0 2px rgba(201,100,66,0.5)',
      },
    },
  },
  plugins: [],
}
