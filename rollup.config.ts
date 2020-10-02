import commonjs from 'rollup-plugin-commonjs';
import typescript from 'rollup-plugin-typescript2';
import json from 'rollup-plugin-json'

export default [
  { src: 'src/main.ts', dst: 'built/ml4f.js' }
].map(({ src, dst }) => ({
  external: ['@tensorflow/tfjs'],

  input: src,
  plugins: [
    json(),
    typescript({
      tsconfigOverride: {
        compilerOptions: {
          module: 'ES2015',
          declaration: false
        }
      }
    }),
    commonjs({
      include: 'node_modules/**',
    })
  ],
  output: {
    extend: true,
    file: dst,
    format: 'umd',
    name: 'ml4f',
    globals: {
      '@tensorflow/tfjs': 'tf'
    },
    sourcemap: true
  },
  onwarn: (warning) => {
    const ignoreWarnings = ['EVAL']
    if (ignoreWarnings.indexOf(warning.code) >= 0)
      return

    console.warn(warning.code, warning.message)
  }
}))
