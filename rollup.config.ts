import commonjs from 'rollup-plugin-commonjs';
import node from 'rollup-plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2';
import json from 'rollup-plugin-json'

export default {
  external: ['@tensorflow/tfjs', '@tensorflow/tfjs-vis'],

  input: 'src/index.ts',
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
    node(),
    commonjs({
      include: 'node_modules/**'
    })
  ],
  output: {
    extend: true,
    file: `built/full.js`,
    format: 'umd',
    name: 'gestrec',
    globals: {
      '@tensorflow/tfjs': 'tf',
      '@tensorflow/tfjs-vis': 'tfvis'
    },
    sourcemap: true
  },
  onwarn: (warning) => {
    const ignoreWarnings = ['CIRCULAR_DEPENDENCY', 'CIRCULAR', 'THIS_IS_UNDEFINED']
    if (ignoreWarnings.some(w => w === warning.code))
      return

    if (warning.missing === 'alea')
      return

    console.warn(warning.message)
  }
}