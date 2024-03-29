import commonjs from "rollup-plugin-commonjs"
import typescript from "rollup-plugin-typescript2"
import json from "rollup-plugin-json"
import dts from "rollup-plugin-dts"

function tsbuild(name, src, tsconfig) {
    return ({
        external: ["@tensorflow/tfjs"],
        input: src,
        plugins: [
            json(),
            typescript({
                tsconfig,
                tsconfigOverride: {
                    compilerOptions: {
                        module: "ES2015",
                    },
                },
            }),
            commonjs({
                include: "node_modules/**",
            }),
        ],
        output: [
            {
                extend: true,
                file: `built/${name}.js`,
                format: "umd",
                name,
                globals: {
                    "@tensorflow/tfjs": "tf",
                },
                sourcemap: true,
            },
            {
                extend: true,
                file: `built/${name}.cjs`,
                format: "cjs",
                name,
                globals: {
                    "@tensorflow/tfjs": "tf",
                },
                sourcemap: true,
            },
        ],
        onwarn: warning => {
            const ignoreWarnings = ["EVAL"]
            if (ignoreWarnings.indexOf(warning.code) >= 0) return

            console.warn(warning.code, warning.message)
        },
    })
}

export default [
    tsbuild("ml4f", "src/main.ts", undefined),
    {
        input: "./built/main.d.ts",
        output: [{ file: "built/ml4f.d.ts", format: "es" }],
        plugins: [dts()],
    },
    tsbuild("pxtml4f", "pxt/extension.ts", "pxt/tsconfig.json"),
]
