import { float16AsUintToFloat } from "./float16"

export function mkRuntime(mem: Float32Array) {
    return {
        softmax: (ptr: number, len: number) => {
            let max = mem[ptr]
            for (let i = 1; i < len; ++i)
                max = Math.max(mem[ptr + i], max)
            let sum = 0
            for (let i = 0; i < len; ++i)
                sum += (mem[ptr + i] = Math.exp(mem[ptr + i] - max))
            for (let i = 0; i < len; ++i)
                mem[ptr + i] /= sum
        },
        f32: (v: number) => {
            const arr = new Float32Array(1)
            arr[0] = v
            return arr[0]
        },
        vcvtb_f32_f16: (v: number) =>
            float16AsUintToFloat(v & 0xffff),
        vcvtt_f32_f16: (v: number) =>
            float16AsUintToFloat((v >> 16) & 0xffff),
    }
}
