import * as tf from '@tensorflow/tfjs'
import * as tfi from './tfi'
import * as U from './util'

interface ModelInfo {
    weights: number[];
    opts: Options;
}

interface LayerInfo {
    layer: tf.layers.Layer;
    model: ModelInfo;
    inputShape: tf.Shape;
    outputShape: tf.Shape;
    inputOff: number;
    outputOff: number;
}

enum OpCode {
    repeat,
    loadWeightAddr,
    loadDataAddr,
    advancePtr,
    loadConst,
    load,
    store,
    vmul,
    vadd,
    relu,
}

enum Reg {
    S0 = 0,
    S1 = 1,
    S15 = 15,
    S31 = 32,
    InputPtr = 200,
    OutputPtr,
    KernelPtr,
    Index0 = 300,
}

interface Op {
    opcode: OpCode
    dst?: Reg
    src?: Reg
    srcAlt?: Reg
    num?: number
    adv?: boolean
    body?: Op[]
}



const numRegs = 32

function unsupported(msg: string) {
    debugger
    throw new Error("Unsupported operator or config: " + msg)
}

function assert(cond: boolean, msg = "assertion failed") {
    if (!cond)
        unsupported(msg)
}

function getLayerInfo(l: tf.layers.Layer) {
    const ll = l as any
    let r = ll.__ml4f_info as LayerInfo
    if (!r) {
        r = {
            layer: l
        } as any
        ll.__ml4f_info = r
    }
    return r
}

function indent(s: string) {
    return "  " + s.replace(/\n$/, "").replace(/\n/g, "\n  ") + "\n"
}

function reg(r: Reg) {
    if (r <= Reg.S31)
        return "s" + (r - Reg.S0)
    if (r >= Reg.Index0)
        return "idx" + (r - Reg.Index0)
    switch (r) {
        case Reg.InputPtr:
            return "input"
        case Reg.KernelPtr:
            return "kernel"
        case Reg.OutputPtr:
            return "output"
        default:
            assert(false, "bad reg " + r)
    }
}

function numCycles(ops: Op[]): number {
    let cycles = 0
    let prevDst: Reg = null
    for (const op of ops) {
        switch (op.opcode) {
            case OpCode.repeat:
                cycles += (numCycles(op.body) + 4) * op.num + 1
                break
            case OpCode.loadWeightAddr:
                cycles += 2
                break
            case OpCode.loadDataAddr:
                cycles += 2
                break
            case OpCode.advancePtr:
                if (op.num == 1)
                    cycles += 1
                else
                    cycles += 3
                break
            case OpCode.loadConst:
                if (op.num == 0 || op.num == 1)
                    cycles += 1
                cycles += 4
                break
            case OpCode.load:
                cycles += 1 + op.num
                break
            case OpCode.store:
                cycles += 1 + op.num
                break
            case OpCode.relu:
                cycles += 5
                break
            case OpCode.vmul:
            case OpCode.vadd:
                if (op.src === prevDst || op.srcAlt === prevDst)
                    cycles += 2
                else
                    cycles += 1
                prevDst = op.dst
                break
            default:
                throw new Error("bad op " + op.opcode)
        }
    }
    return cycles
}

function toJS(op: Op): string {
    const dst = op.dst == null ? null : reg(op.dst)
    const src = op.src == null ? null : reg(op.src)

    switch (op.opcode) {
        case OpCode.repeat:
            return `for (let ${dst} = 0; ${dst} < ${op.num}; ${dst}++) {\n${indent(toJSs(op.body))}}\n`
        case OpCode.loadWeightAddr:
            return `${dst} = weightOff + ${op.num}\n`
        case OpCode.loadDataAddr:
            return `${dst} = dataOff + ${op.num}\n`
        case OpCode.advancePtr:
            if (op.src == null)
                return `${dst} += ${op.num}\n`
            return `${dst} += ${reg(op.src)}${op.num == 1 ? "" : " * " + op.num}\n`
        case OpCode.loadConst:
            return `${dst} = ${op.num}\n`
        case OpCode.load: {
            let r = ""
            let dp = op.dst + 0
            for (let i = 0; i < op.num; ++i)
                r += `${reg(dp++)} = mem[${src} + ${i}]\n`
            if (op.adv)
                r += `${src} += ${op.num}\n`
            return r
        }
        case OpCode.store: {
            let r = ""
            let dp = op.src + 0
            for (let i = 0; i < op.num; ++i)
                r += ` mem[${dst} + ${i}] = ${reg(dp++)}\n`
            if (op.adv)
                r += `${dst} += ${op.num}\n`
            return r
        }
        case OpCode.relu:
            return `if (mem[${dst}] < 0) mem[${dst}] = 0; ${dst}++\n`
        case OpCode.vmul:
            return `${dst} = ${src} * ${reg(op.srcAlt)}\n`
        case OpCode.vadd:
            return `${dst} = ${src} + ${reg(op.srcAlt)}\n`
        default:
            throw new Error("bad op " + op.opcode)
    }
}

function toJSs(op: Op[]) {
    return op.map(toJS).join("")
}

let repIdx = 0
function repeat(n: number, f: (idx: Reg) => Op[]): Op {
    const idx = Reg.Index0 + repIdx++
    return {
        opcode: OpCode.repeat,
        dst: idx,
        num: n,
        body: f(idx)
    }
}

function loadWeightAddr(dst: Reg, idx: number): Op {
    assert(idx >= 0)
    return {
        opcode: OpCode.loadWeightAddr,
        dst,
        num: idx
    }
}

function loadDataAddr(dst: Reg, idx: number): Op {
    assert(idx >= 0)
    return {
        opcode: OpCode.loadDataAddr,
        dst,
        num: idx
    }
}

function advancePtr(dst: Reg, idx: Reg | null, mult = 1): Op {
    return {
        opcode: OpCode.advancePtr,
        dst,
        src: idx,
        num: mult
    }
}

function load0(dst: number): Op {
    return {
        opcode: OpCode.loadConst,
        dst,
        num: 0.0
    }
}

function load(dst: Reg, num: number, src: Reg, adv: boolean): Op {
    return {
        opcode: OpCode.load,
        dst,
        src,
        num: num,
        adv
    }
}

function store(dst: Reg, src: Reg, num: number, adv: boolean): Op {
    return {
        opcode: OpCode.store,
        dst,
        src,
        num: num,
        adv
    }
}

function relu(dst: Reg): Op {
    return {
        opcode: OpCode.relu,
        dst,
        adv: true
    }
}

function vmul(dst: Reg, a: Reg, b: Reg) {
    return {
        opcode: OpCode.vmul,
        dst,
        src: a,
        srcAlt: b,
    }
}

function vadd(dst: Reg, a: Reg, b: Reg) {
    return {
        opcode: OpCode.vadd,
        dst,
        src: a,
        srcAlt: b,
    }
}

function flatten(...args: (Op | Op[] | Op[][])[]) {
    const res: Op[] = []
    const add = (a: Op) => {
        if (a) res.push(a)
    }
    for (const a of args) {
        if (Array.isArray(a)) {
            for (const b of a) {
                if (Array.isArray(b)) {
                    for (const c of b)
                        add(c)
                } else {
                    add(b)
                }
            }
        } else {
            add(a)
        }
    }
    return res
}

function compileConv2D(layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as tfi.ConvLayerArgs
    const info = getLayerInfo(layer)
    const memRegs = numRegs >> 1
    const flashRegs = numRegs >> 1

    if (info.model.opts.verbose)
        console.log(info.inputShape, info.outputShape, config)

    if (info.inputShape.length != 4)
        unsupported("inputShape: " + info.inputShape.length)
    if (config.dataFormat != "channelsLast")
        unsupported("dataFormat: " + config.dataFormat)
    if (config.dtype && config.dtype != "float32")
        unsupported("dtype: " + config.dtype)

    const weights = layer.weights[0].read().arraySync() as number[][][][]

    const kh = config.kernelSize[0]
    const kw = config.kernelSize[1]

    const strh = config.strides[0]
    const strw = config.strides[1]

    const inph = info.inputShape[1]
    const inpw = info.inputShape[2]
    const inpch = info.inputShape[3]

    // padding not implemented yet
    assert(kh <= inph, "KH2")
    assert(kw <= inpw, "KW2")

    assert(weights.length == kh, "KH")
    assert(weights[0].length == kw, "KW")
    assert(weights[0][0].length == inpch, "CH")
    assert(weights[0][0][0].length == config.filters, "F")
    assert(info.outputShape[3] == config.filters, "FF")

    const weightData = info.model.weights
    const weightsIdx = weightData.length
    const bias = config.useBias ? layer.weights[1].read().arraySync() as number[] : null

    const addParam = (v: number) => {
        assert(v != null)
        weightData.push(v)
    }

    let sz = 0
    for (let f = 0; f < config.filters; f++) {
        if (bias)
            addParam(bias[f])
        for (let y = 0; y < kh; y++)
            for (let x = 0; x < kw; x++)
                for (let c = 0; c < inpch; ++c)
                    addParam(weights[y][x][c][f])
        if (sz == 0)
            sz = weightData.length - weightsIdx
    }

    const res = [
        loadWeightAddr(Reg.KernelPtr, weightsIdx),
        repeat(config.filters, filt => {
            const res: Op[] = []

            const setOutput = (res: Op[]) => {
                res.push(loadDataAddr(Reg.OutputPtr, info.outputOff))
                res.push(advancePtr(Reg.OutputPtr, filt))
            }

            // set bias
            setOutput(res)
            if (config.useBias)
                res.push(load(Reg.S0, 1, Reg.KernelPtr, true))
            else
                res.push(load0(Reg.S0))

            res.push(
                repeat(info.outputShape[1] * info.outputShape[2], () => [
                    store(Reg.OutputPtr, 0, 1, false),
                    advancePtr(Reg.OutputPtr, null, config.filters)
                ]))

            res.push(repeat(kh, kline => {
                const res: Op[] = []
                const kernSz = kw * inpch
                let chunk = 0
                for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
                    chunk = kernSz - kernOff
                    if (chunk > flashRegs)
                        chunk = flashRegs
                    res.push(load(memRegs, chunk, Reg.KernelPtr, true))

                    res.push(loadDataAddr(Reg.InputPtr, info.inputOff + kernOff))
                    res.push(advancePtr(Reg.InputPtr, kline, inpw * inpch))

                    setOutput(res)

                    const wSkip = strw * inpch
                    const hSkip = strh * inpw * inpch

                    res.push(repeat(info.outputShape[1], () =>
                        [repeat(info.outputShape[2], () => flatten(
                            load(Reg.S0, chunk, Reg.InputPtr, true),
                            advancePtr(Reg.InputPtr, null, wSkip - chunk),
                            U.range(chunk + 1).map(i =>
                                [
                                    i < chunk ? vmul(i, i, i + memRegs) : null,
                                    i >= 2 ? vadd(Reg.S0, Reg.S0, i - 1) : null
                                ]),
                            load(Reg.S1, 1, Reg.OutputPtr, false),
                            vadd(Reg.S0, Reg.S0, Reg.S1),
                            store(Reg.OutputPtr, Reg.S0, 1, false),
                            advancePtr(Reg.OutputPtr, null, config.filters)
                        )),
                        advancePtr(Reg.InputPtr, null, hSkip - info.outputShape[2] * wSkip)]))
                }

                return res
            }))

            return res
        })]

    // maybe fold activation into last row of convolution?
    if (config.activation == "relu")
        res.push(
            loadDataAddr(Reg.OutputPtr, info.outputOff),
            repeat(shapeElts(info.outputShape), () => [relu(Reg.OutputPtr)])
        )
    else
        unsupported("activation: " + config.activation)

    return res
}

function noop(l: tf.layers.Layer): Op[] {
    return []
}

export function shapeElts(shape: tf.Shape) {
    let r = 1
    for (const s of shape)
        if (s != null)
            r *= s
    return r
}

const compilers: SMap<(l: tf.layers.Layer) => Op[]> = {
    Conv2D: compileConv2D,
    Dropout: noop,
    Flatten: noop,
}

function isInPlace(layer: tf.layers.Layer) {
    switch (layer.getClassName()) {
        case "Dropout":
        case "Flatten":
            return true
        default:
            return false
    }
}

export interface Options {
    verbose?: boolean
}

export function compileModel(m: tf.LayersModel, opts: Options = {}) {
    if (opts.verbose)
        m.summary()

    const modelInfo: ModelInfo = {
        weights: [],
        opts
    }

    const inputShape = m.layers[0].batchInputShape
    let maxSize = [shapeElts(inputShape), 0]
    let currIdx = 0
    let prev: LayerInfo
    for (const l of m.layers) {
        const info = getLayerInfo(l)
        info.model = modelInfo
        if (prev) {
            info.inputShape = prev.outputShape
        } else {
            info.inputShape = inputShape
        }
        info.outputShape = l.computeOutputShape(info.inputShape) as tf.Shape
        const elts = shapeElts(info.outputShape)
        info.inputOff = currIdx
        if (!isInPlace(l))
            currIdx = currIdx == 0 ? 1 : 0
        info.outputOff = currIdx
        if (elts > maxSize[currIdx])
            maxSize[currIdx] = elts
        prev = info
    }

    // TODO alignment?
    const midOff = maxSize[0]
    for (const l of m.layers) {
        const info = getLayerInfo(l)
        if (info.inputOff) info.inputOff = midOff
        if (info.outputOff) info.outputOff = midOff
    }

    const arenaSize = maxSize[0] + maxSize[1]

    const ops: Op[][] = []

    for (const l of m.layers) {
        const info = getLayerInfo(l)
        if (opts.verbose)
            console.log(l.getClassName(), info.inputShape, info.inputOff, info.outputOff)
        const f = compilers[l.getClassName()]
        if (f)
            ops.push(f(l))
        else
            console.log("unsupported layer: ", l.getClassName())
    }

    const flat = flatten(ops)

    const lastInfo = getLayerInfo(m.layers[m.layers.length - 1])

    let fn = `
(weights => {
    "use strict";
    const weightOff = ${arenaSize}
    const dataOff = 0
    const mem = new Float32Array(weightOff + ${modelInfo.weights.length})
    mem.fill(1000.2342)
    mem.set(weights, weightOff)
    return (inputs => {
        if (inputs.length != ${shapeElts(getLayerInfo(m.layers[0]).inputShape)})
            throw new Error("invalid input size")
        mem.set(inputs, dataOff)
        let input, output, kernel
        let ${U.range(numRegs).map(r => "s" + r).join(", ")}

${toJSs(flat)}
        
        return mem.slice(${lastInfo.outputOff}, ${lastInfo.outputOff + shapeElts(lastInfo.outputShape)})
    })
})
`

    if (opts.verbose) {
        console.log(fn)
        console.log("cycles:", numCycles(flat))
    }

    const modelFn: (inp: ArrayLike<number>) => Float32Array = (eval(fn))(modelInfo.weights)
    return modelFn
}
