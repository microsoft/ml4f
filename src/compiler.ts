///<reference path="pxtpackage.d.ts" />

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
    addPtr,
    loadConst,
    load,
    store,
    vmul,
    vmax,
    vadd,
    relu,
    fcall,
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
    Tmp0 = 400,
    Zero = 500,
}

interface Op {
    opcode: OpCode
    dst?: Reg
    src?: Reg
    srcAlt?: Reg
    num?: number
    adv?: boolean
    body?: Op[]
    fname?: string
}

const numFPRegs = 32
const numTmpRegs = 8
const unrollLimit = 10

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
    if (r >= Reg.Zero)
        return r - Reg.Zero
    if (r >= Reg.Tmp0)
        return "tmp" + (r - Reg.Tmp0)
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
            case OpCode.addPtr:
                if (op.num == 1)
                    cycles += 1
                else
                    cycles += 3
                break
            case OpCode.loadConst:
                if (op.num == 0 || op.num == 1)
                    cycles += 1
                else
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
            case OpCode.vmax:
                cycles += 4
                break
            case OpCode.vmul:
            case OpCode.vadd:
                if (op.src === prevDst || op.srcAlt === prevDst)
                    cycles += 2
                else
                    cycles += 1
                prevDst = op.dst
                break
            case OpCode.fcall:
                cycles += op.num * 200 // estimate
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
    const srcAlt = op.srcAlt == null ? null : reg(op.srcAlt)

    switch (op.opcode) {
        case OpCode.repeat:
            return `for (let ${dst} = 0; ${dst} < ${op.num}; ${dst}++) {\n${indent(toJSs(op.body))}}\n`
        case OpCode.loadWeightAddr:
            return `${dst} = weightOff + ${op.num}\n`
        case OpCode.loadDataAddr:
            return `${dst} = dataOff + ${op.num}\n`
        case OpCode.addPtr:
            if (op.src == null)
                return `${dst} = ${srcAlt} + ${op.num}\n`
            return `${dst} = ${srcAlt} + ${src}${op.num == 1 ? "" : " * " + op.num}\n`
        case OpCode.loadConst:
            return `${dst} = ${op.num}\n`
        case OpCode.load: {
            let r = ""
            let dp = op.dst + 0
            if (op.adv) {
                for (let i = 0; i < op.num; ++i)
                    r += `${reg(dp++)} = mem[${src}++]\n`
            } else {
                for (let i = 0; i < op.num; ++i)
                    r += `${reg(dp++)} = mem[${src} + ${i}]\n`
            }
            return r
        }
        case OpCode.store: {
            let r = ""
            let dp = op.src + 0
            if (op.adv) {
                for (let i = 0; i < op.num; ++i)
                    r += `mem[${dst}++] = ${reg(dp++)}\n`
            } else {
                for (let i = 0; i < op.num; ++i)
                    r += `mem[${dst} + ${i}] = ${reg(dp++)}\n`
            }
            return r
        }
        case OpCode.relu:
            return `if (mem[${dst}] < 0) mem[${dst}] = 0; ${dst}++\n`
        case OpCode.vmul:
            return `${dst} = ${src} * ${srcAlt}\n`
        case OpCode.vadd:
            return `${dst} = ${src} + ${srcAlt}\n`
        case OpCode.vmax:
            return `${dst} = Math.max(${src}, ${srcAlt})\n`
        case OpCode.fcall:
            return `${op.fname}(${dst}, ${op.num})`
        default:
            throw new Error("bad op " + op.opcode)
    }
}

function toJSs(op: Op[]) {
    return op.map(toJS).join("")
}

let repIdx = 0
function repeatIdx(n: number, f: (idx: Reg) => Op[]): Op {
    const idx = Reg.Index0 + repIdx++
    return {
        opcode: OpCode.repeat,
        dst: idx,
        num: n,
        body: f(idx),
        adv: true
    }
}

function repeat(n: number, f: () => Op[]): Op {
    const r = repeatIdx(n, f)
    r.adv = false
    return r
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

function addPtr(dst: Reg, idx: Reg | null, mult = 1, base?: Reg): Op {
    if (!base) base = dst
    return {
        opcode: OpCode.addPtr,
        dst,
        src: idx,
        srcAlt: base,
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

function vmax(dst: Reg, a: Reg, b: Reg) {
    return {
        opcode: OpCode.vmax,
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

function fcall(name: string, dst: Reg, len: number): Op {
    return {
        opcode: OpCode.fcall,
        fname: name,
        dst,
        num: len,
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

function validateConfig(layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as (tfi.Pooling2DLayerArgs | tfi.BaseConvLayerArgs)
    const info = getLayerInfo(layer)

    if (info.model.opts.verbose)
        console.log(info.inputShape, info.outputShape, config)

    if (info.inputShape.length != 4)
        unsupported("inputShape: " + info.inputShape.length)
    if (config.dataFormat != "channelsLast")
        unsupported("dataFormat: " + config.dataFormat)
    if (config.dtype && config.dtype != "float32")
        unsupported("dtype: " + config.dtype)
}

function addActivation(res: Op[], layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as tfi.DenseLayerArgs
    const info = getLayerInfo(layer)
    const numoutp = shapeElts(info.outputShape)

    res.push(loadDataAddr(Reg.OutputPtr, info.outputOff))

    if (config.activation == "relu")
        res.push(repeat(numoutp, () => [relu(Reg.OutputPtr)]))
    else if (config.activation == "softmax")
        res.push(fcall("softmax", Reg.OutputPtr, numoutp))
    else
        unsupported("activation: " + config.activation)
}

function compileConv2D(layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as tfi.ConvLayerArgs
    const info = getLayerInfo(layer)
    const memRegs = numFPRegs >> 1
    const flashRegs = numFPRegs >> 1

    validateConfig(layer)

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
        repeatIdx(config.filters, filt => {
            const res: Op[] = []

            const setOutput = (res: Op[]) => {
                res.push(loadDataAddr(Reg.OutputPtr, info.outputOff))
                res.push(addPtr(Reg.OutputPtr, filt))
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
                    addPtr(Reg.OutputPtr, null, config.filters)
                ]))

            res.push(repeatIdx(kh, kline => {
                const res: Op[] = []
                const kernSz = kw * inpch
                let chunk = 0
                for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
                    chunk = kernSz - kernOff
                    if (chunk > flashRegs)
                        chunk = flashRegs
                    res.push(load(memRegs, chunk, Reg.KernelPtr, true))

                    res.push(loadDataAddr(Reg.InputPtr, info.inputOff + kernOff))
                    res.push(addPtr(Reg.InputPtr, kline, inpw * inpch))

                    setOutput(res)

                    const wSkip = strw * inpch
                    const hSkip = strh * inpw * inpch

                    res.push(repeat(info.outputShape[1], () =>
                        [repeat(info.outputShape[2], () => flatten(
                            load(Reg.S0, chunk, Reg.InputPtr, true),
                            addPtr(Reg.InputPtr, null, wSkip - chunk),
                            U.range(chunk + 1).map(i =>
                                [
                                    i < chunk ? vmul(i, i, i + memRegs) : null,
                                    i >= 2 ? vadd(Reg.S0, Reg.S0, i - 1) : null
                                ]),
                            load(Reg.S1, 1, Reg.OutputPtr, false),
                            vadd(Reg.S0, Reg.S0, Reg.S1),
                            store(Reg.OutputPtr, Reg.S0, 1, false),
                            addPtr(Reg.OutputPtr, null, config.filters)
                        )),
                        addPtr(Reg.InputPtr, null, hSkip - info.outputShape[2] * wSkip)]))
                }

                return res
            }))

            return res
        })]

    addActivation(res, layer)

    return res
}

function compileMaxPooling2D(layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as tfi.Pooling2DLayerArgs
    const info = getLayerInfo(layer)

    validateConfig(layer)

    const kh = config.poolSize[0]
    const kw = config.poolSize[1]

    const strh = config.strides[0]
    const strw = config.strides[1]

    const inph = info.inputShape[1]
    const inpw = info.inputShape[2]
    const numch = info.inputShape[3]

    // padding not implemented yet
    assert(kh <= inph, "KH2")
    assert(kw <= inpw, "KW2")

    assert(info.outputShape[3] == info.inputShape[3], "CH")

    if (kh - 1 > numTmpRegs)
        unsupported(`too high MaxPool2D area`)

    const lineW = inpw * numch

    return [
        repeatIdx(numch, filt => {
            const res = [
                loadDataAddr(Reg.OutputPtr, info.outputOff),
                addPtr(Reg.OutputPtr, filt),
                loadDataAddr(Reg.InputPtr, info.inputOff),
                addPtr(Reg.InputPtr, filt),
            ]

            const ptrRegs = U.range(kh - 1).map(i => Reg.Tmp0 + i)
            ptrRegs.unshift(Reg.InputPtr)

            for (let i = 1; i < kh; ++i)
                res.push(addPtr(ptrRegs[i], null, lineW * i, Reg.InputPtr))

            res.push(
                repeat(info.outputShape[1], () => flatten(
                    repeat(info.outputShape[2], () => {
                        const res: Op[] = []
                        for (let i = 0; i < kh; ++i) {
                            for (let j = 0; j < kw; ++j) {
                                const reg = i == 0 && j == 0 ? Reg.S0 : Reg.S1
                                res.push(
                                    load(reg, 1, ptrRegs[i], true),
                                    addPtr(ptrRegs[i], null, numch - 1)
                                )
                                if (reg != Reg.S0)
                                    res.push(vmax(Reg.S0, Reg.S0, reg))
                            }
                            res.push(
                                addPtr(ptrRegs[i], null, (strw - kw) * numch)
                            )
                        }
                        res.push(
                            store(Reg.OutputPtr, Reg.S0, 1, true),
                            addPtr(Reg.OutputPtr, null, numch - 1)
                        )
                        return res
                    }),
                    ptrRegs.map(r => addPtr(r, null, strh * lineW - info.outputShape[2] * strw * numch)))))

            return res
        })
    ]
}

function compileDense(layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as tfi.DenseLayerArgs
    const info = getLayerInfo(layer)

    const maxChunk = (numFPRegs >> 1) - 2
    const memReg0 = Reg.S1
    const flashReg0 = memReg0 + maxChunk

    if (info.model.opts.verbose)
        console.log(info.inputShape, info.outputShape, config)

    if (info.inputShape.length != 2)
        unsupported("inputShape: " + info.inputShape.length)

    if (config.dtype && config.dtype != "float32")
        unsupported("dtype: " + config.dtype)

    const weights = layer.weights[0].read().arraySync() as number[][]
    //console.log(weights)

    const inpsize = info.inputShape[1]

    assert(weights.length == inpsize, "IH")
    assert(weights[0].length == config.units, "UN")

    const weightData = info.model.weights
    const weightsIdx = weightData.length
    const bias = config.useBias ? layer.weights[1].read().arraySync() as number[] : null
    //console.log(bias)

    const addParam = (v: number) => {
        assert(v != null)
        weightData.push(v)
    }

    for (let f = 0; f < config.units; f++) {
        if (bias)
            addParam(bias[f])
        for (let i = 0; i < inpsize; ++i)
            addParam(weights[i][f])
    }

    const res = [
        loadWeightAddr(Reg.KernelPtr, weightsIdx),
        loadDataAddr(Reg.OutputPtr, info.outputOff),
        repeat(config.units, () => {
            const res: Op[] = []

            // set bias
            if (config.useBias)
                res.push(load(Reg.S0, 1, Reg.KernelPtr, true))
            else
                res.push(load0(Reg.S0))

            res.push(loadDataAddr(Reg.InputPtr, info.inputOff))

            const addChunk = (len: number) => flatten(
                load(memReg0, len, Reg.InputPtr, true),
                load(flashReg0, len, Reg.KernelPtr, true),
                U.range(len + 1).map(i => [
                    i < len ? vmul(memReg0 + i, memReg0 + i, flashReg0 + i) : null,
                    i >= 2 ? vadd(Reg.S0, Reg.S0, memReg0 + i - 1) : null
                ])
            )

            const numRep = (inpsize / maxChunk) | 0
            if (numRep > 0)
                res.push(repeat(numRep, () => addChunk(maxChunk)))
            const left = inpsize - numRep * maxChunk
            if (left > 0)
                U.pushRange(res, addChunk(left))

            res.push(store(Reg.OutputPtr, Reg.S0, 1, true))

            return res
        })]

    addActivation(res, layer)

    return res
}

function noop(l: tf.layers.Layer): Op[] {
    return []
}

function optimize(ops: Op[], replMap: SMap<Reg> = {}): Op[] {
    const repl = (r: Reg) => {
        if (!r) return r
        if (replMap[r] != undefined)
            return replMap[r]
        return r
    }

    const res: Op[] = []
    for (let op of ops) {
        op = {
            opcode: op.opcode,
            dst: repl(op.dst),
            src: repl(op.src),
            srcAlt: repl(op.srcAlt),
            adv: op.adv,
            num: op.num,
            body: op.body,
            fname: op.fname
        }
        switch (op.opcode) {
            case OpCode.repeat:
                if (op.num == 0) { }
                else if (op.num == 1) {
                    replMap[op.dst] = Reg.Zero
                    U.pushRange(res, optimize(op.body, replMap))
                } else {
                    op.body = optimize(op.body, replMap)

                    const stripLoop = op.num * op.body.length < unrollLimit * 2
                    const canUnroll = !op.adv && 2 * op.body.length < unrollLimit

                    if (stripLoop) {
                        for (let i = 0; i < op.num; ++i) {
                            replMap[op.dst] = Reg.Zero + i
                            U.pushRange(res, optimize(op.body, replMap))
                        }
                    } else if (canUnroll) {
                        const unrollCnt = (unrollLimit / op.body.length) | 0
                        const tmp = op.body.slice()
                        for (let i = 1; i < unrollCnt; ++i)
                            U.pushRange(op.body, tmp)
                        const newnum = (op.num / unrollCnt) | 0
                        res.push(op)
                        const left = op.num - newnum * unrollCnt
                        op.num = newnum
                        for (let i = 0; i < left; ++i)
                            U.pushRange(res, tmp)
                    } else {
                        res.push(op)
                    }
                }
                break
            case OpCode.addPtr:
                if (op.dst == op.srcAlt && (op.num == 0 || op.src == Reg.Zero)) { }
                else res.push(op)
                break
            default:
                res.push(op)
        }
    }
    return res
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
    MaxPooling2D: compileMaxPooling2D,
    Dense: compileDense,
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
    repIdx = 0

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

        const f = compilers[l.getClassName()]
        if (f) {
            let tmp = f(l)
            const c0 = numCycles(tmp)
            tmp = optimize(tmp)
            const c1 = numCycles(tmp)
            if (opts.verbose)
                console.log(l.getClassName(), c0, c1, info.inputShape, info.inputOff, info.outputOff)
            ops.push(tmp)
        } else
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
    function softmax(ptr, len) {
        let max = mem[ptr]
        for (let i = 1; i < len; ++i)
            max = Math.max(mem[ptr + i], max)
        let sum = 0
        for (let i = 0; i < len; ++i)
            sum += (mem[ptr + i] = Math.exp(mem[ptr + i] - max))
        for (let i = 0; i < len; ++i)
            mem[ptr + i] /= sum
    }
    return (inputs => {
        if (inputs.length != ${shapeElts(getLayerInfo(m.layers[0]).inputShape)})
            throw new Error("invalid input size")
        mem.set(inputs, dataOff)
        let input, output, kernel
        let ${U.range(numTmpRegs).map(r => "tmp" + r).join(", ")}
        let ${U.range(numFPRegs).map(r => "s" + r).join(", ")}

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