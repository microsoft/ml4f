///<reference path="pxtpackage.d.ts" />

import * as tf from '@tensorflow/tfjs'
import { asmDeps, asmFns } from './library'
import * as tfi from './tfi'
import * as U from './util'

import * as ir from "./ir"

export type Options = ir.Options

interface LayerInfo {
    layer: tf.layers.Layer;
    model: ir.ModelInfo;
    inputShape: tf.Shape;
    outputShape: tf.Shape;
    inputOff: number;
    outputOff: number;
}


const numFPRegs = 32
const numTmpRegs = 8

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

    if (config.activation == "linear")
        return // linear is identity

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

            for (let i = 1; i < kh; ++i) {
                const op = addPtr(ptrRegs[i], null, lineW * i, Reg.InputPtr)
                op.isDef = true
                res.push(op)
            }

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

    //if (info.model.opts.verbose)
    //    console.log(info.inputShape, info.outputShape, config)

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
                    i >= 1 ? vadd(Reg.S0, Reg.S0, memReg0 + i - 1) : null
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
    InputLayer: noop,
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

export interface CompileResult {
    execute: (inp: ArrayLike<number>) => Float32Array
    js: string
    thumb: string
    machineCode: Uint8Array
}

function shapeToString(shape: tf.Shape) {
    return `[${shape.filter(x => x != null).join(",")}]`
}

export function compileModel(m: tf.LayersModel, opts: Options = {}) {
    repIdx = 0

    if (opts.verbose)
        m.summary()

    const inputShape = m.layers[0].batchInputShape

    const modelInfo: ModelInfo = {
        weights: [],
        inputShape,
        outputShape: null,
        outputOffset: -1,
        arenaSize: -1,
        opts,
    }

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

    modelInfo.outputShape = prev.outputShape

    // TODO alignment?
    const midOff = maxSize[0]
    for (const l of m.layers) {
        const info = getLayerInfo(l)
        if (info.inputOff) info.inputOff = midOff
        if (info.outputOff) info.outputOff = midOff
    }

    const arenaSize = maxSize[0] + maxSize[1]
    modelInfo.arenaSize = arenaSize

    const ops: Op[][] = []

    for (const l of m.layers) {
        const info = getLayerInfo(l)

        const f = compilers[l.getClassName()]
        if (f) {
            let tmp = f(l)
            const c0 = numCycles(tmp)
            tmp = optimize(tmp)
            const c1 = numCycles(tmp)
            const optRate = 100 * (c0 - c1) / c0
            const optinfo = c0 ? `${c1} cycles (${optRate.toFixed(1)}% opt)` : "(no computation)"
            const shapeinfo = `data: ${shapeToString(info.inputShape)} => ${shapeToString(info.outputShape)}`
            const meminfo = `mem: @${info.inputOff} -> @${info.outputOff}`
            const infostr = `Layer: ${l.getClassName()}; ${optinfo} ${shapeinfo} ${meminfo}`
            tmp.unshift(comment(infostr))
            if (opts.verbose)
                console.log(infostr)
            ops.push(tmp)
        } else
            console.log("unsupported layer: ", l.getClassName())
    }

    const flat = flatten(ops)

    const lastInfo = getLayerInfo(m.layers[m.layers.length - 1])
    modelInfo.outputOffset = lastInfo.outputOff

    const cycles = numCycles(flat)
    const cycleinfo = `total cycles: ${cycles} (${(cycles / 84000).toFixed(3)}ms at 84MHz)`
    flat.unshift(comment(cycleinfo))

    if (opts.verbose)
        console.log(cycleinfo)

    const js = `
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
    function f32(v) {
        const arr = new Float32Array(1)
        arr[0] = v
        return arr[0]
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

    const thumb = toThumb(modelInfo, flat)
    const res: CompileResult = {
        execute: (eval(js))(modelInfo.weights),
        js,
        thumb,
        machineCode: null
    }

    return res
}
