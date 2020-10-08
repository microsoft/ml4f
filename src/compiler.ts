///<reference path="pxtpackage.d.ts" />

import * as tf from '@tensorflow/tfjs'
import * as tfi from './tfi'
import * as U from './util'

import * as ir from "./ir"

export type Options = ir.Options
export interface CompileResult {
    execute: (inp: ArrayLike<number>) => Float32Array
    js: string
    thumb: string
    machineCode: Uint8Array
    options: Options
}

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

function addActivation(res: ir.Op[], layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as tfi.DenseLayerArgs
    const info = getLayerInfo(layer)
    const numoutp = shapeElts(info.outputShape)

    if (config.activation == "linear")
        return // linear is identity

    res.push(ir.loadDataAddr(ir.Reg.OutputPtr, info.outputOff))

    if (config.activation == "relu")
        res.push(ir.repeat(numoutp, () => [ir.relu(ir.Reg.OutputPtr)]))
    else if (config.activation == "softmax")
        res.push(ir.fcall("softmax", ir.Reg.OutputPtr, numoutp))
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

    const outh = info.outputShape[1]
    const outw = info.outputShape[2]
    const outch = info.outputShape[3]

    const paddingX = outw - (inpw - kw + 1)
    const paddingY = outh - (inph - kh + 1)
    assert(paddingX >= 0, "x0")
    assert(paddingY >= 0, "y0")
    const paddingX0 = paddingX >> 1
    const paddingX1 = paddingX - paddingX0
    const paddingY0 = paddingY >> 1
    const paddingY1 = paddingY - paddingY0

    // padding not implemented yet
    assert(kh <= inph, "KH2")
    assert(kw <= inpw, "KW2")

    assert(weights.length == kh, "KH")
    assert(weights[0].length == kw, "KW")
    assert(weights[0][0].length == inpch, "CH")
    assert(weights[0][0][0].length == config.filters, "F")
    assert(outch == config.filters, "FF")

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
        ir.loadWeightAddr(ir.Reg.KernelPtr, weightsIdx),
        ir.repeatIdx(config.filters, filt => {
            const res: ir.Op[] = []

            const setOutput = (res: ir.Op[]) => {
                res.push(ir.loadDataAddr(ir.Reg.OutputPtr, info.outputOff))
                res.push(ir.addPtr(ir.Reg.OutputPtr, filt))
            }

            // set bias
            setOutput(res)
            if (config.useBias)
                res.push(ir.load(ir.Reg.S0, 1, ir.Reg.KernelPtr, true))
            else
                res.push(ir.load0(ir.Reg.S0))

            res.push(
                ir.repeat(outw * outh, () => [
                    ir.store(ir.Reg.OutputPtr, 0, 1, false),
                    ir.addPtr(ir.Reg.OutputPtr, null, config.filters)
                ]))

            res.push(ir.repeatIdx(kh, kline => {
                const res: ir.Op[] = []
                const kernSz = kw * inpch
                let chunk = 0
                for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
                    chunk = kernSz - kernOff
                    if (chunk > flashRegs)
                        chunk = flashRegs
                    res.push(ir.load(memRegs, chunk, ir.Reg.KernelPtr, true))

                    res.push(ir.loadDataAddr(ir.Reg.InputPtr, info.inputOff + kernOff))
                    res.push(ir.addPtr(ir.Reg.InputPtr, kline, inpw * inpch))

                    setOutput(res)

                    const wSkip = strw * inpch
                    const hSkip = strh * inpw * inpch

                    res.push(ir.repeat(outh, () =>
                        [ir.repeat(outw, () => ir.flatten(
                            ir.load(ir.Reg.S0, chunk, ir.Reg.InputPtr, true),
                            ir.addPtr(ir.Reg.InputPtr, null, wSkip - chunk),
                            U.range(chunk + 1).map(i =>
                                [
                                    i < chunk ? ir.vmul(i, i, i + memRegs) : null,
                                    i >= 2 ? ir.vadd(ir.Reg.S0, ir.Reg.S0, i - 1) : null
                                ]),
                            ir.load(ir.Reg.S1, 1, ir.Reg.OutputPtr, false),
                            ir.vadd(ir.Reg.S0, ir.Reg.S0, ir.Reg.S1),
                            ir.store(ir.Reg.OutputPtr, ir.Reg.S0, 1, false),
                            ir.addPtr(ir.Reg.OutputPtr, null, config.filters)
                        )),
                        ir.addPtr(ir.Reg.InputPtr, null, hSkip - outw * wSkip)]))
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
        ir.repeatIdx(numch, filt => {
            const res = [
                ir.loadDataAddr(ir.Reg.OutputPtr, info.outputOff),
                ir.addPtr(ir.Reg.OutputPtr, filt),
                ir.loadDataAddr(ir.Reg.InputPtr, info.inputOff),
                ir.addPtr(ir.Reg.InputPtr, filt),
            ]

            const ptrRegs = U.range(kh - 1).map(i => ir.Reg.Tmp0 + i)
            ptrRegs.unshift(ir.Reg.InputPtr)

            for (let i = 1; i < kh; ++i) {
                const op = ir.addPtr(ptrRegs[i], null, lineW * i, ir.Reg.InputPtr)
                op.isDef = true
                res.push(op)
            }

            res.push(
                ir.repeat(info.outputShape[1], () => ir.flatten(
                    ir.repeat(info.outputShape[2], () => {
                        const res: ir.Op[] = []
                        for (let i = 0; i < kh; ++i) {
                            for (let j = 0; j < kw; ++j) {
                                const reg = i == 0 && j == 0 ? ir.Reg.S0 : ir.Reg.S1
                                res.push(
                                    ir.load(reg, 1, ptrRegs[i], true),
                                    ir.addPtr(ptrRegs[i], null, numch - 1)
                                )
                                if (reg != ir.Reg.S0)
                                    res.push(ir.vmax(ir.Reg.S0, ir.Reg.S0, reg))
                            }
                            res.push(
                                ir.addPtr(ptrRegs[i], null, (strw - kw) * numch)
                            )
                        }
                        res.push(
                            ir.store(ir.Reg.OutputPtr, ir.Reg.S0, 1, true),
                            ir.addPtr(ir.Reg.OutputPtr, null, numch - 1)
                        )
                        return res
                    }),
                    ptrRegs.map(r => ir.addPtr(r, null, strh * lineW - info.outputShape[2] * strw * numch)))))

            return res
        })
    ]
}

function compileDense(layer: tf.layers.Layer) {
    const config = layer.getConfig() as unknown as tfi.DenseLayerArgs
    const info = getLayerInfo(layer)

    const maxChunk = (numFPRegs >> 1) - 2
    const memReg0 = ir.Reg.S1
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
        ir.loadWeightAddr(ir.Reg.KernelPtr, weightsIdx),
        ir.loadDataAddr(ir.Reg.OutputPtr, info.outputOff),
        ir.repeat(config.units, () => {
            const res: ir.Op[] = []

            // set bias
            if (config.useBias)
                res.push(ir.load(ir.Reg.S0, 1, ir.Reg.KernelPtr, true))
            else
                res.push(ir.load0(ir.Reg.S0))

            res.push(ir.loadDataAddr(ir.Reg.InputPtr, info.inputOff))

            const addChunk = (len: number) => ir.flatten(
                ir.load(memReg0, len, ir.Reg.InputPtr, true),
                ir.load(flashReg0, len, ir.Reg.KernelPtr, true),
                U.range(len + 1).map(i => [
                    i < len ? ir.vmul(memReg0 + i, memReg0 + i, flashReg0 + i) : null,
                    i >= 1 ? ir.vadd(ir.Reg.S0, ir.Reg.S0, memReg0 + i - 1) : null
                ])
            )

            const numRep = (inpsize / maxChunk) | 0
            if (numRep > 0)
                res.push(ir.repeat(numRep, () => addChunk(maxChunk)))
            const left = inpsize - numRep * maxChunk
            if (left > 0)
                U.pushRange(res, addChunk(left))

            res.push(ir.store(ir.Reg.OutputPtr, ir.Reg.S0, 1, true))

            return res
        })]

    addActivation(res, layer)

    return res
}

function noop(l: tf.layers.Layer): ir.Op[] {
    return []
}

export function shapeElts(shape: tf.Shape) {
    let r = 1
    for (const s of shape)
        if (s != null)
            r *= s
    return r
}

interface LayerCompileInfo {
    compile?: (l: tf.layers.Layer) => ir.Op[]
    inPlace?: boolean
    testable?: boolean
}

function fixupCompileInfo(info: LayerCompileInfo) {
    if (info.testable === undefined)
        info.testable = !!info.compile
    if (!info.compile) {
        if (info.inPlace === undefined)
            info.inPlace = true
        info.compile = noop
    }
}

let inited = false
const compilers: SMap<LayerCompileInfo> = {
    Conv2D: { compile: compileConv2D },
    MaxPooling2D: { compile: compileMaxPooling2D },
    Dense: { compile: compileDense },
    Dropout: {},
    Flatten: {},
    InputLayer: {},
}

function isInPlace(layer: tf.layers.Layer) {
    return !!compilers[layer.getClassName()]?.inPlace
}

function isTestable(layer: tf.layers.Layer) {
    return !!compilers[layer.getClassName()]?.testable
}

function shapeToString(shape: tf.Shape) {
    return `[${shape.filter(x => x != null).join(",")}]`
}

export function assignLayerInfos(m: tf.LayersModel, opts: ir.Options) {
    if (!inited) {
        inited = true
        Object.values(compilers).forEach(fixupCompileInfo)
    }

    ir.reset()

    if (opts.verbose)
        m.summary()

    const inputShape = m.layers[0].batchInputShape

    const modelInfo: ir.ModelInfo = {
        weights: [],
        inputShape,
        outputShape: null,
        outputOffset: -1,
        arenaSize: -1,
        opts,
        stats: ""
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

    return modelInfo
}

export function compileModelCore(m: tf.LayersModel, opts: ir.Options) {
    const modelInfo = assignLayerInfos(m, opts)

    const ops: ir.Op[][] = []

    for (const l of m.layers) {
        const info = getLayerInfo(l)

        const cinfo = compilers[l.getClassName()]
        if (cinfo) {
            let tmp = cinfo.compile(l)
            const c0 = ir.numCycles(tmp)
            tmp = ir.optimize(tmp)
            const c1 = ir.numCycles(tmp)
            const optRate = 100 * (c0 - c1) / c0
            const optinfo = c0 ? `${c1} cycles (${optRate.toFixed(1)}% opt)` : "(no computation)"
            const shapeinfo = `data: ${shapeToString(info.inputShape)} => ${shapeToString(info.outputShape)}`
            const meminfo = `mem: @${info.inputOff} -> @${info.outputOff}`
            const infostr = `Layer: ${l.getClassName()}; ${optinfo} ${shapeinfo} ${meminfo}`
            tmp.unshift(ir.comment(infostr))
            if (opts.verbose)
                console.log(infostr)
            ops.push(tmp)
        } else
            console.log("unsupported layer: ", l.getClassName())
    }

    const flat = ir.flatten(ops)

    const lastInfo = getLayerInfo(m.layers[m.layers.length - 1])
    modelInfo.outputOffset = lastInfo.outputOff

    const cycles = ir.numCycles(flat)
    const cycleinfo = `total cycles: ${cycles} (${(cycles / 84000).toFixed(3)}ms at 84MHz)`
    modelInfo.stats = cycleinfo

    if (opts.verbose)
        console.log(modelInfo.stats)

    const js = `
${ir.stringifyComment(modelInfo.stats)}
(weights => {
    "use strict";
    const weightOff = ${modelInfo.arenaSize}
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

${ir.toJSs(modelInfo, flat)}
        
        return mem.slice(${lastInfo.outputOff}, ${lastInfo.outputOff + shapeElts(lastInfo.outputShape)})
    })
})
`

    const thumb = ir.toThumb(modelInfo, flat)
    const res: CompileResult = {
        execute: (eval(js))(modelInfo.weights),
        js,
        thumb,
        machineCode: null,
        options: opts
    }

    return res
}

/**
 * Split model into single-layer models for testing.
 */
export async function* partialModels(m: tf.LayersModel, opts: Options) {
    let mod: tf.io.ModelArtifacts
    await m.save({
        save: m => {
            mod = m
            const res: tf.io.SaveResult = {
                modelArtifactsInfo: {
                    dateSaved: new Date(),
                    modelTopologyType: "JSON"
                }
            }
            return Promise.resolve(res)
        }
    })

    delete mod.weightData
    delete mod.weightSpecs
    const cfg = (mod.modelTopology as any)?.config
    const layersJson: any[] = cfg?.layers || []

    for (let i = 0; i < m.layers.length; ++i) {
        const layerJson = layersJson[i]
        const layer = m.layers[i]
        const info = getLayerInfo(layer)
        if (layerJson?.class_name != layer.getClassName())
            throw new Error("invalid serialization")
        if (!isTestable(layer))
            continue
        const lcfg = layerJson.config
        lcfg.batch_input_shape = info.inputShape
        cfg.layers = [layerJson]
        const copy = await tf.loadLayersModel({ load: () => Promise.resolve(mod) })
        console.log(`testing ${layer.getClassName()}: ${shapeToString(info.inputShape)} => ${shapeToString(info.outputShape)}...`)
        yield copy
        layerJson.config.batch_input_shape = info.inputShape
        // also test it without activation
        if (lcfg.activation) {
            lcfg.activation = null
            const withoutAct = await tf.loadLayersModel({ load: () => Promise.resolve(mod) })
            console.log(`also with no activation...`)
            yield withoutAct
        }
    }
}
