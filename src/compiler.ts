///<reference path="pxtpackage.d.ts" />

import * as tf from '@tensorflow/tfjs'
import * as tfi from './tfi'
import * as U from './util'

import * as ir from "./ir"
import { Reg } from "./ir"
import { mkRuntime } from './runtime'

export type Options = ir.Options
export interface CompileResult {
    execute: (inp: ArrayLike<number>) => Float32Array
    js: string
    thumb: string
    machineCode: Uint8Array
    options: Options
    memInfo: string
    timeInfo: string
    stats: {
        total: LayerStats
        layers: LayerStats[]
    }
}

export interface LayerStats {
    // This comes from tf.js, something like flatten_Flatten1.
    name: string
    // If layer has padding, the arena size computation is somewhat unusual
    hasPadding?: boolean
    inputShape: number[]
    outputShape: number[]
    unoptimizedCycles: number
    // time to execute; divide by 64000 to get milliseconds on micro:bit
    optimizedCycles: number
    // Size in flash is the sum of these two. Total size in flash in sum over all of them.
    weightBytes: number
    codeBytes: number
    // Size in RAM for this layer, ie. sum of input and output layer sizes.
    // Total size in RAM for model is close to the *max* (not sum) of these.
    arenaBytes: number
}

interface LayerInfo {
    layer: tf.layers.Layer;
    model: ir.ModelInfo;
    rawInputShape: tf.Shape; // before padding
    inputShape: tf.Shape;
    outputShape: tf.Shape;
    rawInputOff: number; // before padding
    inputOff: number;
    outputOff: number;
    stats: LayerStats;
}

let inited = false
const compilers: SMap<LayerCompileInfo> = {
    Conv2D: { compile: compileConv, computePaddedInputShape: paddingConv },
    Conv1D: { compile: compileConv, computePaddedInputShape: paddingConv },
    DepthwiseConv2D: { compile: compileDepthConv, computePaddedInputShape: paddingConv },
    DepthwiseConv1D: { compile: compileDepthConv, computePaddedInputShape: paddingConv },
    MaxPooling1D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool, needsMInfPadding: true },
    MaxPooling2D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool, needsMInfPadding: true },
    AveragePooling1D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool },
    AveragePooling2D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool },
    Dense: { compile: compileDense },
    Activation: { compile: compileActivation, inPlace: true },
    Softmax: { compile: compileSoftmax, inPlace: true },
    BatchNormalization: { compile: compileBatchNorm, inPlace: true },
    Dropout: {},
    Flatten: {},
    InputLayer: {},
    Reshape: {},
}

const numFPRegs = 32
const numTmpRegs = 6 // ??

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

function validateConfig(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as (tfi.Pooling2DLayerArgs | tfi.BaseConvLayerArgs)

    if (info.model.opts.verbose)
        console.log(info.inputShape, info.outputShape, config)

    const is2D = info.inputShape.length == 4

    if (is2D) {
        if (info.inputShape.length != 4 && info.inputShape.length != 3)
            unsupported("inputShape: " + info.inputShape.length)
        if (config.dataFormat != "channelsLast")
            unsupported("dataFormat: " + config.dataFormat)
    } else {
        if (info.inputShape.length != 3)
            unsupported("inputShape: " + info.inputShape.length)
    }

    if (config.dtype && config.dtype != "float32")
        unsupported("dtype: " + config.dtype)
}

function addActivation(res: ir.Op[], info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.DenseLayerArgs
    const numoutp = shapeElts(info.outputShape)

    if (config.activation == "linear")
        return // linear is identity

    res.push(ir.loadDataAddr(Reg.OutputPtr, info.outputOff))

    if (config.activation == "relu")
        res.push(ir.repeat(numoutp, () => [ir.relu(Reg.OutputPtr)]))
    else if (config.activation == "softmax")
        res.push(ir.fcall("softmax", Reg.OutputPtr, numoutp))
    else
        unsupported("activation: " + config.activation)
}

function addSoftmax(res: ir.Op[], info: LayerInfo) {
    const numoutp = shapeElts(info.outputShape)
    res.push(ir.loadDataAddr(Reg.OutputPtr, info.outputOff))
    res.push(ir.fcall("softmax", Reg.OutputPtr, numoutp))
}

function paddingConv(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.ConvLayerArgs
    const res = info.inputShape.slice()

    for (let i = 1; i <= config.kernelSize.length; ++i) {
        const str = config.strides[i - 1]
        const tmp = info.outputShape[i] * str + config.kernelSize[i - 1] - str
        assert(tmp + str - 1 >= res[i], `${tmp} >= ${res[i]}`)
        if (tmp > res[i])
            res[i] = tmp
    }

    return res
}

function paddingPool(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.Pooling2DLayerArgs
    const res = info.inputShape.slice()

    for (let i = 1; i <= config.poolSize.length; ++i) {
        // TODO this may be wrong if config.poolSize != config.strides
        const tmp = info.outputShape[i] * config.strides[i - 1]
        if (tmp > res[i])
            res[i] = tmp
    }

    return res
}

function compileConv(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.ConvLayerArgs
    const memRegs = numFPRegs >> 1
    const flashRegs = numFPRegs >> 1

    validateConfig(info)

    const is2D = config.kernelSize.length == 2

    const weights0 = info.layer.weights[0].read().arraySync() as any
    const weights = (is2D ? weights0 : [weights0]) as number[][][][]

    const fix1D = (a: number[]) => {
        a = a.slice()
        if (!is2D) a.unshift(1)
        return a
    }

    const [kh, kw] = fix1D(config.kernelSize)
    const [strh, strw] = fix1D(config.strides)
    const [inph, inpw, inpch] = fix1D(info.inputShape.slice(1))
    const [outh, outw, outch] = fix1D(info.outputShape.slice(1))

    // padding not implemented yet
    assert(kh <= inph, "KH2")
    assert(kw <= inpw, "KW2")

    assert(weights.length == kh, "KH")
    assert(weights[0].length == kw, "KW")
    assert(weights[0][0].length == inpch, "CH")
    assert(weights[0][0][0].length == config.filters, "F")
    assert(outch == config.filters, "FF")

    const mi = info.model
    const weightsIdx = ir.weightOffset(mi)
    const bias = config.useBias ? info.layer.weights[1].read().arraySync() as number[] : null

    for (let f = 0; f < config.filters; f++) {
        if (bias)
            ir.addBias(mi, bias[f]) // ???
        for (let y = 0; y < kh; y++) {
            for (let x = 0; x < kw; x++)
                for (let c = 0; c < inpch; ++c)
                    ir.addWeight(mi, weights[y][x][c][f])
            ir.alignWeights(mi)
        }
    }

    const res = [
        ir.loadWeightAddr(Reg.KernelPtr, weightsIdx),
        ir.repeatIdx(config.filters, filt => {
            const res: ir.Op[] = []

            const setOutput = (res: ir.Op[]) => {
                res.push(ir.loadDataAddr(Reg.OutputPtr, info.outputOff))
                res.push(ir.addPtr(Reg.OutputPtr, filt))
            }

            // set bias
            setOutput(res)
            if (config.useBias)
                res.push(ir.load(Reg.S0, 1, Reg.KernelPtr, true))
            else
                res.push(ir.load0(Reg.S0))

            res.push(
                ir.repeat(outw * outh, () => [
                    ir.store(Reg.OutputPtr, Reg.S0, 1, false),
                    ir.addPtr(Reg.OutputPtr, null, config.filters)
                ]))

            res.push(ir.repeatIdx(kh, kline => {
                const res: ir.Op[] = []
                const kernSz = kw * inpch
                let chunk = 0
                for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
                    chunk = kernSz - kernOff
                    if (chunk > flashRegs)
                        chunk = flashRegs
                    res.push(ir.loadWeight(mi, memRegs, chunk))

                    res.push(ir.loadDataAddr(Reg.InputPtr, info.inputOff + kernOff))
                    res.push(ir.addPtr(Reg.InputPtr, kline, inpw * inpch))

                    setOutput(res)

                    const wSkip = strw * inpch
                    const hSkip = strh * inpw * inpch

                    res.push(ir.repeat(outh, () =>
                        [ir.repeat(outw, () => ir.flatten(
                            ir.load(Reg.S0, chunk, Reg.InputPtr, true),
                            ir.addPtr(Reg.InputPtr, null, wSkip - chunk),
                            U.range(chunk + 1).map(i =>
                                [
                                    i < chunk ? ir.vmul(i, i, i + memRegs) : null,
                                    i >= 2 ? ir.vadd(Reg.S0, Reg.S0, i - 1) : null
                                ]),
                            ir.load(Reg.S1, 1, Reg.OutputPtr, false),
                            ir.vadd(Reg.S0, Reg.S0, Reg.S1),
                            ir.store(Reg.OutputPtr, Reg.S0, 1, false),
                            ir.addPtr(Reg.OutputPtr, null, config.filters)
                        )),
                        ir.addPtr(Reg.InputPtr, null, hSkip - outw * wSkip)]))
                }

                res.push(ir.relaxWeights())

                return res
            }))

            res.push(ir.relaxWeights())

            return res
        })]

    addActivation(res, info)

    return res
}

function compileDepthConv(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.DepthwiseConvLayerArgs
    const flashRegOff = 2
    const flashRegs = numFPRegs - flashRegOff

    validateConfig(info)

    const is2D = config.kernelSize.length == 2

    const weights0 = info.layer.weights[0].read().arraySync() as any
    const weights = (is2D ? weights0 : [weights0]) as number[][][][]

    const fix1D = (a: number[]) => {
        a = a.slice()
        if (!is2D) a.unshift(1)
        return a
    }

    const [kh, kw] = fix1D(config.kernelSize)
    const [strh, strw] = fix1D(config.strides)
    const [inph, inpw, inpch] = fix1D(info.inputShape.slice(1))
    const [outh, outw, outch] = fix1D(info.outputShape.slice(1))

    assert(kh <= inph, "KH2")
    assert(kw <= inpw, "KW2")

    assert(weights.length == kh, "KH")
    assert(weights[0].length == kw, "KW")
    assert(weights[0][0].length == inpch, "CH")
    assert(weights[0][0][0].length == config.depthMultiplier, "F")
    assert(outch == config.depthMultiplier * inpch, "FF")

    const mi = info.model
    const weightsIdx = ir.weightOffset(mi)
    const bias = config.useBias ? info.layer.weights[1].read().arraySync() as number[] : null

    if (bias)
        unsupported("bias in depthwise")

    /*
output[i, j, k * channel_multiplier + q] =
    sum_{di, dj} 
           input[strides[1] * i + di,
                 strides[2] * j + dj, k] *
           filter[di, dj, k, q]

for q up to channel_mult
  for k up to num_ch
    num_ch*chmult==outch
    F = load filter k,q
    for i, j
      op = &output[i,j,k*chmult+q]
      *op = 0
      for di, dj
        *op += filter[di,dj,k,q]
           */

    for (let q = 0; q < config.depthMultiplier; q++) {
        if (bias)
            ir.addBias(mi, bias[q]) // ???
        for (let k = 0; k < inpch; ++k) {
            for (let y = 0; y < kh; y++) {
                for (let x = 0; x < kw; x++)
                    ir.addWeight(mi, weights[y][x][k][q])
            }
            ir.alignWeights(mi)
        }
    }

    const res = [
        ir.loadWeightAddr(Reg.KernelPtr, weightsIdx),
        ir.repeatIdx(config.depthMultiplier, q =>
            [ir.repeatIdx(inpch, k => {
                const res: ir.Op[] = []

                const setOutput = (res: ir.Op[]) => {
                    res.push(ir.loadDataAddr(Reg.OutputPtr, info.outputOff))
                    res.push(ir.addPtr(Reg.OutputPtr, k, config.depthMultiplier))
                    res.push(ir.addPtr(Reg.OutputPtr, q))
                }

                setOutput(res)
                if (config.useBias)
                    res.push(ir.load(Reg.S0, 1, Reg.KernelPtr, true))
                else
                    res.push(ir.load0(Reg.S0))

                res.push(
                    ir.repeat(outw * outh, () => [
                        ir.store(Reg.OutputPtr, Reg.S0, 1, false),
                        ir.addPtr(Reg.OutputPtr, null, outch)
                    ]))

                const kernSz = kh * kw

                let skipAcc = 0
                const skipAfter = (kernOff: number) => {
                    const r = (kernOff % kw == kw - 1 ? inpw - kw + 1 : 1) * inpch
                    skipAcc += r
                    return r
                }

                let chunk = 0
                for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
                    chunk = kernSz - kernOff
                    if (chunk > flashRegs)
                        chunk = flashRegs
                    res.push(ir.loadWeight(mi, flashRegOff as Reg, chunk))

                    let skip = 0
                    for (let i = 0; i < kernOff; ++i)
                        skip += skipAfter(i)

                    res.push(
                        ir.loadDataAddr(Reg.InputPtr, info.inputOff + skip),
                        ir.addPtr(Reg.InputPtr, k)
                    )

                    setOutput(res)

                    const wSkip = strw * inpch
                    const hSkip = strh * inpw * inpch

                    res.push(ir.repeat(outh, () =>
                        [ir.repeat(outw, () => {
                            skipAcc = 0
                            const tmp = ir.flatten(
                                ir.load0(Reg.S1),
                                U.range(chunk).map(i => [
                                    ir.load(Reg.S0, 1, Reg.InputPtr, false),
                                    ir.addPtr(Reg.InputPtr, null, skipAfter(kernOff + i)),
                                    ir.vmul(Reg.S0, Reg.S0, i + flashRegOff),
                                    ir.vadd(Reg.S1, Reg.S1, Reg.S0),
                                ]),
                                ir.load(Reg.S0, 1, Reg.OutputPtr, false),
                                ir.vadd(Reg.S0, Reg.S0, Reg.S1),
                                ir.store(Reg.OutputPtr, Reg.S0, 1, false),
                                ir.addPtr(Reg.OutputPtr, null, outch))
                            tmp.push(ir.addPtr(Reg.InputPtr, null, wSkip - skipAcc))
                            return tmp
                        }),
                        ir.addPtr(Reg.InputPtr, null, hSkip - outw * wSkip)]))
                }

                res.push(ir.relaxWeights())

                return res
            })])]

    addActivation(res, info)

    return res
}

function compileMaxPooling(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.Pooling2DLayerArgs

    const is2D = config.poolSize.length == 2

    const isAvg = info.layer.getClassName().startsWith("Average")

    validateConfig(info)

    if (isAvg && config.padding != "valid")
        unsupported("only 'valid' padding supported for AvgPool")

    const fix1D = (a: number[]) => {
        a = a.slice()
        if (!is2D) a.unshift(1)
        return a
    }

    const [kh, kw] = fix1D(config.poolSize)
    const [strh, strw] = fix1D(config.strides)
    const [inph, inpw, numch] = fix1D(info.inputShape.slice(1))
    const [outh, outw, outch] = fix1D(info.outputShape.slice(1))

    // padding not implemented yet
    assert(kh <= inph, "KH2")
    assert(kw <= inpw, "KW2")

    assert(numch == outch, "CH")

    const singleInputPtr = kh - 1 > numTmpRegs
    const lineW = inpw * numch

    return [
        ir.repeatIdx(numch, filt => {
            const res = [
                ir.loadDataAddr(Reg.OutputPtr, info.outputOff),
                ir.addPtr(Reg.OutputPtr, filt),
                ir.loadDataAddr(Reg.InputPtr, info.inputOff),
                ir.addPtr(Reg.InputPtr, filt),
            ]

            const ptrRegs = singleInputPtr ? [] : U.range(kh - 1).map(i => Reg.Tmp0 + i)
            ptrRegs.unshift(Reg.InputPtr)

            if (!singleInputPtr)
                for (let i = 1; i < kh; ++i) {
                    const op = ir.addPtr(ptrRegs[i], null, lineW * i, Reg.InputPtr)
                    op.isDef = true
                    res.push(op)
                }

            res.push(
                ir.repeat(outh, () => ir.flatten(
                    ir.repeat(outw, () => {
                        const res: ir.Op[] = []
                        for (let i = 0; i < kh; ++i) {
                            let preg = ptrRegs[i]
                            if (singleInputPtr) {
                                preg = Reg.Tmp0
                                const op = ir.addPtr(preg, null, lineW * i, Reg.InputPtr)
                                if (i == 0)
                                    op.isDef = true
                                res.push(op)
                            }
                            for (let j = 0; j < kw; ++j) {
                                const reg = i == 0 && j == 0 ? Reg.S0 : Reg.S1
                                res.push(
                                    ir.load(reg, 1, preg, true),
                                    ir.addPtr(preg, null, numch - 1)
                                )
                                if (reg != Reg.S0) {
                                    if (isAvg)
                                        res.push(ir.vadd(Reg.S0, Reg.S0, reg))
                                    else
                                        res.push(ir.vmax(Reg.S0, Reg.S0, reg))
                                }
                            }
                            if (!singleInputPtr)
                                res.push(
                                    ir.addPtr(preg, null, (strw - kw) * numch)
                                )
                        }
                        if (isAvg)
                            res.push(
                                ir.loadLit(Reg.S1, 1 / (kw * kh)),
                                ir.vmul(Reg.S0, Reg.S0, Reg.S1)
                            )
                        res.push(
                            ir.store(Reg.OutputPtr, Reg.S0, 1, true),
                            ir.addPtr(Reg.OutputPtr, null, numch - 1)
                        )
                        if (singleInputPtr)
                            res.push(
                                ir.addPtr(Reg.InputPtr, null, strw * numch)
                            )
                        return res
                    }),
                    ptrRegs.map(r => ir.addPtr(r, null, strh * lineW - outw * strw * numch)))))

            return res
        })
    ]
}

function compileBatchNorm(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.BatchNormalizationLayerArgs

    const flashRegs = numFPRegs - 2
    const flashReg0 = Reg.S0 + 2

    let inpShape = info.inputShape
    if (inpShape.length == 2)
        inpShape = [inpShape[0], 1, 1, inpShape[1]]

    if (inpShape.length != 4)
        unsupported("inputShape: " + inpShape.length)

    if (config.dtype && config.dtype != "float32")
        unsupported("dtype: " + config.dtype)

    const [_null, outh, outw, numch] = inpShape

    function readVar(name: string) {
        const r = info.layer.weights.find(w => w.originalName.endsWith("/" + name)).read().arraySync() as number[]
        assert(r.length == numch)
        return r
    }

    const gamma = readVar("gamma")
    const beta = readVar("beta")
    const movingMean = readVar("moving_mean")
    const movingVar = readVar("moving_variance")

    // gamma * (batch - moving_mean) / sqrt(moving_var+epsilon) + beta
    // Q = 1/sqrt(moving_var+epsilon)
    // Q * gamma * (batch - moving_mean) + beta
    // Q * gamma * batch - Q * gamma * moving_mean + beta

    const mi = info.model
    const weightsIdx = ir.weightOffset(mi)

    for (let i = 0; i < numch; i++) {
        const q = 1 / Math.sqrt(movingVar[i] + config.epsilon)
        const mult = q * gamma[i]
        const offset = - q * gamma[i] * movingMean[i] + beta[i]
        // console.log({ e: config.epsilon, mv: movingVar[i], q, mult, offset })
        ir.addWeight(mi, mult)
        ir.addWeight(mi, offset)
    }

    assert(info.inputOff == info.outputOff)

    const res = [
        ir.loadWeightAddr(Reg.KernelPtr, weightsIdx),
    ]

    const kernSz = numch * 2
    let chunk = 0
    for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
        assert((kernOff & 1) == 0)
        chunk = kernSz - kernOff
        if (chunk > flashRegs)
            chunk = flashRegs

        res.push(
            ir.loadWeight(mi, flashReg0, chunk),
            ir.loadDataAddr(Reg.OutputPtr, info.outputOff + (kernOff >> 1)),
            ir.repeat(outh * outw, () =>
                ir.flatten(
                    U.range(chunk >> 1).map(i =>
                        [
                            ir.load(Reg.S0, 1, Reg.OutputPtr, false),
                            ir.vmul(Reg.S0, Reg.S0, (i * 2) + flashReg0),
                            ir.vadd(Reg.S0, Reg.S0, (i * 2) + 1 + flashReg0),
                            ir.store(Reg.OutputPtr, Reg.S0, 1, true),
                        ]),
                    ir.addPtr(Reg.OutputPtr, null, numch - (chunk >> 1))
                )))
    }

    return res
}

function compileDense(info: LayerInfo) {
    const config = info.layer.getConfig() as unknown as tfi.DenseLayerArgs

    const maxChunk = (numFPRegs >> 1) - 2
    const memReg0 = Reg.S1
    const flashReg0 = memReg0 + maxChunk

    //if (info.model.opts.verbose)
    //    console.log(info.inputShape, info.outputShape, config)

    if (info.inputShape.length != 2)
        unsupported("inputShape: " + info.inputShape.length)

    if (config.dtype && config.dtype != "float32")
        unsupported("dtype: " + config.dtype)

    const weights = info.layer.weights[0].read().arraySync() as number[][]
    //console.log(weights)

    const inpsize = info.inputShape[1]

    assert(weights.length == inpsize, "IH")
    assert(weights[0].length == config.units, "UN")

    const mi = info.model
    const weightsIdx = ir.weightOffset(mi)
    const bias = config.useBias ? info.layer.weights[1].read().arraySync() as number[] : null
    //console.log(bias)

    for (let f = 0; f < config.units; f++) {
        if (bias)
            ir.addBias(mi, bias[f])
        for (let i = 0; i < inpsize; ++i)
            ir.addWeight(mi, weights[i][f])
        ir.alignWeights(mi)
    }

    const res = [
        ir.loadWeightAddr(Reg.KernelPtr, weightsIdx),
        ir.loadDataAddr(Reg.OutputPtr, info.outputOff),
        ir.repeat(config.units, () => {
            const res: ir.Op[] = []

            // set bias
            if (config.useBias)
                res.push(ir.load(Reg.S0, 1, Reg.KernelPtr, true))
            else
                res.push(ir.load0(Reg.S0))

            res.push(ir.loadDataAddr(Reg.InputPtr, info.inputOff))

            const addChunk = (len: number) => ir.flatten(
                ir.load(memReg0, len, Reg.InputPtr, true),
                ir.loadWeight(mi, flashReg0, len),
                U.range(len + 1).map(i => [
                    i < len ? ir.vmul(memReg0 + i, memReg0 + i, flashReg0 + i) : null,
                    i >= 1 ? ir.vadd(Reg.S0, Reg.S0, memReg0 + i - 1) : null
                ])
            )

            const numRep = (inpsize / maxChunk) | 0
            if (numRep > 0)
                res.push(ir.repeat(numRep, () => addChunk(maxChunk)))
            const left = inpsize - numRep * maxChunk
            if (left > 0)
                U.pushRange(res, addChunk(left))

            res.push(ir.store(Reg.OutputPtr, Reg.S0, 1, true))
            res.push(ir.relaxWeights())

            return res
        })]

    addActivation(res, info)

    return res
}

function compileActivation(info: LayerInfo) {
    const res: ir.Op[] = []
    addActivation(res, info)
    return res
}

function compileSoftmax(info: LayerInfo) {
    const res: ir.Op[] = []
    addSoftmax(res, info)
    return res
}

function noop(info: LayerInfo): ir.Op[] {
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
    compile?: (info: LayerInfo) => ir.Op[]
    inPlace?: boolean
    testable?: boolean
    needsMInfPadding?: boolean
    computePaddedInputShape?: (info: LayerInfo) => number[]
}

function fixupCompileInfo(info: LayerCompileInfo) {
    if (info.testable === undefined)
        info.testable = !!info.compile
    if (!info.compile) {
        if (info.inPlace === undefined)
            info.inPlace = true
        info.compile = noop
    }
    if (!info.computePaddedInputShape)
        info.computePaddedInputShape = info => info.inputShape.slice()
}

function isInPlace(layer: tf.layers.Layer) {
    return !!compilers[layer.getClassName()]?.inPlace
}

function isTestable(layer: tf.layers.Layer) {
    return !!compilers[layer.getClassName()]?.testable
}

function needMInfPadding(layer: tf.layers.Layer) {
    return !!compilers[layer.getClassName()]?.needsMInfPadding
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
        weightPtr: 0,
        weightBuffer: new Uint8Array(128),
        weightAsm: "",
        inputShape,
        outputShape: null,
        outputOffset: -1,
        arenaSize: -1,
        minArenaSize: -1,
        opts,
        stats: ""
    }

    let maxSize = [shapeElts(inputShape), 0]
    let currIdx = 0
    let prev: LayerInfo
    let totalMax = maxSize[0]
    const recordMax = (n: number) => totalMax = Math.max(n, totalMax)

    for (const l of m.layers) {
        const info = getLayerInfo(l)
        info.model = modelInfo
        if (prev) {
            info.inputShape = prev.outputShape
        } else {
            info.inputShape = inputShape
        }
        info.outputShape = l.computeOutputShape(info.inputShape) as tf.Shape
        const comp = compilers[l.getClassName()]
        const paddedShape = comp ? comp.computePaddedInputShape(info) : info.inputShape.slice()
        info.inputOff = currIdx

        info.rawInputShape = info.inputShape.slice()

        const paddedElts = shapeElts(paddedShape)
        const needsPadding = shapeElts(info.inputShape) != paddedElts
        if (needsPadding) {
            currIdx = currIdx == 0 ? 1 : 0
            info.rawInputOff = info.inputOff
            info.inputOff = currIdx
            info.inputShape = paddedShape
            if (paddedElts > maxSize[currIdx])
                maxSize[currIdx] = paddedElts
            recordMax(paddedElts + shapeElts(info.rawInputShape))
        } else {
            info.rawInputOff = null
        }

        const elts = shapeElts(info.outputShape)
        if (isInPlace(l)) {
            recordMax(shapeElts(info.inputShape))
            recordMax(shapeElts(info.outputShape))
        } else {
            recordMax(shapeElts(info.inputShape) + shapeElts(info.outputShape))
            currIdx = currIdx == 0 ? 1 : 0
        }
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
        if (info.rawInputOff) info.rawInputOff = midOff
        info.stats = { name: l.name } as LayerStats
    }

    const arenaSize = maxSize[0] + maxSize[1]
    modelInfo.arenaSize = arenaSize
    modelInfo.minArenaSize = totalMax

    if (arenaSize > totalMax * 1.2) {
        // TODO
        console.log("possible arena shrink with wiser allocation: " + (arenaSize / totalMax).toFixed(3) + "x")
    }

    return modelInfo
}

function compilePadding(info: LayerInfo) {
    const res: ir.Op[] = [
        ir.comment("padding")
    ]

    if (info.rawInputOff == null)
        return res

    const is2D = info.rawInputShape.length >= 4
    const fix1D = (a: number[]) => {
        a = a.slice()
        a.shift() // shift initial null
        if (!is2D) a.unshift(1)
        return a
    }

    const [inpy, inpx, numch] = fix1D(info.rawInputShape)
    const [outy, outx, outch] = fix1D(info.inputShape)
    assert(numch == outch)

    const padx = outx - inpx
    const x0 = padx >> 1
    const x1 = padx - x0

    const pady = outy - inpy
    const y0 = pady >> 1
    const y1 = pady - y0

    const numZero = numFPRegs >> 1
    const numData = numFPRegs - numZero
    const dataReg = Reg.S0 + numZero

    res.push(needMInfPadding(info.layer) ? ir.loadMInf(Reg.S0) : ir.load0(Reg.S0))
    // this is slightly cheaper than loading zeros (or -Inf) from memory
    for (let i = 1; i < numZero; ++i)
        res.push(ir.vadd(Reg.S0 + i, Reg.S0, Reg.S0))

    res.push(ir.loadDataAddr(Reg.InputPtr, info.rawInputOff))
    res.push(ir.loadDataAddr(Reg.OutputPtr, info.inputOff))

    const topPad = y0 * outx + x0
    const linePad = x1 + x0
    const bottomPad = x1 + y1 * outx

    res.push(...setZero(topPad))
    res.push(ir.repeat(inpy - 1, () => ir.flatten(
        copyOver(inpx),
        setZero(linePad)
    )))
    res.push(...copyOver(inpx))
    res.push(...setZero(bottomPad))

    return res

    function setZero(n: number) {
        const res: ir.Op[] = []
        n *= numch
        const leftover = n % numZero
        const reps = (n - leftover) / numZero
        if (reps)
            res.push(ir.repeat(reps, () => [
                ir.store(Reg.OutputPtr, Reg.S0, numZero, true)
            ]))
        if (leftover)
            res.push(ir.store(Reg.OutputPtr, Reg.S0, leftover, true))
        return res
    }

    function copyOver(n: number) {
        const res: ir.Op[] = []
        n *= numch
        const leftover = n % numData
        const reps = (n - leftover) / numData
        if (reps)
            res.push(ir.repeat(reps, () => [
                ir.load(dataReg, numData, Reg.InputPtr, true),
                ir.store(Reg.OutputPtr, dataReg, numData, true)
            ]))
        if (leftover) {
            res.push(
                ir.load(dataReg, leftover, Reg.InputPtr, true),
                ir.store(Reg.OutputPtr, dataReg, leftover, true))
        }
        return res
    }
}

function optimizeWithComment(opts: Options, opcodes: ir.Op[], stats: LayerStats) {
    if (opts.float16weights)
        opcodes = ir.fixupAndMarkF16(opcodes)
    const c0 = ir.numCycles(opcodes)
    if (opts.optimize)
        opcodes = ir.optimize(opcodes)
    const c1 = ir.numCycles(opcodes)
    stats.unoptimizedCycles += c0
    stats.optimizedCycles += c1
    const optRate = 100 * (c0 - c1) / c0
    const optinfo = c0 ? `${c1} cycles (${optRate.toFixed(1)}% opt)` : "(no computation)"
    if (c0)
        opcodes.unshift(ir.comment(optinfo))
    return { opcodes, optinfo }
}

function statsShape(shape: tf.Shape): number[] {
    return shape.filter(x => x != null)
}

export function compileModelCore(m: tf.LayersModel, opts: ir.Options) {
    const modelInfo = assignLayerInfos(m, opts)

    if (opts.optimize === undefined)
        opts.optimize = true

    const ops: ir.Op[][] = []
    const layerStats: LayerStats[] = []

    const layer0 = getLayerInfo(m.layers[0])
    const layerN = getLayerInfo(m.layers[m.layers.length - 1])
    const totalStats: LayerStats = {
        name: "TOTAL",
        inputShape: statsShape(layer0.rawInputShape || layer0.inputShape),
        outputShape: statsShape(layerN.outputShape),
        arenaBytes: 0,
        codeBytes: 0,
        weightBytes: 0,
        unoptimizedCycles: 0,
        optimizedCycles: 0
    }

    for (const l of m.layers) {
        const info = getLayerInfo(l)

        info.stats.unoptimizedCycles = 0
        info.stats.optimizedCycles = 0
        info.stats.arenaBytes = 0

        info.stats.inputShape = statsShape(info.rawInputShape || info.inputShape)
        info.stats.outputShape = statsShape(info.outputShape)

        const statsIdx = layerStats.length
        layerStats.push(info.stats)
        ops.push([ir.label("begin_" + statsIdx)])

        if (info.rawInputOff != null) {
            const tmp = optimizeWithComment(opts, compilePadding(info), info.stats)
            ops.push(tmp.opcodes)
            info.stats.arenaBytes = (shapeElts(info.rawInputShape) + shapeElts(info.inputShape)) << 2
            info.stats.hasPadding = true
        }

        const cinfo = compilers[l.getClassName()]
        if (cinfo) {
            const size0 = ir.weightOffset(modelInfo)
            const tmp = optimizeWithComment(opts, cinfo.compile(info), info.stats)
            info.stats.weightBytes = (ir.weightOffset(modelInfo) - size0) << 2
            const shapeinfo = `data: ${shapeToString(info.inputShape)}@${info.inputOff} => ${shapeToString(info.outputShape)}@${info.outputOff}`
            const infostr = `Layer: ${l.getClassName()}; ${shapeinfo}`
            tmp.opcodes.unshift(ir.comment(infostr))
            if (opts.verbose)
                console.log(infostr + " " + tmp.optinfo)
            ops.push(tmp.opcodes)
        } else {
            console.log(l.getConfig())
            unsupported("layer: " + l.getClassName())
        }

        if (info.stats.unoptimizedCycles)
            info.stats.arenaBytes = Math.max(info.stats.arenaBytes, (shapeElts(info.inputShape) + shapeElts(info.outputShape)) << 2)

        totalStats.unoptimizedCycles += info.stats.unoptimizedCycles
        ops.push([ir.label("end_" + statsIdx)])
    }

    let flat = ir.flatten(ops)

    const lastInfo = getLayerInfo(m.layers[m.layers.length - 1])
    modelInfo.outputOffset = lastInfo.outputOff

    const mhz = 64
    const cycles = ir.numCycles(flat)
    const cycleinfo = `total cycles: ${cycles} (${(cycles / (mhz * 1000)).toFixed(3)}ms at ${mhz}MHz)`
    modelInfo.stats = cycleinfo

    totalStats.optimizedCycles = cycles

    if (opts.verbose)
        console.log(modelInfo.stats)

    modelInfo.weightBuffer = modelInfo.weightBuffer.slice(0, modelInfo.weightPtr)

    const inputSize = shapeElts(getLayerInfo(m.layers[0]).rawInputShape)

    let js = `
((weights, mkRuntime) => {
    "use strict";
    const weightOff = ${modelInfo.arenaSize}
    const dataOff = 0
    const mem = new Float32Array(weightOff + ${ir.weightOffset(modelInfo)})
    mem.fill(1000.2342)
    new Uint8Array(mem.buffer).set(weights, weightOff << 2)
    const memU32 = new Uint32Array(mem.buffer)
    const rt = mkRuntime(mem)
    const { softmax, f32 } = rt
    return (inputs => {
        if (inputs.length != ${inputSize})
            throw new Error("invalid input size; expected ${inputSize}, got " + inputs.length)
        mem.set(inputs, dataOff)
        let input, output, kernel
        let ${U.range(numTmpRegs).map(r => "tmp" + r).join(", ")}
        let ${U.range(numFPRegs).map(r => "s" + r).join(", ")}

${ir.toJSs(modelInfo, flat)}
        
        return mem.slice(${lastInfo.outputOff}, ${lastInfo.outputOff + shapeElts(lastInfo.outputShape)})
    })
})
`

    const execute = ((0, eval)(js))(modelInfo.weightBuffer, mkRuntime)

    js = `${ir.stringifyComment(modelInfo.stats)}\nconst modelFromWeights = ${js};\n`

    const w = Array.from(new Uint32Array(modelInfo.weightBuffer.buffer))
    js += `const weights = new Uint8Array(new Uint32Array(${JSON.stringify(w)}).buffer);\n`
    js += `const modelFromRuntime = mkR => modelFromWeights(weights, mkR);\n`
    js += `return { weights, modelFromRuntime, modelFromWeights, inputSize: ${inputSize} };\n`

    let thumb = ""
    if (opts.includeTest && opts.testOutput && opts.testOutputFromJS) {
        // If requested, embed the output from JS code as reference in Thumb code
        // This is important for float16 - the JS and Thumb should be equivalent
        // but the TF.JS may be further out, as it only does float32
        const prev = opts.testOutput
        opts.testOutput = execute(opts.testInput)
        thumb = ir.toThumb(modelInfo, flat)
        opts.testOutput = prev
    } else {
        thumb = ir.toThumb(modelInfo, flat)
    }

    const res: CompileResult = {
        execute,
        js,
        thumb,
        machineCode: null,
        options: opts,
        memInfo: null,
        timeInfo: modelInfo.stats,
        stats: {
            total: totalStats,
            layers: layerStats,
        }
    }

    return res
}

async function serializeModel(m: tf.LayersModel) {
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

    return mod
}

/**
 * Split model into single-layer models for testing.
 */
export async function* partialModels(m: tf.LayersModel, opts: Options) {
    const mod = await serializeModel(m)
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
        lcfg.batch_input_shape = info.rawInputShape
        cfg.layers = [layerJson]
        const copy = await tf.loadLayersModel({ load: () => Promise.resolve(mod) })
        console.log(`testing ${layer.getClassName()}: ${shapeToString(info.rawInputShape)} => ${shapeToString(info.outputShape)}...`)
        yield copy
        layerJson.config.batch_input_shape = info.rawInputShape
        // also test it without activation
        if (lcfg.activation && lcfg.activation != "linear") {
            lcfg.activation = null
            const withoutAct = await tf.loadLayersModel({ load: () => Promise.resolve(mod) })
            console.log(`also with no activation...`)
            yield withoutAct
        }
    }
}

export async function* prefixModels(m: tf.LayersModel, opts: Options) {
    const mod = await serializeModel(m)

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
        cfg.layers = layersJson.slice(0, i + 1)
        const copy = await tf.loadLayersModel({ load: () => Promise.resolve(mod) }, { strict: false })
        console.log(`testing prefix ${layer.getClassName()} => ${shapeToString(info.outputShape)}...`)
        yield copy
    }
}
