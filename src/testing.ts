import * as tf from '@tensorflow/tfjs'
import * as U from './util'
import { CompileResult, Options } from './compiler';
import { compileAndTest, compileModelAndFullValidate, setRandomWeights } from './driver';
import { testFloatConv } from './float16';
import { mkRuntime } from './runtime';


function randomModel() {
    const model = tf.sequential();

    const inputShape = [U.randomInclusive(1, 100), U.randomInclusive(1, 50), U.randomInclusive(1, 32)]
    const kernelSize = [U.randomInclusive(1, 5), U.randomInclusive(1, 5)]
    const strides = [U.randomInclusive(1, 3), U.randomInclusive(1, 3)]
    const filters = U.randomInclusive(1, 5)

    kernelSize[0] = Math.min(kernelSize[0], inputShape[0])
    kernelSize[1] = Math.min(kernelSize[1], inputShape[1])

    const desc = `Conv2D ${inputShape} X ${kernelSize} @${strides} -> ${filters}`

    model.add(tf.layers.conv2d({
        inputShape,
        kernelSize,
        filters,
        strides,
        padding: 'valid',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // make sure weights are deterministic
    for (const l of model.layers)
        setRandomWeights(l)

    model.name = desc
    return model
}

function logThumb(cres: CompileResult) {
    let str = cres.thumb
    function hex2(n: number) {
        return ("0" + n.toString(16)).slice(-2)
    }
    str += "// BUF: "
    for (const v of cres.machineCode) str += hex2(v)
    console.log(str)
    console.log(cres.memInfo)
    console.log(cres.timeInfo)
}


export async function runBrowser(seed: number) {
    tf.setBackend('cpu');
    const t0 = Date.now()

    U.seedRandom(seed || 220)

    testFloatConv()

    // const m = await tf.loadLayersModel("./models/gestures.tfjsmodel.json")
    const sample = sampleModel("oneD")
    const float16weights = true
    const optimize = false
    const opts: Options = { verbose: true, float16weights, optimize }
    logThumb(compileAndTest(sample, opts))

    await testAllModels({ verbose: false, optimize })

    console.log(Date.now() - t0 + "ms")
}

function getSampleModels(): SMap<tf.layers.Layer[]> {
    return {
        id: [tf.layers.inputLayer({
            inputShape: [10, 3, 1]
        })],
        conv2d: [tf.layers.conv2d({
            inputShape: [50, 3, 1],
            kernelSize: [4, 4],
            filters: 16,
            strides: [1, 1],
            padding: 'same',
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        })],
        dense: [
            tf.layers.flatten({
                inputShape: [10, 3, 1],
            }),
            tf.layers.dense({
                units: 5,
                activation: "softmax",
            })],
        padding: [
            tf.layers.inputLayer({
                inputShape: [50, 3, 1]
            }),
            tf.layers.conv2d({
                filters: 16,
                kernelSize: 4,
                strides: 1,
                padding: "same",
                activation: "relu"
            })
        ],
        dspDense: [
            tf.layers.inputLayer({ inputShape: [33] }),
            tf.layers.dense({ units: 20, activation: "relu" }),
            tf.layers.dense({ units: 10, activation: "relu" }),
            tf.layers.dense({ units: 3, activation: "softmax" }),
        ],
        noDsp: [
            tf.layers.inputLayer({ inputShape: [150] }),
            tf.layers.reshape({ targetShape: [50, 3, 1] }),
            tf.layers.conv2d({ filters: 16, kernelSize: 4, strides: 1, padding: "same", activation: "relu" }),
            tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: "same" }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.conv2d({ filters: 16, kernelSize: 2, strides: 1, padding: "same", activation: "relu" }),
            tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: "same" }),
            tf.layers.flatten(),
            tf.layers.dense({ units: 30, activation: "relu" }),
            tf.layers.dense({ units: 3, activation: "softmax" }),
        ],
        tfjsGest: [
            tf.layers.conv2d({
                inputShape: [50, 3, 1],
                kernelSize: [4, 3],
                filters: 16,
                strides: [1, 1],
                padding: 'valid',
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }),
            tf.layers.maxPooling2d({ poolSize: [2, 1], strides: [2, 1] }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.conv2d({
                kernelSize: [2, 1],
                filters: 16,
                strides: 1,
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.conv2d({
                kernelSize: [2, 1],
                filters: 16,
                strides: 1,
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 4,
                kernelInitializer: 'varianceScaling',
                activation: 'softmax'
            })
        ],
        microSpeech: [
            tf.layers.conv2d({
                inputShape: [49, 40, 1],
                kernelSize: [10, 8],
                filters: 8,
                padding: "same",
                activation: "relu",
                strides: 2,
            }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 4,
                kernelInitializer: 'varianceScaling',
                activation: 'softmax'
            })
        ],
        oneD: [
            tf.layers.conv1d({
                inputShape: [50, 4],
                kernelSize: [4],
                strides: 1,
                filters: 16,
                activation: 'relu'
            }),
            tf.layers.maxPooling1d({ poolSize: [2] }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.conv1d({
                kernelSize: [2],
                strides: 1,
                filters: 16,
                activation: 'relu'
            }),
            tf.layers.maxPooling1d({ poolSize: [2] }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.conv1d({
                kernelSize: [2],
                strides: 1,
                filters: 16,
                activation: 'relu'
            }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 3,
                activation: "softmax",
            })
        ],
        oneD2: [
            tf.layers.conv1d({
                inputShape: [10, 3],
                kernelSize: [4],
                strides: 1,
                padding: 'same',
                filters: 1,
                activation: 'relu'
            }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 3,
                activation: "softmax",
            })
        ],
        oneD2_x: [
            tf.layers.conv1d({
                inputShape: [23, 3],
                kernelSize: [4],
                strides: 1,
                padding: 'same',
                filters: 16,
                activation: 'relu'
            }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 3,
                activation: "softmax",
            })
        ],
        avgPool: [
            tf.layers.inputLayer({ inputShape: [150] }),
            tf.layers.reshape({ targetShape: [50, 3, 1] }),
            tf.layers.conv2d({ filters: 16, kernelSize: 4, strides: 1, padding: "same", activation: "relu" }),
            tf.layers.avgPooling2d({ poolSize: 2, strides: 2, padding: "valid" }),
            tf.layers.flatten(),
            tf.layers.dense({ units: 3, activation: "softmax" }),
        ],
        avgPool2: [
            tf.layers.inputLayer({ inputShape: [150] }),
            tf.layers.reshape({ targetShape: [50, 3, 1] }),
            tf.layers.conv2d({ filters: 16, kernelSize: 4, strides: 1, padding: "same", activation: "relu" }),
            tf.layers.avgPooling2d({ poolSize: [2, 1], strides: [2, 1], padding: "valid" }),
            tf.layers.flatten(),
            tf.layers.dense({ units: 3, activation: "softmax" }),
        ],
        avgPool3: [
            tf.layers.inputLayer({ inputShape: [150] }),
            tf.layers.reshape({ targetShape: [50, 3, 1] }),
            // tf.layers.conv2d({ filters: 16, kernelSize: 4, strides: 1, padding: "same", activation: "relu" }),
            tf.layers.avgPooling2d({ poolSize: [8, 1], strides: [2, 1], padding: "valid" }),
            tf.layers.flatten(),
            tf.layers.dense({ units: 3, activation: "softmax" }),
        ],
        depth0: [
            tf.layers.inputLayer({ inputShape: [15, 1, 1] }),
            tf.layers.depthwiseConv2d({ kernelSize: [5, 1], depthMultiplier: 4, strides: 1, useBias: false })
        ],
        depth1: [
            tf.layers.inputLayer({ inputShape: [213, 1, 15] }),
            tf.layers.depthwiseConv2d({ kernelSize: [10, 1], depthMultiplier: 4, strides: 2, useBias: false })
        ],
        batch1: [
            tf.layers.inputLayer({ inputShape: [213, 1, 15] }),
            tf.layers.batchNormalization({})
        ],
        batch2: [
            tf.layers.inputLayer({ inputShape: [213, 1, 100] }),
            tf.layers.batchNormalization({})
        ]
    }
}

let _models: SMap<tf.layers.Layer[]>

export function allSampleModels() {
    if (!_models) _models = getSampleModels()
    return Object.keys(_models).map(sampleModel)
}

export function sampleModel(id: string) {
    const model = tf.sequential();
    model.name = id

    if (!_models) _models = getSampleModels()

    const layers = _models[id]
    if (!layers) {
        let msg = `no such model ${id}; options:\n`
        for (const name of Object.keys(_models)) {
            msg += `- ${name}: ${_models[name].length} layer(s)\n`
        }
        throw new Error(msg)
    }

    for (const l of layers)
        model.add(l);

    // make sure weights are deterministic
    for (const l of model.layers)
        setRandomWeights(l)

    return model;
}

export async function testAllModels(opts: Options) {
    const t0 = Date.now()
    opts = U.flatClone(opts)
    for (const m of allSampleModels()) {
        console.log(`***\n*** ${m.name}\n***`)
        console.log(opts.float16weights ? "--- F16" : "--- F32")
        await compileModelAndFullValidate(m, opts)
        opts.float16weights = !opts.float16weights
        console.log(opts.float16weights ? "--- F16" : "--- F32")
        await compileModelAndFullValidate(m, opts)
    }
    console.log(`\n*** All OK (${Date.now() - t0}ms)\n`)
}

export type EvalSample = number | number[] | number[][] | number[][][]
export interface EvalData {
    x: EvalSample[]
    y: number[][]
}

function flattenSample(s: EvalSample) {
    const res: number[] = []
    const rec = (v: any) => {
        if (Array.isArray(v))
            v.forEach(rec)
        else if (typeof v == "number")
            res.push(v)
        else
            throw new Error("invalid input")
    }
    rec(s)
    return res
}

function argmax(r: ArrayLike<number>) {
    let maxI = 0
    let max = r[0]
    for (let i = 1; i < r.length; ++i) {
        if (r[i] > max) {
            max = r[i]
            maxI = i
        }
    }
    return maxI
}

export function evalModel(cres: CompileResult, data: EvalData) {
    let numOK = 0
    const dim = data.y[0].length
    const confusion = U.range(dim).map(_ => U.range(dim).map(_ => 0))
    for (let i = 0; i < data.x.length; ++i) {
        const predProb = cres.execute(flattenSample(data.x[i]))
        const pred = argmax(predProb)
        const ok = argmax(data.y[i])
        confusion[pred][ok]++
        if (pred == ok) numOK++
    }

    let r = ""

    r += `Accuracy: ${(numOK / data.x.length).toFixed(4)}\n`
    for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
            r += ("     " + confusion[i][j]).slice(-5)
        }
        r += "\n"
    }

    return r
}

function flatten(d: any): number[] {
    const r: number[] = []
    if (Array.isArray(d)) {
        for (const e of d) {
            for (const q of flatten(e)) {
                r.push(q)
            }
        }
    } else {
        r.push(d)
    }
    return r
}

export function runModel(js: string, data: any) {
    const { modelFromRuntime, inputSize } = new Function(js)()
    const runModel: (inp: number[]) => Float32Array = modelFromRuntime(mkRuntime)
    const reqSize: number = inputSize

    let inputs = data.x ? data.x : data
    let outputs = data.y ? data.y : []

    if (Array.isArray(inputs) && flatten(inputs[0]).length == reqSize) {
        for (let i = 0; i < inputs.length; ++i) {
            execModel(inputs[i], outputs[i])
        }
    } else {
        execModel(inputs, outputs)
    }

    function execModel(inp: number[], exp: any) {
        if (inp.length != reqSize) {
            console.error(`bad input size - need ${reqSize} got ${inp.length}`)
            return
        }
        const res = runModel(inp)
        const max = argmax(res)
        if (typeof exp == "number") {
            if (max == exp) {
                console.log("OK!", max)
            } else {
                const tmp = Array.from(res)
                tmp.sort()
                console.log(`got ${max} (${res[max]}), exp ${exp} (we have ${res[exp]}); median ${tmp[tmp.length >> 1]}`)
            }
        } else {
            console.log(max)
        }
    }
}
