import * as tf from '@tensorflow/tfjs'
import * as U from './util'
import { CompileResult, Options } from './compiler';
import { compileAndTest, compileModelAndFullValidate, setRandomWeights } from './driver';
import { testFloatConv } from './float16';


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


export async function runBrowser() {
    tf.setBackend('cpu');
    const t0 = Date.now()
    U.seedRandom(220)

    testFloatConv()

    // const m = await tf.loadLayersModel("./models/gestures.tfjsmodel.json")
    const sample = sampleModel("tfjsGest")
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
        await compileModelAndFullValidate(m, opts)
        opts.float16weights = !opts.float16weights
        await compileModelAndFullValidate(m, opts)
    }
    console.log(`\n*** All OK (${Date.now() - t0}ms)\n`)
}