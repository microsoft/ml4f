import * as tf from '@tensorflow/tfjs'
import * as U from './util'
import { CompileResult } from './compiler';
import { compileAndTest, setRandomWeights } from './driver';

function sampleModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [50, 3, 1],
        kernelSize: [4, 3],
        filters: 16,
        strides: [1, 1],
        padding: 'valid',
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: [2, 1], strides: [2, 1] }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.conv2d({
        kernelSize: [2, 1],
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    //model.add(tf.layers.maxPooling2d({ poolSize: [2, 1], strides: [2, 1] }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.conv2d({
        kernelSize: [2, 1],
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 4,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    // make sure weights are deterministic
    for (const l of model.layers)
        setRandomWeights(l)

    return model;
}


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

    return { model, desc }
}

function logThumb(cres: CompileResult) {
    let str = cres.thumb
    function hex2(n: number) {
        return ("0" + n.toString(16)).slice(-2)
    }
    str += "// BUF: "
    for (const v of cres.machineCode) str += hex2(v)
    console.log(str)
}


export async function runBrowser() {
    tf.setBackend('cpu');
    const t0 = Date.now()
    U.seedRandom(220)
    // const m = await tf.loadLayersModel("./models/gestures.tfjsmodel.json")
    const sample = sampleModel()
    // compileModel(sample, { verbose: true }) 
    logThumb(compileAndTest(sample, "sample"))

    for (let i = 0; i < 0; ++i) {
        const { model, desc } = randomModel()
        console.log(desc)
        compileAndTest(model, desc)
    }

    console.log(Date.now() - t0 + "ms")
}
