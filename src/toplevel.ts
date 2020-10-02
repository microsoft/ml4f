import * as tf from '@tensorflow/tfjs'
import { ThumbProcessor } from './thumb';
import * as assembler from './assembler'
import * as U from './util'
import { compileModel, shapeElts } from './compiler';

function mkProcessorFile() {
    const b = new assembler.File(new ThumbProcessor())

    b.ei.testAssembler(); // just in case

    //   b.disablePeepHole = true

    b.lookupExternalLabel = _name => null;
    b.normalizeExternalLabel = s => s;
    b.throwOnError = true;

    return b
}

function throwAssemblerErrors(b: assembler.File) {
    if (b.errors.length > 0) {
        throw new Error(b.errors[0].message)
    }
}

export function assemble(src: string) {
    let b = mkProcessorFile()
    b.emit(src);

    throwAssemblerErrors(b)

    return {
        src: src,
        buf: b.buf,
        thumbFile: b
    }
}

function randomTensor(shape: tf.Shape, mult = 1) {
    shape = shape.map(s => s == null ? 1 : s)
    const num = shapeElts(shape)
    return tf.tidy(() => tf.tensor(U.range(num).map(_ => mult * U.randomSFloat())).reshape(shape))
}

function setRandomWeights(l: tf.layers.Layer) {
    let idx = 0
    for (const w of l.weights) {
        const mult = 1
        w.write(randomTensor(w.shape, mult))
        idx++
    }
}

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

    /*
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });
    */

    // make sure weights are deterministic
    for (const l of model.layers)
        setRandomWeights(l)

    return { model, desc }
}

const eps = 0.00002
function isNear(a: number, b: number) {
    const diff = Math.abs(a - b)
    if (diff < eps)
        return true
    if (diff / (Math.abs(a) + Math.abs(b)) < eps)
        return true
    return false
}

export async function runBrowser() {
    tf.setBackend('cpu');
    const t0 = Date.now()
    U.seedRandom(220)
    // const m = await tf.loadLayersModel("./models/gestures.tfjsmodel.json")
    const sample = sampleModel()
    // compileModel(sample, { verbose: true })
    compareModel(sample, "sample")

    for (let i = 0; i < 0; ++i) {
        const { model, desc } = randomModel()
        console.log(desc)
        compareModel(model, desc)
    }

    console.log(Date.now() - t0 + "ms")
}

function compareModel(m: tf.LayersModel, desc: string) {
    const verbose = true
    try {
        const randomInput = randomTensor(m.inputs[0].shape)
        const resTensor = m.predict(randomInput) as tf.Tensor
        const res = resTensor.flatten().arraySync()
        const cres = compileModel(m, {
            verbose,
            testInput: randomInput.flatten().arraySync(),
            testOutput: res
        })
        //console.log(res)
        const res2 = cres.execute(randomInput.flatten().arraySync())
        //console.log(res2)

        let numerr = 0
        for (let i = 0; i < res2.length; ++i) {
            if (!isNear(res[i], res2[i])) {
                console.log(`at ${i} ${res[i]} - ${res2[i]} = ${res[i] - res2[i]}`)
                numerr++
                if (numerr > 5) break
            }
        }

        if (numerr)
            throw new Error("mismatch")

        const asmr = assemble(cres.thumb)
        function hex2(n: number) {
            return ("0" + n.toString(16)).slice(-2)
        }
        cres.thumb += "// BUF: " + asmr.buf.map(k => hex2(k & 0xff) + hex2(k >> 8)).join("") + "\n"

        if (verbose)
            console.log(cres.thumb)

    } catch (e) {
        console.log(desc)
        if (!verbose)
            compileModel(m, { verbose: true })
        throw e
    }
}
