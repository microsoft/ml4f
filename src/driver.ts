import * as tf from '@tensorflow/tfjs'
import { ThumbProcessor } from './thumb';
import * as assembler from './assembler'
import * as U from './util'
import { compileModel, CompileResult, shapeElts } from './compiler';

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

    const buf = new Uint8Array(b.buf.length << 1)
    for (let i = 0; i < b.buf.length; ++i) {
        buf[i << 1] = b.buf[i] & 0xff
        buf[(i << 1) + 1] = (b.buf[i] >> 8) & 0xff
    }

    return buf
}

function randomTensor(shape: tf.Shape, mult = 1) {
    shape = shape.map(s => s == null ? 1 : s)
    const num = shapeElts(shape)
    return tf.tidy(() => tf.tensor(U.range(num).map(_ => mult * U.randomSFloat())).reshape(shape))
}

export function setRandomWeights(l: tf.layers.Layer) {
    let idx = 0
    for (const w of l.weights) {
        const mult = 1
        w.write(randomTensor(w.shape, mult))
        idx++
    }
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

export function compileAndTest(m: tf.LayersModel, desc: string = "") {
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

        cres.machineCode = assemble(cres.thumb)

        return cres
    } catch (e) {
        if (desc)
            console.log(desc)
        if (!verbose)
            compileModel(m, { verbose: true })
        throw e
    }
}
