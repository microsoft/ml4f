import * as tf from '@tensorflow/tfjs'
import { ThumbProcessor } from './thumb';
import * as assembler from './assembler'
import * as U from './util'
import { assignLayerInfos, compileModelCore, CompileResult, partialModels, shapeElts } from './compiler';
import { Options } from './ir';

function mkProcessorFile() {
    const b = new assembler.File(new ThumbProcessor())

    b.ei.testAssembler(); // just in case

    b.disablePeepHole = true

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

export function optionsWithTestData(m: tf.LayersModel, opts: Options) {
    opts = U.flatClone(opts)
    let count = 0
    let maxMul = 0
    while (true) {
        const randomInput = randomTensor(m.inputs[0].shape)
        const resTensor = m.predict(randomInput) as tf.Tensor
        const res = resTensor.flatten().arraySync()
        let sum = 0
        let mul = 1
        for (const r of res) {
            sum += r
            mul *= r
        }

        const isSoftmax = Math.abs(sum - 1) < 0.1
        if (!isSoftmax) {
            save()
            break
        }

        if (mul > maxMul) {
            maxMul = mul
            save()
        }

        if (count++ > (opts.tryHard ? 1000 : 100) || maxMul > 0.1) {
            if (!mul)
                save()
            break
        }

        function save() {
            opts.testInput = randomInput.flatten().arraySync()
            opts.testOutput = res
        }
    }
    return opts
}

export function compileModel(m: tf.LayersModel, opts: Options) {
    const cres = compileModelCore(m, opts)
    cres.machineCode = assemble(cres.thumb)
    return cres
}

export async function compileModelAndFullValidate(m: tf.LayersModel, opts: Options) {
    assignLayerInfos(m, opts)

    console.log("Validating partial models...")
    const iter = partialModels(m, opts)
    while (true) {
        const m = (await iter.next()).value
        if (!m)
            break
        for (const l of m.layers)
            setRandomWeights(l)
        compileAndTest(m, opts)
    }

    console.log("Compiling full model...")

    opts.tryHard = true

    // also test the top-level one again
    return compileAndTest(m, opts)
}

export function validateCompilation(cres: CompileResult) {
    const opts = cres.options
    const res = opts.testOutput
    const res2 = cres.execute(opts.testInput)
    if (cres.options.verbose)
        console.log("Test output", res2)
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
}

export function compileAndTest(m: tf.LayersModel, options: Options) {
    try {
        options = optionsWithTestData(m, options)
        const cres = compileModel(m, options)
        validateCompilation(cres)
        return cres
    } catch (e) {
        if (options.info)
            console.log(options.info)
        if (!options.verbose) {
            options.verbose = true
            compileModelCore(m, options)
        }
        throw e
    }
}
