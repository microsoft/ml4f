import * as tf from '@tensorflow/tfjs'
import { ThumbProcessor } from './thumb';
import * as assembler from './assembler'
import * as U from './util'
import { assignLayerInfos, compileModelCore, CompileResult, LayerStats, partialModels, prefixModels, shapeElts } from './compiler';
import { Options } from './ir';

const epsF32 = 0.00009
const epsF16 = 0.01

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
    const procFile = mkProcessorFile()
    procFile.emit(src);

    throwAssemblerErrors(procFile)

    // 16-byte aligned size
    const binary = new Uint8Array(((procFile.buf.length << 1) + 15) & ~15)
    for (let i = 0; i < procFile.buf.length; ++i) {
        binary[i << 1] = procFile.buf[i] & 0xff
        binary[(i << 1) + 1] = (procFile.buf[i] >> 8) & 0xff
    }

    return { binary, procFile }
}

function randomTensor(shape: tf.Shape, mult = 1) {
    shape = shape.map(s => s == null ? 1 : s)
    const num = shapeElts(shape)
    return tf.tidy(() => tf.tensor(U.range(num).map(_ => mult * U.randomSFloat())).reshape(shape))
}

function randomPosTensor(shape: tf.Shape, mult = 1) {
    shape = shape.map(s => s == null ? 1 : s)
    const num = shapeElts(shape)
    return tf.tidy(() => tf.tensor(U.range(num).map(_ => mult * U.randomUFloat())).reshape(shape))
}

export function setRandomWeights(l: tf.layers.Layer) {
    let idx = 0
    for (const w of l.weights) {
        const mult = 1
        if (w.originalName.endsWith("/moving_variance"))
            w.write(randomPosTensor(w.shape, mult))
        else
            w.write(randomTensor(w.shape, mult))
        idx++
    }
}

function isNear(a: number, b: number, eps: number) {
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

        if (count++ > (opts.includeTest ? 1000 : 100) || maxMul > 0.1) {
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
    const ares = assemble(cres.thumb)
    cres.machineCode = ares.binary

    let idx = 0
    for (const st of cres.stats.layers) {
        st.codeBytes = ares.procFile.lookupLabel("end_" + idx) - ares.procFile.lookupLabel("begin_" + idx)
        idx++
    }

    const st = getStatsFromBin(cres.machineCode, cres.stats.total)
    cres.memInfo = st.info
    return cres
}

export async function compileModelAndFullValidate(m: tf.LayersModel, opts: Options) {
    assignLayerInfos(m, opts)

    const optsPart = U.flatClone(opts)
    optsPart.includeTest = false

    console.log("Validating partial models...")
    for await (const mod of partialModels(m, optsPart)) {
        for (const l of mod.layers)
            setRandomWeights(l)
        compileAndTest(mod, optsPart)
    }

    console.log("Validating prefix models...")
    for await (const mod of prefixModels(m, optsPart)) {
        compileAndTest(mod, optsPart)
    }

    console.log("Compiling full model...")

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
        if (!isNear(res[i], res2[i], opts.float16weights ? epsF16 : epsF32)) {
            console.log(`at ${i} ${res[i]}[exp] - ${res2[i]} = ${res[i] - res2[i]}`)
            numerr++
            if (numerr > 5) break
        }
    }
    if (numerr)
        throw new Error("mismatch")
}

export function compileAndTest(m: tf.LayersModel, options: Options) {
    let cres: CompileResult
    try {
        options = optionsWithTestData(m, options)
        cres = compileModel(m, options)
        validateCompilation(cres)
        return cres
    } catch (e) {
        if (options.info)
            console.log(options.info)
        if (!cres || !options.verbose) {
            options.verbose = true
            cres = compileModelCore(m, options)
        }
        console.log(cres.js)
        console.log("Failing model: ", m.name)
        throw e
    }
}

function readU32(bin: Uint8Array, off: number) {
    return (bin[off] | (bin[off + 1] << 8) | (bin[off + 2] << 16) | (bin[off + 3] << 24)) >>> 0
}

function readU32s(bin: Uint8Array) {
    const res: number[] = []
    for (let i = 0; i < bin.length; i += 4) {
        res.push(readU32(bin, i))
    }
    return res
}

export function getStatsFromBin(bin: Uint8Array, stats?: LayerStats) {
    let [magic0, magic1, hdSize, totalSize, weightsOff, testInpOff, testOutOff, arenaSize] = readU32s(bin.slice(0, 64))
    if (magic0 != 0x30470f62)
        return null
    const modelSize = testInpOff || totalSize
    const codeSize = weightsOff - hdSize
    const codePerc = codeSize * 100 / modelSize
    const testSize = totalSize - modelSize

    function sz(n: number) {
        return (n / 1024).toFixed(2) + "k"
    }
    const info =
        `model: ${sz(modelSize)}; ` +
        `code: ${sz(codeSize)} (${codePerc.toFixed(1)}%); ` +
        `arena: ${sz(arenaSize)}; test ${sz(testSize)}`

    if (stats) {
        stats.arenaBytes = arenaSize
        stats.codeBytes = codeSize
        stats.weightBytes = modelSize - codeSize
    }

    return {
        info,
        modelSize,
        codeSize,
        testSize,
        totalSize,
        arenaSize
    }
}

export function loadTfjsModelJSON(modelJSON: tf.io.ModelJSON) {
    // remove regularizers, as we're not going to train the model, and unknown regularizers
    // cause it to fail to load
    const cfg = (modelJSON.modelTopology as any)?.model_config?.config
    const outLayers: any[] = []

    let seq_id = 0

    function addLayer(layer: any) {
        const layerConfig = layer?.config
        if (layerConfig) {
            layerConfig.bias_regularizer = null
            layerConfig.activity_regularizer = null
            layerConfig.bias_constraint = null
        }

        if (layer.class_name == "Sequential") {
            seq_id++
            for (const l of layer.config.layers) {
                if (l.class_name == "InputLayer")
                    continue
                if (l.config.name == "dropout")
                l.config.name += "_seq_" + seq_id
                addLayer(l)
            }
        } else {
            outLayers.push(layer)
        }
    }

    if (cfg?.layers) {
        cfg.layers.forEach(addLayer)
        cfg.layers = outLayers
    }

    const model: tf.io.ModelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy,
        trainingConfig: modelJSON.trainingConfig,
        userDefinedMetadata: modelJSON.userDefinedMetadata
    }

    return model
}

export function loadFlatJSONModel(preModel: any) {
    if (!preModel.modelJSON)
        return null

    let modelJSON: tf.io.ModelJSON
    if (typeof preModel.modelJSON == "string")
        modelJSON = JSON.parse(preModel.modelJSON)
    else
        modelJSON = preModel.modelJSON

    const model = loadTfjsModelJSON(modelJSON)
    const arr: number[] = preModel.weights
    if (Array.isArray(arr)) {
        model.weightData = new Uint32Array(arr).buffer
        model.weightSpecs = (modelJSON as any).weightSpecs
    }

    return model
}

export function toCSource(name: string, machineCode: Uint8Array) {
    if (machineCode.length & 3) throw new Error()
    const u32 = new Uint32Array(machineCode.buffer)
    let r = `const unsigned ${name}[${u32.length}] = {\n`
    const chunk = 8
    for (let off = 0; off < u32.length; off += chunk) {
        r += "    "
        r += Array.from(u32.slice(off, off + chunk))
            .map(n => "0x" + ("00000000" + n.toString(16)).slice(-8) + ", ")
            .join("")
        r += "\n"
    }
    r += "};\n"
    return r
}
