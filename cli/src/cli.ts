import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import * as child_process from 'child_process'
import { program as commander } from "commander"
import {
    compileModel, compileModelAndFullValidate,
    evalModel,
    loadFlatJSONModel,
    loadTfjsModelJSON,
    Options, sampleModel, testAllModels,
    testFloatConv
} from '../..'

interface CmdOptions {
    debug?: boolean
    output?: string
    basename?: string
    validate?: boolean
    testData?: boolean
    sampleModel?: string
    selfTest?: boolean
    optimize?: boolean
    float16?: boolean
    eval?: string
}

let options: CmdOptions

function getCompileOptions(): Options {
    return {
        optimize: options.optimize,
        verbose: options.debug,
        includeTest: options.testData,
        float16weights: options.float16,
        testOutputFromJS: true,
    }
}

function mkdirP(thePath: string) {
    if (thePath == "." || !thePath) return;
    if (!fs.existsSync(thePath)) {
        mkdirP(path.dirname(thePath))
        fs.mkdirSync(thePath)
    }
}

function built(fn: string) {
    return path.join(options.output, fn)
}

function loadJSONModel(modelPath: string) {
    const modelBuf = fs.readFileSync(modelPath)

    if (modelBuf[0] != 0x7b)
        throw new Error("model not in JSON format")

    const preModel = JSON.parse(modelBuf.toString("utf8"))

    const model0 = loadFlatJSONModel(preModel)
    if (model0) return model0

    const modelJSON: tf.io.ModelJSON = preModel

    if (!modelJSON.modelTopology)
        throw new Error("model not in tf.js JSON format")

    const model = loadTfjsModelJSON(modelJSON)

    if (modelJSON.weightsManifest != null) {
        const dirName = path.dirname(modelPath);
        const buffers: Buffer[] = [];
        model.weightSpecs = [];
        for (const group of modelJSON.weightsManifest) {
            for (const weightPath of group.paths) {
                buffers.push(fs.readFileSync(path.join(dirName, weightPath)));
            }
            model.weightSpecs.push(...group.weights);
        }
        model.weightData = Buffer.concat(buffers).buffer;
    }

    return model;
}

function runCmd(cmd: string, args: string[]) {
    const info = `${cmd} ${args.join(" ")}`
    console.log(`RUN ${info}`)
    const res = child_process.spawnSync(cmd, args, {
        stdio: "inherit"
    })
    if (res.status != 0)
        throw new Error(`non-zero status from ${info}`)
    console.log("RUN OK")
}

function fromPB(modelPath: string) {
    const tmpPath = built("converted.h5")
    runCmd("python3", [
        "-c", `import tensorflow; m = tensorflow.keras.models.load_model('${path.dirname(modelPath)}'); m.save('${tmpPath}')`
    ])
    return loadModel(tmpPath)
}

function fromH5(modelPath: string) {
    const tmpPath = built("converted.tfjs")
    runCmd("tensorflowjs_converter", [
        "--input_format", "keras",
        "--output_format", "tfjs_layers_model",
        modelPath, tmpPath
    ])
    return loadJSONModel(path.join(tmpPath, "model.json"))
}

function checkSubdir(modelPath: string, n: string) {
    const saved = path.join(modelPath, n)
    if (fs.existsSync(saved))
        return saved
    return modelPath
}

async function loadModel(modelPath: string): Promise<tf.io.ModelArtifacts> {
    modelPath = checkSubdir(modelPath, "saved_model.pb")
    modelPath = checkSubdir(modelPath, "model.json")

    let modelBuf = fs.readFileSync(modelPath)

    if (modelBuf[0] == 0x08)
        return fromPB(modelPath)

    if (modelBuf[0] == 0x89)
        return fromH5(modelPath)

    return loadJSONModel(modelPath)

}

async function processModelFile(modelFile: string) {
    tf.setBackend("cpu")

    mkdirP(options.output)

    let m: tf.LayersModel
    if (options.sampleModel) {
        m = sampleModel(options.sampleModel)
    } else {
        const model = await loadModel(modelFile)
        if (!model.weightData)
            throw new Error(`model '${modelFile}' is missing weights`)
        m = await tf.loadLayersModel({ load: () => Promise.resolve(model) })
    }

    const opts = getCompileOptions()
    const cres = !options.validate ? compileModel(m, opts) : await compileModelAndFullValidate(m, opts)

    write(".asm", cres.thumb)
    write(".js", cres.js)
    write(".ml4f", cres.machineCode)

    let evalInfo = `\n*** ${built(options.basename + ".ml4f")}\n\n`

    if (options.eval) {
        const ev = evalModel(cres, JSON.parse(fs.readFileSync(options.eval, "utf8")))
        evalInfo += ev + "\n"
    }

    evalInfo += cres.memInfo + "\n"
    evalInfo += cres.timeInfo + "\n"

    write(".txt", evalInfo + "\n")

    console.log("\n" + evalInfo)

    function write(ext: string, buf: string | Uint8Array) {
        const fn = built(options.basename + ext)
        const binbuf = typeof buf == "string" ? Buffer.from(buf, "utf8") : buf
        console.log(`write ${fn} (${binbuf.length} bytes)`)
        fs.writeFileSync(fn, binbuf)
    }
}

export async function mainCli() {
    // require('@tensorflow/tfjs-node');

    // shut up warning
    (tf.backend() as any).firstUse = false;

    const pkg = require("../../package.json")
    commander
        .version(pkg.version)
        .option("-d, --debug", "enable debugging")
        .option("-n, --no-validate", "don't validate resulting model")
        .option("-g, --no-optimize", "don't optimize IR")
        .option("-h, --float16", "use float16 weights")
        .option("-t, --test-data", "include test data in binary model")
        .option("-T, --self-test", "run self-test of all included sample models")
        .option("-s, --sample-model <name>", "use an included sample model")
        .option("-e, --eval <file.json>", "evaluate model (confusion matrix, accuracy) on a given test data")
        .option("-o, --output <folder>", "path to store compilation results (default: 'built')")
        .option("-b, --basename <name>", "basename of model files (default: 'model')")
        .arguments("<model>")
        .parse(process.argv)

    options = commander as CmdOptions

    if (!options.output) options.output = "built"
    if (!options.basename) options.basename = "model"

    if (options.selfTest) {
        testFloatConv()
        const opts = getCompileOptions()
        opts.includeTest = false
        await testAllModels(opts)
        process.exit(0)
    }

    if (!options.sampleModel && commander.args.length != 1) {
        console.error("exactly one model argument expected")
        process.exit(1)
    }

    try {
        await processModelFile(commander.args[0])
    } catch (e) {
        console.error(e.stack)
    }
}

if (require.main === module) mainCli()
