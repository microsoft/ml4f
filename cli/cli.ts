import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as U from '../src/util'
import * as path from 'path'
import * as child_process from 'child_process'
import { option, program as commander } from "commander"
import { compileModel, compileModelAndFullValidate, optionsWithTestData, setRandomWeights } from '../src/driver'
import { Options } from '../src/compiler'
import { sampleModel, testAllModels } from '../src/main'
import { testFloatConv } from '../src/float16'

interface CmdOptions {
    debug?: boolean
    output?: string
    validate?: boolean
    testData?: boolean
    sampleModel?: string
    testAll?: boolean
    optimize?: boolean
    float16?: boolean
}

let options: CmdOptions

function getCompileOptions(): Options {
    return {
        optimize: options.optimize,
        verbose: options.debug,
        includeTest: options.testData,
        float16weights: options.float16,
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

    const modelJSON = JSON.parse(modelBuf.toString("utf8")) as tf.io.ModelJSON

    // remove regularizers, as we're not going to train the model, and unknown regularizers
    // cause it to fail to load
    const cfg = (modelJSON.modelTopology as any)?.model_config?.config
    for (const layer of cfg?.layers || []) {
        const layerConfig = layer?.config
        if (layerConfig) {
            layerConfig.bias_regularizer = null
            layerConfig.activity_regularizer = null
            layerConfig.bias_constraint = null
        }
    }

    const model: tf.io.ModelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy,
        trainingConfig: modelJSON.trainingConfig,
        userDefinedMetadata: modelJSON.userDefinedMetadata
    }
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
        const model = loadModel(modelFile)
        m = await tf.loadLayersModel({ load: () => model })
    }

    const opts = getCompileOptions()
    const cres = !options.validate ? compileModel(m, opts) : await compileModelAndFullValidate(m, opts)

    write(".asm", cres.thumb)
    write(".js", cres.js)
    write(".bin", cres.machineCode)

    console.log(cres.memInfo)
    console.log(cres.timeInfo)

    function write(ext: string, buf: string | Uint8Array) {
        const fn = built("model" + ext)
        const binbuf = typeof buf == "string" ? Buffer.from(buf, "utf8") : buf
        console.log(`write ${fn} (${binbuf.length} bytes)`)
        fs.writeFileSync(fn, binbuf)
    }
}

export async function mainCli() {
    // require('@tensorflow/tfjs-node');

    const pkg = require("../../package.json")
    commander
        .version(pkg.version)
        .option("-d, --debug", "enable debugging")
        .option("-n, --no-validate", "don't validate resulting model")
        .option("-g, --no-optimize", "don't optimize IR")
        .option("-h, --float16", "use float16 weights")
        .option("-t, --test-data", "include test data in binary model")
        .option("-T, --test-all", "test all included sample models")
        .option("-s, --sample-model <name>", "use an included sample model")
        .option("-o, --output <folder>", "path to store compilation results (default: 'built')")
        .arguments("<model>")
        .parse(process.argv)

    options = commander as CmdOptions

    if (!options.output) options.output = "built"

    if (options.testAll) {
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
