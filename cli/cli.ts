import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as U from '../src/util'
import * as path from 'path'
import * as child_process from 'child_process'
import { program as commander } from "commander"

interface CmdOptions {
    debug?: boolean;
    keepTmp?: boolean;
}

let options: CmdOptions

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
    const tmpPath = "ml4f-tmppb-" + U.randomUint32() + ".h5"
    try {
        runCmd("python3", [
            "-c", `import tensorflow; m = tensorflow.keras.models.load_model('${path.dirname(modelPath)}'); m.save('${tmpPath}')`
        ])
        return loadModel(tmpPath)
    } finally {
        if (!options.keepTmp) {
            if (fs.existsSync(tmpPath))
                fs.unlinkSync(tmpPath)
        }
    }
}

function fromH5(modelPath: string) {
    const tmpPath = "ml4f-tmp-" + U.randomUint32()
    try {
        runCmd("tensorflowjs_converter", [
            "--input_format", "keras",
            "--output_format", "tfjs_layers_model",
            modelPath, tmpPath
        ])
        return loadJSONModel(path.join(tmpPath, "model.json"))
    } finally {
        if (!options.keepTmp) {
            if (fs.existsSync(tmpPath)) {
                for (const fn of fs.readdirSync(tmpPath)) {
                    fs.unlinkSync(path.join(tmpPath, fn))
                }
                fs.rmdirSync(tmpPath)
            }
        }
    }
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

export async function mainCli() {
    // require('@tensorflow/tfjs-node');

    const pkg = require("../../package.json")
    commander
        .version(pkg.version)
        .option("-d, --debug", "enable debugging")
        .option("-t, --keep-tmp", "keep temporary files")
        .arguments("<model>")
        .parse(process.argv)

    options = commander as CmdOptions

    if (commander.args.length != 1) {
        console.error("exactly one model argument expected")
        process.exit(1)
    }

    const modelFile = commander.args[0]

    try {
        const model = loadModel(modelFile)
        const m = await tf.loadLayersModel({ load: () => model })
        m.summary()
    } catch (e) {
        console.error(e.stack)
    }
}
