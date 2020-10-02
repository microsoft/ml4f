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

async function loadModel(modelPath: string) {
    let modelBuf = fs.readFileSync(modelPath)

    if (modelBuf[0] == 0x08)
        throw new Error(".savedmodel files not supported; use model.save('mymodel.h5') to save in H5 format")

    if (modelBuf[0] == 0x89) {
        const tmpPath = "ml4f-tmp-" + U.randomUint32()
        const cmd = "tensorflowjs_converter"
        const args = [
            "--input_format", "keras",
            "--output_format", "tfjs_layers_model",
            modelPath, tmpPath
        ]
        const info = `${cmd} ${args.join(" ")}`
        console.log(`RUN ${info}`)
        try {
            const res = child_process.spawnSync(cmd, args, {
                stdio: "inherit"
            })
            if (res.status != 0)
                throw new Error(`non-zero status from ${info}`)
            console.log("converted!")
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
    const m = await tf.loadLayersModel({ load: () => loadModel(modelFile) })
    m.summary()
}
