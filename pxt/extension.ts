import * as ml4f from "..";
import * as tf from '@tensorflow/tfjs'

export function inIFrame() {
    try {
        return typeof window !== "undefined" && window.self !== window.top
    } catch (e) {
        return typeof window !== "undefined"
    }
}

const CHANGE = 'change'
export const READ = "read"
export const MESSAGE_PACKET = "messagepacket"
const HIDDEN = "hidden"
const SHOWN = "shown"
const SENDER = "jacdac-editor-extension"
const CONNECT = "connect"

export interface ReadResponse {
    code?: string;
    json?: string;
    jres?: string;
}


export interface SMap<T> {
    [index: string]: T
}

const accelSample =
    `export function _sample() {
    return [
        input.acceleration(Dimension.X),
        input.acceleration(Dimension.Y),
        input.acceleration(Dimension.Z)
    ]
}`

export class MakeCodeEditorExtensionClient {
    private readonly pendingCommands: {
        [key: string]: {
            action: string;
            resolve: (resp: any) => void;
            reject: (e: any) => void;
        }
    } = {};
    private readonly extensionId: string = inIFrame() ? window.location.hash.substr(1) : undefined;
    private _target: any; // full apptarget
    private _connected = false;
    private _visible = false;

    constructor() {
        this.handleMessage = this.handleMessage.bind(this);
        window.addEventListener("message", this.handleMessage, false);
        // notify parent that we're ready
        this.init();
    }

    emit(id: string, arg?: any) {
        console.log("EMIT", id, { arg })
    }

    log(msg: string) {
        console.log(`ML4F-PXT: ${msg}`)
    }

    get target() {
        return this._target;
    }

    get connected() {
        return this._connected;
    }

    get visible() {
        return this._visible;
    }

    private setVisible(vis: boolean) {
        if (this._visible !== vis) {
            this._visible = vis;
            this.emit(CHANGE);
        }
    }

    private nextRequestId = 1;
    private mkRequest(resolve: (resp: any) => void, reject: (e: any) => void, action: string, body?: any): any {
        const id = "ml_" + this.nextRequestId++;
        this.pendingCommands[id] = { action, resolve, reject };
        return {
            type: "pxtpkgext",
            action,
            extId: this.extensionId,
            response: true,
            id,
            body
        }
    }

    private sendRequest<T>(action: string, body?: any): Promise<T> {
        this.log(`send ${action}`)
        if (!this.extensionId)
            return Promise.resolve(undefined);

        return new Promise((resolve, reject) => {
            const msg = this.mkRequest(resolve, reject, action, body);
            window.parent.postMessage(msg, "*");
        })
    }

    private handleMessage(ev: any) {
        const msg = ev.data;
        if (msg?.type !== "pxtpkgext")
            return;
        if (!msg.id) {
            switch (msg.event) {
                case "extinit":
                    this.log(`init`)
                    this._target = msg.target;
                    this._connected = true;
                    this.emit(CONNECT);
                    this.emit(CHANGE);
                    break;
                case "extloaded":
                    this.log(`loaded`)
                    break;
                case "extshown":
                    this.setVisible(true)
                    this.refresh();
                    this.emit(SHOWN);
                    this.emit(CHANGE);
                    break;
                case "exthidden":
                    this.setVisible(false)
                    this.emit(HIDDEN);
                    this.emit(CHANGE);
                    break;
                case "extdatastream":
                    this.emit('datastream', true);
                    break;
                case "extconsole":
                    this.emit('console', msg.body);
                    break;
                case "extmessagepacket":
                    this.emit(MESSAGE_PACKET, msg.body);
                    break;
                default:
                    console.debug("Unhandled event", msg);
            }
        }
        else {
            const { action, resolve, reject } = this.pendingCommands[msg.id] || {};
            delete this.pendingCommands[msg.id];

            if (msg.success && resolve)
                resolve(msg.resp);
            else if (!msg.success && reject)
                reject(msg.resp);
            // raise event as well
            switch (action) {
                case "extinit":
                    this._connected = true;
                    this.emit('CONNECT');
                    this.emit(CHANGE);
                    break;
                case "extusercode":
                    // Loaded, set the target
                    this.emit('readuser', msg.resp);
                    this.emit(CHANGE);
                    break;
                case "extreadcode":
                    // Loaded, set the target
                    this.emit(READ, msg.resp);
                    this.emit(CHANGE);
                    break;
                case "extwritecode":
                    this.emit('written', undefined);
                    break;
            }
        }
    }

    private async init() {
        this.log(`initializing`)
        await this.sendRequest<void>('extinit');
        this.log(`connected`)
        await this.refresh();
    }

    private async refresh() {
        this.log(`refresh`)
        const r = await this.read();
    }

    async read(): Promise<ReadResponse> {
        if (!this.extensionId) {
            const r: ReadResponse = {};
            this.emit(READ, r);
            return r;
        } else {
            const resp: ReadResponse = await this.sendRequest('extreadcode');
            return resp;
        }
    }

    async readUser() {
        await this.sendRequest('extusercode')
    }

    async write(code: string, json?: string, jres?: string, dependencies?: SMap<string>): Promise<void> {
        if (!this.extensionId) {
            // Write to local storage instead
            this.emit('written', undefined);
        } else {
            await this.sendRequest<void>('extwritecode', {
                code: code || undefined,
                json: json || undefined,
                jres: jres || undefined,
                dependencies
            })
        }
    }

    async queryPermission() {
        await this.sendRequest('extquerypermission');
    }

    async requestPermission(console: boolean) {
        await this.sendRequest('extrequestpermission', {
            console
        })
    }

    async dataStreamConsole(console: boolean) {
        await this.sendRequest('extdatastream', {
            console
        })
    }

    async dataStreamMessages(messages: boolean) {
        await this.sendRequest('extdatastream', {
            messages
        })
    }
}

export interface FlatJSONModel {
    name: string
    inputTypes: string[] // ["x","y","z"]; ["pressure"]
    labels: string[]
    modelJSON: {}
    inputInterval: number // ms
    weights: number[] // UInt32Array (little endian)
}

export async function start() {
    tf.setBackend("cpu")

    const options: SMap<boolean> = {
        f16: true
    }
    const pxtClient = new MakeCodeEditorExtensionClient()

    const maindiv = document.createElement("div")
    maindiv.style.background = "white"
    document.body.appendChild(maindiv)

    const status = div("")
    maindiv.append(status)
    setStatus("waiting for model file")

    const d = div("Drop TF.JS model file here")
    d.style.padding = "2em"
    d.style.margin = "1em 0em"
    d.style.border = "1px dotted gray"
    maindiv.append(d)
    addCheckbox("f16", "Use float16 type")

    const dropbox = maindiv
    dropbox.addEventListener("dragenter", stopEv, false);
    dropbox.addEventListener("dragover", stopEv, false);
    dropbox.addEventListener("drop", e => {
        setStatus("reading model")
        stopEv(e)
        const file = e.dataTransfer.files.item(0)
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const mod: FlatJSONModel = JSON.parse(e.target.result as string)
                await compileModel(mod, file.name)
            } catch (e) {
                console.error(e.stack)
                setError(e.message)
            }
        }
        reader.readAsText(file)
    }, false);

    function shapeElements(shape: number[]) {
        let res = 1
        for (const s of shape) if (s != null) res *= s
        return res
    }

    function toCamelCase(name: string) {
        return name.replace(/(^|( +))(.)/g, (_0, _1, _2, l) => l.toUpperCase())
    }

    async function compileModel(mod: FlatJSONModel, fileName: string) {
        const name = mod.name || fileName
        const ma = ml4f.loadFlatJSONModel(mod)
        const m = await tf.loadLayersModel({ load: () => Promise.resolve(ma) })
        const inpTen = m.getInputAt(0) as tf.SymbolicTensor
        const numClasses = shapeElements((m.getOutputAt(0) as tf.SymbolicTensor).shape)
        const labels = (mod.labels || []).slice()
        while (labels.length > numClasses) labels.pop()
        while (labels.length < numClasses) labels.push("class " + labels.length)
        const inputShape = inpTen.shape
        const samplingPeriod = mod.inputInterval || 100
        setStatus("compiling...") // can't see that...
        const res = await ml4f.compileModelAndFullValidate(m, {
            verbose: false,
            includeTest: true,
            float16weights: options.f16,
            optimize: true
        })
        setStatus("compiled!")
        const shape2 = inputShape.filter(v => v != null)
        const samplesInWindow = shape2.shift()
        const elementsInSample = shapeElements(shape2)

        let code =
            `// model: ${name}; input: ${JSON.stringify(inputShape)}; sampling at: ${samplingPeriod}ms\n` +
            `// ${res.memInfo}\n` +
            `// ${res.timeInfo}\n`

        code += "export const enum MLEvent {\n"
        let idx = 0
        for (let lbl of labels) {
            lbl = lbl.replace(/_/g, " ")
            code += `    //% block="${lbl}"\n`
            code += `    ${toCamelCase(lbl)} = ${idx},\n`
            idx++
        }
        code += `}\n\n`
        code += `namespace ml {\n`
        code += `
            let _classifier: Classifier
            export function classifier() {
                if (_classifier) return _classifier
                _classifier = new Classifier(input => _model.invoke(input), _sample)
                _classifier.detectionThreshold = 0.7
                _classifier.samplingInterval = ${Math.round(samplingPeriod)} // ms
                _classifier.samplesOverlap = ${Math.max(samplesInWindow >> 2, 1)}
                _classifier.samplesInWindow = ${samplesInWindow}
                _classifier.elementsInSample = ${elementsInSample}
                _classifier.noiseClassNo = -1 // disable
                _classifier.noiseSuppressionTime = 500 // ms
                return _classifier
            }

            /**
             * Run some code when a particular ML event is detected.
             */
            //% blockId=ml_on_event block="on ml event %condition"
            //% blockGap=12 shim=input::onLightConditionChanged
            export function onEvent(mlevent: MLEvent, handler: () => void) {
                classifier().onEvent(mlevent, handler)
            }
            `
        code += "\n" + accelSample + "\n" // TODO

        code +=
            `export const _model = new ml4f.Model(\n` +
            "hex`"
        for (let i = 0; i < res.machineCode.length; ++i) {
            code += ("0" + res.machineCode[i].toString(16)).slice(-2)
            if ((i + 3) % 32 == 0)
                code += "\n"
        }
        code += "`);\n"
        code += "\n} // namespace ml\n"

        console.log(code.replace(/([a-f0-9]{64}\n)+/, "..."))
        await pxtClient.write(code)
        setStatus("done; you can Go back now")
    }

    function stopEv(e: Event) {
        e.stopPropagation();
        e.preventDefault();
    }

    function div(text: string): HTMLDivElement {
        const d = document.createElement("div")
        d.textContent = text
        return d
    }

    function setError(msg: string) {
        status.style.color = "red"
        status.textContent = "Error: " + msg
    }

    function setStatus(msg: string) {
        status.style.color = "green"
        status.textContent = msg
    }

    function addCheckbox(field: string, name: string) {
        const lbl = document.createElement("label")
        lbl.textContent = name
        const box = document.createElement("input")
        lbl.prepend(box)
        box.type = "checkbox"
        box.checked = !!options[field]
        box.addEventListener('change', () => {
            if (box.checked)
                options[field] = !!box.checked
        });
        maindiv.appendChild(lbl)
    }
}
