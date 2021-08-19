(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('..'), require('@tensorflow/tfjs')) :
    typeof define === 'function' && define.amd ? define(['exports', '..', '@tensorflow/tfjs'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.pxtml4f = global.pxtml4f || {}, global.ml4f, global.tf));
}(this, (function (exports, ml4f, tf) { 'use strict';

    function inIFrame() {
        try {
            return typeof window !== "undefined" && window.self !== window.top;
        }
        catch (e) {
            return typeof window !== "undefined";
        }
    }
    const CHANGE = 'change';
    const READ = "read";
    const MESSAGE_PACKET = "messagepacket";
    const HIDDEN = "hidden";
    const SHOWN = "shown";
    const CONNECT = "connect";
    const accelSample = `export function _sample() {
    return [
        input.acceleration(Dimension.X),
        input.acceleration(Dimension.Y),
        input.acceleration(Dimension.Z)
    ]
}`;
    class MakeCodeEditorExtensionClient {
        constructor() {
            this.pendingCommands = {};
            this.extensionId = inIFrame() ? window.location.hash.substr(1) : undefined;
            this._connected = false;
            this._visible = false;
            this.nextRequestId = 1;
            this.handleMessage = this.handleMessage.bind(this);
            window.addEventListener("message", this.handleMessage, false);
            // notify parent that we're ready
            this.init();
        }
        emit(id, arg) {
            console.log("EMIT", id, { arg });
        }
        log(msg) {
            console.log(`ML4F-PXT: ${msg}`);
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
        setVisible(vis) {
            if (this._visible !== vis) {
                this._visible = vis;
                this.emit(CHANGE);
            }
        }
        mkRequest(resolve, reject, action, body) {
            const id = "ml_" + this.nextRequestId++;
            this.pendingCommands[id] = { action, resolve, reject };
            return {
                type: "pxtpkgext",
                action,
                extId: this.extensionId,
                response: true,
                id,
                body
            };
        }
        sendRequest(action, body) {
            this.log(`send ${action}`);
            if (!this.extensionId)
                return Promise.resolve(undefined);
            return new Promise((resolve, reject) => {
                const msg = this.mkRequest(resolve, reject, action, body);
                window.parent.postMessage(msg, "*");
            });
        }
        handleMessage(ev) {
            const msg = ev.data;
            if ((msg === null || msg === void 0 ? void 0 : msg.type) !== "pxtpkgext")
                return;
            if (!msg.id) {
                switch (msg.event) {
                    case "extinit":
                        this.log(`init`);
                        this._target = msg.target;
                        this._connected = true;
                        this.emit(CONNECT);
                        this.emit(CHANGE);
                        break;
                    case "extloaded":
                        this.log(`loaded`);
                        break;
                    case "extshown":
                        this.setVisible(true);
                        this.refresh();
                        this.emit(SHOWN);
                        this.emit(CHANGE);
                        break;
                    case "exthidden":
                        this.setVisible(false);
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
        async init() {
            this.log(`initializing`);
            await this.sendRequest('extinit');
            this.log(`connected`);
            await this.refresh();
        }
        async refresh() {
            this.log(`refresh`);
            const r = await this.read();
        }
        async read() {
            if (!this.extensionId) {
                const r = {};
                this.emit(READ, r);
                return r;
            }
            else {
                const resp = await this.sendRequest('extreadcode');
                return resp;
            }
        }
        async readUser() {
            await this.sendRequest('extusercode');
        }
        async write(code, json, jres, dependencies) {
            if (!this.extensionId) {
                // Write to local storage instead
                this.emit('written', undefined);
            }
            else {
                await this.sendRequest('extwritecode', {
                    code: code || undefined,
                    json: json || undefined,
                    jres: jres || undefined,
                    dependencies
                });
            }
        }
        async queryPermission() {
            await this.sendRequest('extquerypermission');
        }
        async requestPermission(console) {
            await this.sendRequest('extrequestpermission', {
                console
            });
        }
        async dataStreamConsole(console) {
            await this.sendRequest('extdatastream', {
                console
            });
        }
        async dataStreamMessages(messages) {
            await this.sendRequest('extdatastream', {
                messages
            });
        }
    }
    async function start() {
        tf.setBackend("cpu");
        const options = {
            f16: true
        };
        const pxtClient = new MakeCodeEditorExtensionClient();
        const maindiv = document.createElement("div");
        maindiv.style.background = "white";
        document.body.appendChild(maindiv);
        const status = div("");
        maindiv.append(status);
        setStatus("waiting for model file");
        const d = div("Drop TF.JS model file here");
        d.style.padding = "2em";
        d.style.margin = "1em 0em";
        d.style.border = "1px dotted gray";
        maindiv.append(d);
        addCheckbox("f16", "Use float16 type");
        const dropbox = maindiv;
        dropbox.addEventListener("dragenter", stopEv, false);
        dropbox.addEventListener("dragover", stopEv, false);
        dropbox.addEventListener("drop", e => {
            setStatus("reading model");
            stopEv(e);
            const file = e.dataTransfer.files.item(0);
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const mod = JSON.parse(e.target.result);
                    await compileModel(mod, file.name);
                }
                catch (e) {
                    console.error(e.stack);
                    setError(e.message);
                }
            };
            reader.readAsText(file);
        }, false);
        function shapeElements(shape) {
            let res = 1;
            for (const s of shape)
                if (s != null)
                    res *= s;
            return res;
        }
        function toCamelCase(name) {
            return name.replace(/(^|( +))(.)/g, (_0, _1, _2, l) => l.toUpperCase());
        }
        async function compileModel(mod, fileName) {
            const name = mod.name || fileName;
            const ma = ml4f.loadFlatJSONModel(mod);
            const m = await tf.loadLayersModel({ load: () => Promise.resolve(ma) });
            const inpTen = m.getInputAt(0);
            const numClasses = shapeElements(m.getOutputAt(0).shape);
            const labels = (mod.labels || []).slice();
            while (labels.length > numClasses)
                labels.pop();
            while (labels.length < numClasses)
                labels.push("class " + labels.length);
            const inputShape = inpTen.shape;
            const samplingPeriod = mod.inputInterval || 100;
            setStatus("compiling..."); // can't see that...
            const res = await ml4f.compileModelAndFullValidate(m, {
                verbose: false,
                includeTest: true,
                float16weights: options.f16,
                optimize: true
            });
            setStatus("compiled!");
            const shape2 = inputShape.filter(v => v != null);
            const samplesInWindow = shape2.shift();
            const elementsInSample = shapeElements(shape2);
            let code = `// model: ${name}; input: ${JSON.stringify(inputShape)}; sampling at: ${samplingPeriod}ms\n` +
                `// ${res.memInfo}\n` +
                `// ${res.timeInfo}\n`;
            code += "export const enum MLEvent {\n";
            let idx = 0;
            for (let lbl of labels) {
                lbl = lbl.replace(/_/g, " ");
                code += `    //% block="${lbl}"\n`;
                code += `    ${toCamelCase(lbl)} = ${idx},\n`;
                idx++;
            }
            code += `}\n\n`;
            code += `namespace ml {\n`;
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
            `;
            code += "\n" + accelSample + "\n"; // TODO
            code +=
                `export const _model = new ml4f.Model(\n` +
                    "hex`";
            for (let i = 0; i < res.machineCode.length; ++i) {
                code += ("0" + res.machineCode[i].toString(16)).slice(-2);
                if ((i + 3) % 32 == 0)
                    code += "\n";
            }
            code += "`);\n";
            code += "\n} // namespace ml\n";
            console.log(code.replace(/([a-f0-9]{64}\n)+/, "..."));
            await pxtClient.write(code);
            setStatus("done; you can Go back now");
        }
        function stopEv(e) {
            e.stopPropagation();
            e.preventDefault();
        }
        function div(text) {
            const d = document.createElement("div");
            d.textContent = text;
            return d;
        }
        function setError(msg) {
            status.style.color = "red";
            status.textContent = "Error: " + msg;
        }
        function setStatus(msg) {
            status.style.color = "green";
            status.textContent = msg;
        }
        function addCheckbox(field, name) {
            const lbl = document.createElement("label");
            lbl.textContent = name;
            const box = document.createElement("input");
            lbl.prepend(box);
            box.type = "checkbox";
            box.checked = !!options[field];
            box.addEventListener('change', () => {
                if (box.checked)
                    options[field] = !!box.checked;
            });
            maindiv.appendChild(lbl);
        }
    }

    exports.MESSAGE_PACKET = MESSAGE_PACKET;
    exports.MakeCodeEditorExtensionClient = MakeCodeEditorExtensionClient;
    exports.READ = READ;
    exports.inIFrame = inIFrame;
    exports.start = start;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=pxtml4f.js.map
