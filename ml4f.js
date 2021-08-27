(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.ml4f = global.ml4f || {}, global.tf));
}(this, (function (exports, tf) { 'use strict';

    /// based on: Fast Half Float Conversions, Jeroen van der Zijp, link: http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
    const basetable = new Uint16Array(512);
    const shifttable = new Uint8Array(512);
    const mantissatable = new Uint32Array(2048);
    const offsettable = new Uint16Array(64);
    const exponenttable = new Uint32Array(64);
    let inited = false;
    function init() {
        inited = true;
        for (let i = 0; i < 256; ++i) {
            const e = i - 127;
            if (e < -24) { // Very small numbers map to zero
                basetable[i | 0x000] = 0x0000;
                basetable[i | 0x100] = 0x8000;
                shifttable[i | 0x000] = 24;
                shifttable[i | 0x100] = 24;
            }
            else if (e < -14) { // Small numbers map to denorms
                basetable[i | 0x000] = (0x0400 >> (-e - 14));
                basetable[i | 0x100] = (0x0400 >> (-e - 14)) | 0x8000;
                shifttable[i | 0x000] = -e - 1;
                shifttable[i | 0x100] = -e - 1;
            }
            else if (e <= 15) { // Normal numbers just lose precision
                basetable[i | 0x000] = ((e + 15) << 10);
                basetable[i | 0x100] = ((e + 15) << 10) | 0x8000;
                shifttable[i | 0x000] = 13;
                shifttable[i | 0x100] = 13;
            }
            else if (e < 128) { // Large numbers map to Infinity
                basetable[i | 0x000] = 0x7C00;
                basetable[i | 0x100] = 0xFC00;
                shifttable[i | 0x000] = 24;
                shifttable[i | 0x100] = 24;
            }
            else { // Infinity and NaN's stay Infinity and NaN's
                basetable[i | 0x000] = 0x7C00;
                basetable[i | 0x100] = 0xFC00;
                shifttable[i | 0x000] = 13;
                shifttable[i | 0x100] = 13;
            }
        }
        for (let i = 1; i < 2048; ++i) {
            if (i < 1024)
                mantissatable[i] = convertmantissa(i);
            else
                mantissatable[i] = 0x38000000 + ((i - 1024) << 13);
        }
        exponenttable[32] = 0x80000000;
        exponenttable[31] = 0x47800000;
        exponenttable[63] = 0xC7800000;
        for (let i = 1; i <= 30; ++i)
            exponenttable[i] = i << 23;
        for (let i = 33; i <= 62; ++i)
            exponenttable[i] = 0x80000000 + ((i - 32) << 23);
        for (let i = 1; i < offsettable.length; ++i)
            offsettable[i] = 1024;
        offsettable[32] = 0;
        function convertmantissa(i) {
            let m = i << 13; // Zero pad mantissa bits
            let e = 0; // Zero exponent
            while (!(m & 0x00800000)) { // While not normalized
                e -= 0x00800000; // Decrement exponent (1<<23)
                m <<= 1; // Shift mantissa
            }
            m &= ~0x00800000; // Clear leading 1 bit
            e += 0x38800000; // Adjust bias ((127-14)<<23)
            return (m | e) >>> 0; // Return combined number
        }
    }
    function float32ToUInt32(v) {
        const buf = new Float32Array(1);
        buf[0] = v;
        return new Uint32Array(buf.buffer)[0];
    }
    function float16toUInt16(v) {
        const f = float32ToUInt32(v);
        if (!inited)
            init();
        return basetable[(f >> 23) & 0x1ff] | ((f & 0x007fffff) >> shifttable[(f >> 23) & 0x1ff]);
    }
    function float16AsUintToFloat(h) {
        if (!inited)
            init();
        const tmp = mantissatable[offsettable[h >> 10] + (h & 0x3ff)] + exponenttable[h >> 10];
        const buf = new Uint32Array(1);
        buf[0] = tmp;
        return new Float32Array(buf.buffer)[0];
    }
    function testFloatConv() {
        for (let i = 0; i < 30000; ++i) {
            test(i);
            test(-i);
            test(1 / i);
            test(-1 / i);
            test(1 / (i * 100));
            test(-1 / (i * 100));
        }
        function test(v) {
            const u = float16toUInt16(v) & 0xffff;
            const v2 = float16AsUintToFloat(u);
            const d = Math.min(10000 * Math.abs(v - v2), Math.abs((v - v2) / v));
            if (d > 0.002) {
                throw new Error(`fail: ${v} -> ${u} -> ${v2} (dd=${v - v2} d=${d})`);
            }
        }
    }

    function assert(cond, msg = "Assertion failed") {
        if (!cond) {
            debugger;
            throw new Error(msg);
        }
    }
    function userError(msg) {
        let e = new Error(msg);
        e.isUserError = true;
        throw e;
    }
    function lookup(m, key) {
        if (m.hasOwnProperty(key))
            return m[key];
        return null;
    }
    function oops(msg = "OOPS") {
        debugger;
        throw new Error(msg);
    }
    function endsWith(str, suffix) {
        if (str.length < suffix.length)
            return false;
        if (suffix.length == 0)
            return true;
        return str.slice(-suffix.length) == suffix;
    }
    function startsWith(str, prefix) {
        if (str.length < prefix.length)
            return false;
        if (prefix.length == 0)
            return true;
        return str.slice(0, prefix.length) == prefix;
    }
    function iterMap(m, f) {
        Object.keys(m).forEach(k => f(k, m[k]));
    }
    function mapMap(m, f) {
        let r = {};
        Object.keys(m).forEach(k => r[k] = f(k, m[k]));
        return r;
    }
    function pushRange(trg, src) {
        for (let i = 0; i < src.length; ++i)
            trg.push(src[i]);
    }
    // TS gets lost in type inference when this is passed an array
    function concatArrayLike(arrays) {
        return concat(arrays);
    }
    function concat(arrays) {
        let r = [];
        for (let i = 0; i < arrays.length; ++i) {
            pushRange(r, arrays[i]);
        }
        return r;
    }
    function range(len) {
        let r = [];
        for (let i = 0; i < len; ++i)
            r.push(i);
        return r;
    }
    let seed = 13 * 0x1000193;
    function seedRandom(v) {
        seed = (v * 0x1000193) >>> 0;
    }
    function randomUint32() {
        let x = seed;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        x >>>= 0;
        seed = x;
        return x;
    }
    function randomInclusive(min, max) {
        return min + randomUint32() % (max - min + 1);
    }
    function randomPermute(arr) {
        for (let i = 0; i < arr.length; ++i) {
            let j = randomUint32() % arr.length;
            let tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
    function randomPick(arr) {
        if (arr.length == 0)
            return null;
        return arr[randomUint32() % arr.length];
    }
    function randomUFloat() {
        return randomUint32() / 4294967296;
    }
    function randomSFloat() {
        return 2 * randomUFloat() - 1;
    }
    function flatClone(v) {
        const s = v;
        const d = {};
        for (const k of Object.keys(s)) {
            d[k] = s[k];
        }
        return d;
    }

    let debug = false;
    function lf(fmt, ...args) {
        return fmt.replace(/{(\d+)}/g, (match, index) => args[+index]);
    }
    let badNameError = emitErr("opcode name doesn't match", "<name>");
    // An Instruction represents an instruction class with meta-variables
    // that should be substituted given an actually line (Line) of assembly
    // Thus, the Instruction helps us parse a sequence of tokens in a Line
    // as well as extract the relevant values to substitute for the meta-variables.
    // The Instruction also knows how to convert the particular instance into
    // machine code (EmitResult)
    class Instruction {
        constructor(ei, format, opcode, mask, is32bit) {
            this.opcode = opcode;
            this.mask = mask;
            this.is32bit = is32bit;
            this.canBeShared = false;
            assert((opcode & mask) == opcode);
            this.ei = ei;
            this.code = format.replace(/\s+/g, " ");
            this.friendlyFmt = format.replace(/\$\w+/g, m => {
                if (this.ei.encoders[m])
                    return this.ei.encoders[m].pretty;
                return m;
            });
            let words = tokenize(format);
            this.name = words[0];
            this.args = words.slice(1);
        }
        emit(ln) {
            const tokens = ln.words;
            if (tokens[0] != this.name)
                return badNameError;
            let r = this.opcode;
            let j = 1;
            let stack = 0;
            let numArgs = [];
            let labelName = null;
            let bit32_value = null;
            let bit32_actual = null;
            const isSpecial32 = this.ei.is32bit(this) && !this.is32bit;
            for (let i = 0; i < this.args.length; ++i) {
                let formal = this.args[i];
                let actual = tokens[j++];
                if (formal[0] == "$") {
                    let enc = this.ei.encoders[formal];
                    let v = null;
                    if (enc.isRegister) {
                        v = this.ei.registerNo(actual, enc);
                        if (v == null)
                            return emitErr("expecting register name", actual);
                        if (this.ei.isPush(this.opcode)) // push
                            stack++;
                        else if (this.ei.isPop(this.opcode)) // pop
                            stack--;
                    }
                    else if (enc.isImmediate) {
                        actual = actual.replace(/^#/, "");
                        v = ln.bin.parseOneInt(actual);
                        if (v == null) {
                            return emitErr("expecting number", actual);
                        }
                        else {
                            // explicit manipulation of stack pointer (SP)
                            // ARM only
                            if (this.ei.isAddSP(this.opcode))
                                stack = -(v / this.ei.wordSize());
                            else if (this.ei.isSubSP(this.opcode))
                                stack = (v / this.ei.wordSize());
                        }
                    }
                    else if (enc.isRegList) {
                        // register lists are ARM-specific - this code not used in AVR 
                        if (actual != "{")
                            return emitErr("expecting {", actual);
                        v = 0;
                        while (tokens[j] != "}") {
                            actual = tokens[j++];
                            if (!actual)
                                return emitErr("expecting }", tokens[j - 2]);
                            let no = this.ei.registerNo(actual, enc);
                            if (no == null)
                                return emitErr("expecting register name", actual);
                            if (v & (1 << no))
                                return emitErr("duplicate register name", actual);
                            v |= (1 << no);
                            if (this.ei.isPush(this.opcode)) // push
                                stack++;
                            else if (this.ei.isPop(this.opcode)) // pop
                                stack--;
                            if (tokens[j] == ",")
                                j++;
                        }
                        actual = tokens[j++]; // skip close brace
                    }
                    else if (enc.isLabel) {
                        actual = actual.replace(/^#/, "");
                        if (/^[+-]?\d+$/.test(actual)) {
                            v = parseInt(actual, 10);
                            labelName = "rel" + v;
                        }
                        else if (/^0x[0-9a-fA-F]+$/.test(actual)) {
                            v = parseInt(actual, 16);
                            labelName = "abs" + v;
                        }
                        else {
                            let lbloff = 0;
                            if (actual.indexOf("+") > 0) {
                                const m = /(.*)\+(\d+)$/.exec(actual);
                                if (m) {
                                    actual = m[1];
                                    lbloff = parseInt(m[2]);
                                }
                            }
                            labelName = actual;
                            v = this.ei.getAddressFromLabel(ln.bin, this, actual, enc.isWordAligned);
                            if (v == null) {
                                if (ln.bin.finalEmit)
                                    return emitErr("unknown label", actual);
                                else
                                    // just need some value when we are 
                                    // doing some pass other than finalEmit
                                    v = 8; // needs to be divisible by 4 etc
                            }
                            v += lbloff;
                        }
                        if (isSpecial32) {
                            // console.log(actual + " " + v.toString())
                            bit32_value = v;
                            bit32_actual = actual;
                            continue;
                        }
                    }
                    else {
                        oops();
                    }
                    if (v == null)
                        return emitErr("didn't understand it", actual); // shouldn't happen
                    numArgs.push(v);
                    v = enc.encode(v);
                    // console.log("enc(v) = ",v)
                    if (v == null)
                        return emitErr("argument out of range or mis-aligned", actual);
                    assert((r & v) == 0);
                    r |= v;
                }
                else if (formal == actual) ;
                else {
                    return emitErr("expecting " + formal, actual);
                }
            }
            if (tokens[j])
                return emitErr("trailing tokens", tokens[j]);
            if (isSpecial32)
                return this.ei.emit32(r, bit32_value, ln.bin.normalizeExternalLabel(bit32_actual));
            if (this.is32bit)
                return {
                    opcode: ((r >> 16) & 0xffff) | 0x8000,
                    opcode2: (r >> 0) & 0xffff,
                    stack,
                    numArgs,
                    labelName: ln.bin.normalizeExternalLabel(labelName)
                };
            return {
                stack,
                opcode: r,
                numArgs,
                labelName: ln.bin.normalizeExternalLabel(labelName)
            };
        }
        toString() {
            return this.friendlyFmt;
        }
    }
    // represents a line of assembly from a file
    class Line {
        constructor(bin, text) {
            this.bin = bin;
            this.text = text;
        }
        getOpExt() {
            return this.instruction ? this.instruction.code : "";
        }
        getOp() {
            return this.instruction ? this.instruction.name : "";
        }
        update(s) {
            this.bin.peepOps++;
            s = s.replace(/^\s*/, "");
            if (!s)
                this.bin.peepDel++;
            if (s)
                s += "      ";
            s = "    " + s;
            this.text = s + "; WAS: " + this.text.trim();
            this.instruction = null;
            this.numArgs = null;
            this.words = tokenize(s) || [];
            if (this.words.length == 0)
                this.type = "empty";
            else if (this.words[0][0] == "@")
                this.type = "directive";
        }
    }
    // File is the center of the action: parsing a file into a sequence of Lines
    // and also emitting the binary (buf)
    class File {
        constructor(ei) {
            this.baseOffset = 0;
            this.checkStack = true;
            this.inlineMode = false;
            this.normalizeExternalLabel = (n) => n;
            this.currLineNo = 0;
            this.scope = "";
            this.scopeId = 0;
            this.errors = [];
            this.labels = {};
            this.equs = {};
            this.stackpointers = {};
            this.stack = 0;
            this.commPtr = 0;
            this.peepOps = 0;
            this.peepDel = 0;
            this.peepCounts = {};
            this.stats = "";
            this.throwOnError = false;
            this.disablePeepHole = false;
            this.stackAtLabel = {};
            this.currLine = new Line(this, "<start>");
            this.currLine.lineNo = 0;
            this.ei = ei;
            this.ei.file = this;
        }
        emitShort(op) {
            assert(0 <= op && op <= 0xffff);
            this.buf.push(op);
        }
        emitOpCode(op) {
            this.emitShort(op);
        }
        location() {
            // store one short (2 bytes) per buf location
            return this.buf.length * 2;
        }
        pc() {
            return this.location() + this.baseOffset;
        }
        // parsing of an "integer", well actually much more than 
        // just that
        parseOneInt(s) {
            if (!s)
                return null;
            // fast path
            if (/^\d+$/.test(s))
                return parseInt(s, 10);
            const minP = s.indexOf("-");
            if (minP > 0)
                return this.parseOneInt(s.slice(0, minP)) - this.parseOneInt(s.slice(minP + 1));
            let mul = 1;
            // recursive-descent parsing of multiplication
            if (s.indexOf("*") >= 0) {
                let m = null;
                while (m = /^([^\*]*)\*(.*)$/.exec(s)) {
                    let tmp = this.parseOneInt(m[1]);
                    if (tmp == null)
                        return null;
                    mul *= tmp;
                    s = m[2];
                }
            }
            if (s[0] == "-") {
                mul *= -1;
                s = s.slice(1);
            }
            else if (s[0] == "+") {
                s = s.slice(1);
            }
            // decimal encoding; fast-ish path
            if (/^\d+$/.test(s))
                return mul * parseInt(s, 10);
            // allow or'ing of 1 to least-signficant bit
            if (endsWith(s, "|1")) {
                return this.parseOneInt(s.slice(0, s.length - 2)) | 1;
            }
            // allow subtracting 1 too
            if (endsWith(s, "-1")) {
                return this.parseOneInt(s.slice(0, s.length - 2)) - 1;
            }
            // allow adding 1 too
            if (endsWith(s, "+1")) {
                return this.parseOneInt(s.slice(0, s.length - 2)) + 1;
            }
            let shm = /(.*)>>(\d+)$/.exec(s);
            if (shm) {
                let left = this.parseOneInt(shm[1]);
                let mask = this.baseOffset & ~0xffffff;
                left &= ~mask;
                return left >> parseInt(shm[2]);
            }
            let v = null;
            // handle hexadecimal and binary encodings
            if (s[0] == "0") {
                if (s[1] == "x" || s[1] == "X") {
                    let m = /^0x([a-f0-9]+)$/i.exec(s);
                    if (m)
                        v = parseInt(m[1], 16);
                }
                else if (s[1] == "b" || s[1] == "B") {
                    let m = /^0b([01]+)$/i.exec(s);
                    if (m)
                        v = parseInt(m[1], 2);
                }
            }
            // stack-specific processing
            // more special characters to handle
            if (s.indexOf("@") >= 0) {
                let m = /^(\w+)@(-?\d+)$/.exec(s);
                if (m) {
                    if (mul != 1)
                        this.directiveError(lf("multiplication not supported with saved stacks"));
                    if (this.stackpointers.hasOwnProperty(m[1])) {
                        // console.log(m[1] + ": " + this.stack + " " + this.stackpointers[m[1]] + " " + m[2])
                        v = this.ei.wordSize() * this.ei.computeStackOffset(m[1], this.stack - this.stackpointers[m[1]] + parseInt(m[2]));
                        // console.log(v)
                    }
                    else
                        this.directiveError(lf("saved stack not found"));
                }
                m = /^(.*)@(hi|lo|fn)$/.exec(s);
                if (m && this.looksLikeLabel(m[1])) {
                    v = this.lookupLabel(m[1], true);
                    if (v != null) {
                        if (m[2] == "fn") {
                            v = this.ei.toFnPtr(v, this.baseOffset, m[1]);
                        }
                        else {
                            v >>= 1;
                            if (0 <= v && v <= 0xffff) {
                                if (m[2] == "hi")
                                    v = (v >> 8) & 0xff;
                                else if (m[2] == "lo")
                                    v = v & 0xff;
                                else
                                    oops();
                            }
                            else {
                                this.directiveError(lf("@hi/lo out of range"));
                                v = null;
                            }
                        }
                    }
                }
            }
            if (v == null && this.looksLikeLabel(s)) {
                v = this.lookupLabel(s, true);
                if (v != null) {
                    if (this.ei.postProcessRelAddress(this, 1) == 1)
                        v += this.baseOffset;
                }
            }
            if (v == null || isNaN(v))
                return null;
            return v * mul;
        }
        looksLikeLabel(name) {
            if (/^(r\d|pc|sp|lr)$/i.test(name))
                return false;
            return /^[\.a-zA-Z_][\.:\w+]*$/.test(name);
        }
        scopedName(name) {
            if (name[0] == "." && this.scope)
                return this.scope + "$" + name;
            else
                return name;
        }
        lookupLabel(name, direct = false) {
            let v = null;
            let scoped = this.scopedName(name);
            if (this.labels.hasOwnProperty(scoped)) {
                v = this.labels[scoped];
                v = this.ei.postProcessRelAddress(this, v);
            }
            else if (this.lookupExternalLabel) {
                v = this.lookupExternalLabel(name);
                if (v != null) {
                    v = this.ei.postProcessAbsAddress(this, v);
                }
            }
            if (v == null && this.equs.hasOwnProperty(scoped)) {
                v = this.equs[scoped];
                // no post-processing
            }
            if (v == null && direct) {
                if (this.finalEmit) {
                    this.directiveError(lf("unknown label: {0}", name));
                }
                else
                    // use a number over 1 byte
                    v = 11111;
            }
            return v;
        }
        align(n) {
            assert(n == 2 || n == 4 || n == 8 || n == 16);
            while (this.location() % n != 0)
                this.emitOpCode(0);
        }
        pushError(msg, hints = "") {
            let err = {
                scope: this.scope,
                message: lf("  -> Line {2} ('{1}'), error: {0}\n{3}", msg, this.currLine.text, this.currLine.lineNo, hints),
                lineNo: this.currLine.lineNo,
                line: this.currLine.text,
                coremsg: msg,
                hints: hints
            };
            this.errors.push(err);
            if (this.throwOnError)
                throw new Error(err.message);
        }
        directiveError(msg) {
            this.pushError(msg);
            // this.pushError(lf("directive error: {0}", msg))
        }
        emitString(l, utf16 = false) {
            function byteAt(s, i) { return (s.charCodeAt(i) || 0) & 0xff; }
            let m = /^\s*([\w\.]+\s*:\s*)?.\w+\s+(".*")\s*$/.exec(l);
            let s;
            if (!m || null == (s = parseString(m[2]))) {
                this.directiveError(lf("expecting string"));
            }
            else {
                this.align(2);
                if (utf16) {
                    for (let i = 0; i < s.length; i++) {
                        this.emitShort(s.charCodeAt(i));
                    }
                }
                else {
                    // s.length + 1 to NUL terminate
                    for (let i = 0; i < s.length + 1; i += 2) {
                        this.emitShort((byteAt(s, i + 1) << 8) | byteAt(s, i));
                    }
                }
            }
        }
        parseNumber(words) {
            let v = this.parseOneInt(words.shift());
            if (v == null)
                return null;
            return v;
        }
        parseNumbers(words) {
            words = words.slice(1);
            let nums = [];
            while (true) {
                let n = this.parseNumber(words);
                if (n == null) {
                    this.directiveError(lf("cannot parse number at '{0}'", words[0]));
                    break;
                }
                else
                    nums.push(n);
                if (words[0] == ",") {
                    words.shift();
                    if (words[0] == null)
                        break;
                }
                else if (words[0] == null) {
                    break;
                }
                else {
                    this.directiveError(lf("expecting number, got '{0}'", words[0]));
                    break;
                }
            }
            return nums;
        }
        emitSpace(words) {
            let nums = this.parseNumbers(words);
            if (nums.length == 1)
                nums.push(0);
            if (nums.length != 2)
                this.directiveError(lf("expecting one or two numbers"));
            else if (nums[0] % 2 != 0)
                this.directiveError(lf("only even space supported"));
            else {
                let f = nums[1] & 0xff;
                f = f | (f << 8);
                for (let i = 0; i < nums[0]; i += 2)
                    this.emitShort(f);
            }
        }
        emitBytes(words) {
            let nums = this.parseNumbers(words);
            if (nums.length % 2 != 0) {
                this.directiveError(".bytes needs an even number of arguments");
                nums.push(0);
            }
            for (let i = 0; i < nums.length; i += 2) {
                let n0 = nums[i];
                let n1 = nums[i + 1];
                if (0 <= n0 && n1 <= 0xff &&
                    0 <= n1 && n0 <= 0xff)
                    this.emitShort((n0 & 0xff) | ((n1 & 0xff) << 8));
                else
                    this.directiveError(lf("expecting uint8"));
            }
        }
        emitHex(words) {
            words.slice(1).forEach(w => {
                if (w == ",")
                    return;
                // TODO: why 4 and not 2?
                if (w.length % 4 != 0)
                    this.directiveError(".hex needs an even number of bytes");
                else if (!/^[a-f0-9]+$/i.test(w))
                    this.directiveError(".hex needs a hex number");
                else
                    for (let i = 0; i < w.length; i += 4) {
                        let n = parseInt(w.slice(i, i + 4), 16);
                        n = ((n & 0xff) << 8) | ((n >> 8) & 0xff);
                        this.emitShort(n);
                    }
            });
        }
        emitFloats(words) {
            words.slice(1).forEach(w => {
                if (w == ",")
                    return;
                const v = parseFloat(w);
                if (isNaN(v))
                    this.directiveError("invalid .float");
                const n = float32ToUInt32(v);
                this.emitShort(n & 0xffff);
                this.emitShort((n >> 16) & 0xffff);
            });
        }
        emitFloats16(words) {
            words.slice(1).forEach(w => {
                if (w == ",")
                    return;
                const v = parseFloat(w);
                if (isNaN(v))
                    this.directiveError("invalid .float16");
                const n = float16toUInt16(v);
                this.emitShort(n & 0xffff);
            });
        }
        handleDirective(l) {
            let words = l.words;
            let expectOne = () => {
                if (words.length != 2)
                    this.directiveError(lf("expecting one argument"));
            };
            let num0;
            switch (words[0]) {
                case ".ascii":
                case ".asciz":
                case ".string":
                    this.emitString(l.text);
                    break;
                case ".utf16":
                    this.emitString(l.text, true);
                    break;
                case ".align":
                    expectOne();
                    num0 = this.parseOneInt(words[1]);
                    if (num0 != null) {
                        if (num0 == 0)
                            return;
                        if (num0 <= 4) {
                            this.align(1 << num0);
                        }
                        else {
                            this.directiveError(lf("expecting 1, 2, 3 or 4 (for 2, 4, 8, or 16 byte alignment)"));
                        }
                    }
                    else
                        this.directiveError(lf("expecting number"));
                    break;
                case ".balign":
                    expectOne();
                    num0 = this.parseOneInt(words[1]);
                    if (num0 != null) {
                        if (num0 == 1)
                            return;
                        if (num0 == 2 || num0 == 4 || num0 == 8 || num0 == 16) {
                            this.align(num0);
                        }
                        else {
                            this.directiveError(lf("expecting 2, 4, 8, or 16"));
                        }
                    }
                    else
                        this.directiveError(lf("expecting number"));
                    break;
                case ".p2align":
                    expectOne();
                    num0 = this.parseOneInt(words[1]);
                    if (num0 != null) {
                        this.align(1 << num0);
                    }
                    else
                        this.directiveError(lf("expecting number"));
                    break;
                case ".byte":
                    this.emitBytes(words);
                    break;
                case ".hex":
                    this.emitHex(words);
                    break;
                case ".float":
                    this.emitFloats(words);
                    break;
                case ".float16":
                    this.emitFloats16(words);
                    break;
                case ".hword":
                case ".short":
                case ".2bytes":
                    this.parseNumbers(words).forEach(n => {
                        // we allow negative numbers
                        if (-0x8000 <= n && n <= 0xffff)
                            this.emitShort(n & 0xffff);
                        else
                            this.directiveError(lf("expecting int16"));
                    });
                    break;
                case ".word":
                case ".4bytes":
                case ".long":
                    // TODO: a word is machine-dependent (16-bit for AVR, 32-bit for ARM)
                    this.parseNumbers(words).forEach(n => {
                        // we allow negative numbers
                        if (-0x80000000 <= n && n <= 0xffffffff) {
                            this.emitShort(n & 0xffff);
                            this.emitShort((n >> 16) & 0xffff);
                        }
                        else {
                            this.directiveError(lf("expecting int32"));
                        }
                    });
                    break;
                case ".skip":
                case ".space":
                    this.emitSpace(words);
                    break;
                case ".set":
                case ".equ":
                    if (!/^\w+$/.test(words[1]))
                        this.directiveError(lf("expecting name"));
                    const nums = this.parseNumbers(words.slice(words[2] == "," || words[2] == "="
                        ? 2 : 1));
                    if (nums.length != 1)
                        this.directiveError(lf("expecting one value"));
                    if (this.equs[words[1]] !== undefined &&
                        this.equs[words[1]] != nums[0])
                        this.directiveError(lf("redefinition of {0}", words[1]));
                    this.equs[words[1]] = nums[0];
                    break;
                case ".startaddr":
                    if (this.location())
                        this.directiveError(lf(".startaddr can be only be specified at the beginning of the file"));
                    expectOne();
                    this.baseOffset = this.parseOneInt(words[1]);
                    break;
                // The usage for this is as follows:
                // push {...}
                // @stackmark locals   ; locals := sp
                // ... some push/pops ...
                // ldr r0, [sp, locals@3] ; load local number 3
                // ... some push/pops ...
                // @stackempty locals ; expect an empty stack here
                case "@stackmark":
                    expectOne();
                    this.stackpointers[words[1]] = this.stack;
                    break;
                case "@stackempty":
                    if (this.checkStack) {
                        if (this.stackpointers[words[1]] == null)
                            this.directiveError(lf("no such saved stack"));
                        else if (this.stackpointers[words[1]] != this.stack)
                            this.directiveError(lf("stack mismatch"));
                    }
                    break;
                case "@scope":
                    this.scope = words[1] || "";
                    this.currLineNo = this.scope ? 0 : this.realCurrLineNo;
                    break;
                case ".syntax":
                case "@nostackcheck":
                    this.checkStack = false;
                    break;
                case "@dummystack":
                    expectOne();
                    this.stack += this.parseOneInt(words[1]);
                    break;
                case ".section":
                case ".global":
                    this.stackpointers = {};
                    this.stack = 0;
                    this.scope = "$S" + this.scopeId++;
                    break;
                case ".comm": {
                    words = words.filter(x => x != ",");
                    words.shift();
                    let sz = this.parseOneInt(words[1]);
                    let align = 0;
                    if (words[2])
                        align = this.parseOneInt(words[2]);
                    else
                        align = 4; // not quite what AS does...
                    let val = this.lookupLabel(words[0]);
                    if (val == null) {
                        if (!this.commPtr) {
                            this.commPtr = this.lookupExternalLabel("_pxt_comm_base") || 0;
                            if (!this.commPtr)
                                this.directiveError(lf("PXT_COMM_BASE not defined"));
                        }
                        while (this.commPtr & (align - 1))
                            this.commPtr++;
                        this.labels[this.scopedName(words[0])] = this.commPtr - this.baseOffset;
                        this.commPtr += sz;
                    }
                    break;
                }
                case ".arch":
                case ".thumb":
                case ".file":
                case ".text":
                case ".cpu":
                case ".fpu":
                case ".eabi_attribute":
                case ".code":
                case ".thumb_func":
                case ".type":
                case ".fnstart":
                case ".save":
                case ".size":
                case ".fnend":
                case ".pad":
                case ".globl": // TODO might need this one
                case ".local":
                    break;
                case "@":
                    // @ sp needed
                    break;
                default:
                    if (/^\.cfi_/.test(words[0])) ;
                    else {
                        this.directiveError(lf("unknown directive"));
                    }
                    break;
            }
        }
        handleOneInstruction(ln, instr) {
            let op = instr.emit(ln);
            if (!op.error) {
                this.stack += op.stack;
                if (this.checkStack && this.stack < 0)
                    this.pushError(lf("stack underflow"));
                ln.location = this.location();
                ln.opcode = op.opcode;
                ln.stack = op.stack;
                this.emitOpCode(op.opcode);
                if (op.opcode2 != null)
                    this.emitOpCode(op.opcode2);
                if (op.opcode3 != null)
                    this.emitOpCode(op.opcode3);
                ln.instruction = instr;
                ln.numArgs = op.numArgs;
                return true;
            }
            return false;
        }
        handleInstruction(ln) {
            if (ln.instruction) {
                if (this.handleOneInstruction(ln, ln.instruction))
                    return;
            }
            const getIns = (n) => this.ei.instructions.hasOwnProperty(n) ? this.ei.instructions[n] : [];
            let ins = getIns(ln.words[0]);
            for (let i = 0; i < ins.length; ++i) {
                if (this.handleOneInstruction(ln, ins[i]))
                    return;
            }
            const condless = this.ei.stripCondition(ln.words[0]);
            if (condless) {
                ins = getIns(condless);
                if (ins.length > 0) {
                    ln.words[0] = condless;
                    for (let i = 0; i < ins.length; ++i) {
                        if (this.handleOneInstruction(ln, ins[i]))
                            return;
                    }
                }
            }
            let w0 = ln.words[0].toLowerCase().replace(/s$/, "").replace(/[^a-z]/g, "");
            w0 = this.ei.stripCondition(w0) || w0;
            let hints = "";
            let possibilities = getIns(w0).concat(getIns(w0 + "s"));
            if (possibilities.length > 0) {
                possibilities.forEach(i => {
                    let err = i.emit(ln);
                    hints += lf("   Maybe: {0} ({1} at '{2}')\n", i.toString(), err.error, err.errorAt);
                });
            }
            this.pushError(lf("assembly error"), hints);
        }
        buildLine(tx, lst) {
            let mkLine = (tx) => {
                let l = new Line(this, tx);
                l.scope = this.scope;
                l.lineNo = this.currLineNo;
                lst.push(l);
                return l;
            };
            let l = mkLine(tx);
            let words = tokenize(l.text) || [];
            l.words = words;
            let w0 = words[0] || "";
            if (w0.charAt(w0.length - 1) == ":") {
                let m = /^([\.\w]+):$/.exec(words[0]);
                if (m) {
                    l.type = "label";
                    l.text = m[1] + ":";
                    l.words = [m[1]];
                    if (words.length > 1) {
                        words.shift();
                        l = mkLine(tx.replace(/^[^:]*:/, ""));
                        l.words = words;
                        w0 = words[0] || "";
                    }
                    else {
                        return;
                    }
                }
            }
            let c0 = w0.charAt(0);
            if (c0 == "." || c0 == "@") {
                l.type = "directive";
                if (l.words[0] == "@scope")
                    this.handleDirective(l);
            }
            else {
                if (l.words.length == 0)
                    l.type = "empty";
                else
                    l.type = "instruction";
            }
        }
        prepLines(text) {
            this.currLineNo = 0;
            this.realCurrLineNo = 0;
            this.lines = [];
            text.split(/\r?\n/).forEach(tx => {
                if (this.errors.length > 10)
                    return;
                this.currLineNo++;
                this.realCurrLineNo++;
                this.buildLine(tx, this.lines);
            });
        }
        iterLines() {
            this.stack = 0;
            this.buf = [];
            this.scopeId = 0;
            this.lines.forEach(l => {
                if (this.errors.length > 10)
                    return;
                this.currLine = l;
                if (l.words.length == 0)
                    return;
                if (l.type == "label") {
                    let lblname = this.scopedName(l.words[0]);
                    this.prevLabel = lblname;
                    if (this.finalEmit) {
                        if (this.equs[lblname] != null)
                            this.directiveError(lf(".equ redefined as label"));
                        let curr = this.labels[lblname];
                        if (curr == null)
                            oops();
                        if (this.errors.length == 0 && curr != this.location()) {
                            oops(`invalid location: ${this.location()} != ${curr} at ${lblname}`);
                        }
                        assert(this.errors.length > 0 || curr == this.location());
                        if (this.reallyFinalEmit) {
                            this.stackAtLabel[lblname] = this.stack;
                        }
                    }
                    else {
                        if (this.labels.hasOwnProperty(lblname))
                            this.directiveError(lf("label redefinition"));
                        else if (this.inlineMode && /^_/.test(lblname))
                            this.directiveError(lf("labels starting with '_' are reserved for the compiler"));
                        else {
                            this.labels[lblname] = this.location();
                        }
                    }
                    l.location = this.location();
                }
                else if (l.type == "directive") {
                    this.handleDirective(l);
                }
                else if (l.type == "instruction") {
                    this.handleInstruction(l);
                }
                else if (l.type == "empty") ;
                else {
                    oops();
                }
            });
        }
        getSource(clean, numStmts = 1, flashSize = 0) {
            let lenPrev = 0;
            let size = (lbl) => {
                let curr = this.labels[lbl] || lenPrev;
                let sz = curr - lenPrev;
                lenPrev = curr;
                return sz;
            };
            let lenTotal = this.buf ? this.location() : 0;
            let lenCode = size("_code_end");
            let lenHelpers = size("_helpers_end");
            let lenVtables = size("_vtables_end");
            let lenLiterals = size("_literals_end");
            let lenAllCode = lenPrev;
            let totalSize = (lenTotal + this.baseOffset) & 0xffffff;
            if (flashSize && totalSize > flashSize)
                userError(lf("program too big by {0} bytes!", totalSize - flashSize));
            flashSize = flashSize || 128 * 1024;
            let totalInfo = lf("; total bytes: {0} ({1}% of {2}k flash with {3} free)", totalSize, (100 * totalSize / flashSize).toFixed(1), (flashSize / 1024).toFixed(1), flashSize - totalSize);
            let res = 
            // ARM-specific
            lf("; generated code sizes (bytes): {0} (incl. {1} user, {2} helpers, {3} vtables, {4} lits); src size {5}\n", lenAllCode, lenCode, lenHelpers, lenVtables, lenLiterals, lenTotal - lenAllCode) +
                lf("; assembly: {0} lines; density: {1} bytes/stmt; ({2} stmts)\n", this.lines.length, Math.round(100 * lenCode / numStmts) / 100, numStmts) +
                totalInfo + "\n" +
                this.stats + "\n\n";
            let skipOne = false;
            this.lines.forEach((ln, i) => {
                if (ln.words[0] == "_stored_program") {
                    res += "_stored_program: .string \"...\"\n";
                    skipOne = true;
                    return;
                }
                if (skipOne) {
                    skipOne = false;
                    return;
                }
                let text = ln.text;
                if (clean) {
                    if (ln.words[0] == "@stackempty" &&
                        this.lines[i - 1].text == ln.text)
                        return;
                    text = text.replace(/; WAS: .*/, "");
                    if (!text.trim())
                        return;
                }
                res += text + "\n";
            });
            return res;
        }
        peepHole() {
            // TODO add: str X; ldr X -> str X ?
            let mylines = this.lines.filter(l => l.type != "empty");
            for (let i = 0; i < mylines.length; ++i) {
                let ln = mylines[i];
                if (/^user/.test(ln.scope)) // skip opt for user-supplied assembly
                    continue;
                let lnNext = mylines[i + 1];
                if (!lnNext)
                    continue;
                let lnNext2 = mylines[i + 2];
                if (ln.type == "instruction") {
                    this.ei.peephole(ln, lnNext, lnNext2);
                }
            }
        }
        clearLabels() {
            this.labels = {};
            this.commPtr = 0;
        }
        peepPass(reallyFinal) {
            this.peepOps = 0;
            this.peepDel = 0;
            this.peepCounts = {};
            this.peepHole();
            this.throwOnError = true;
            this.finalEmit = false;
            this.clearLabels();
            this.iterLines();
            assert(!this.checkStack || this.stack == 0);
            this.finalEmit = true;
            this.reallyFinalEmit = reallyFinal || this.peepOps == 0;
            this.iterLines();
            this.stats += lf("; peep hole pass: {0} instructions removed and {1} updated\n", this.peepDel, this.peepOps - this.peepDel);
        }
        getLabels() {
            if (!this.userLabelsCache)
                this.userLabelsCache = mapMap(this.labels, (k, v) => v + this.baseOffset);
            return this.userLabelsCache;
        }
        emit(text) {
            assert(this.buf == null);
            this.prepLines(text);
            if (this.errors.length > 0)
                return;
            this.clearLabels();
            this.iterLines();
            if (this.checkStack && this.stack != 0)
                this.directiveError(lf("stack misaligned at the end of the file"));
            if (this.errors.length > 0)
                return;
            this.ei.expandLdlit(this);
            this.clearLabels();
            this.iterLines();
            this.finalEmit = true;
            this.reallyFinalEmit = this.disablePeepHole;
            this.iterLines();
            if (this.errors.length > 0)
                return;
            if (!this.disablePeepHole) {
                let maxPasses = 5;
                for (let i = 0; i < maxPasses; ++i) {
                    console.debug(`Peephole OPT, pass ${i}`);
                    this.peepPass(i == maxPasses);
                    if (this.peepOps == 0)
                        break;
                }
            }
        }
    }
    class VMFile extends File {
        constructor(ei) {
            super(ei);
        }
    }
    // an assembler provider must inherit from this
    // class and provide Encoders and Instructions
    class AbstractProcessor {
        constructor() {
            this.file = null;
            this.encoders = {};
            this.instructions = {};
        }
        toFnPtr(v, baseOff, lbl) {
            return v;
        }
        wordSize() {
            return -1;
        }
        computeStackOffset(kind, offset) {
            return offset;
        }
        is32bit(i) {
            return false;
        }
        emit32(v1, v2, actual) {
            return null;
        }
        postProcessRelAddress(f, v) {
            return v;
        }
        postProcessAbsAddress(f, v) {
            return v;
        }
        peephole(ln, lnNext, lnNext2) {
            return;
        }
        registerNo(actual, enc) {
            return null;
        }
        getAddressFromLabel(f, i, s, wordAligned = false) {
            return null;
        }
        isPop(opcode) {
            return false;
        }
        isPush(opcode) {
            return false;
        }
        isAddSP(opcode) {
            return false;
        }
        isSubSP(opcode) {
            return false;
        }
        testAssembler() {
            assert(false);
        }
        expandLdlit(f) {
        }
        addEnc(n, p, e) {
            let ee = {
                name: n,
                pretty: p,
                encode: e,
                isRegister: /^\$[sr]\d/i.test(n),
                isImmediate: /^\$i\d/i.test(n),
                isRegList: /^\$[sr]l\d/i.test(n),
                isLabel: /^\$l[a-z]/i.test(n),
            };
            this.encoders[n] = ee;
            return ee;
        }
        inrange(max, v, e) {
            if (Math.floor(v) != v)
                return null;
            if (v < 0)
                return null;
            if (v > max)
                return null;
            return e;
        }
        inminmax(min, max, v, e) {
            if (Math.floor(v) != v)
                return null;
            if (v < min)
                return null;
            if (v > max)
                return null;
            return e;
        }
        inseq(seq, v) {
            let ind = seq.indexOf(v);
            if (ind < 0)
                return null;
            return ind;
        }
        inrangeSigned(max, v, e) {
            if (Math.floor(v) != v)
                return null;
            if (v < -(max + 1))
                return null;
            if (v > max)
                return null;
            let mask = (max << 1) | 1;
            return e & mask;
        }
        addInst(name, code, mask, is32Bit) {
            let ins = new Instruction(this, name, code, mask, is32Bit);
            if (!this.instructions.hasOwnProperty(ins.name))
                this.instructions[ins.name] = [];
            this.instructions[ins.name].push(ins);
            return ins;
        }
        addInst32(name, code, mask) {
            // the high bit should be always set
            const high = 0x80000000;
            assert(!!(code & high));
            assert(!!(mask & high));
            // we clear it to avoid problems with numbers becoming negative
            code &= ~high;
            mask &= ~high;
            return this.addInst(name, code, mask, true);
        }
    }
    // utility functions
    function tokenize(line) {
        let words = [];
        let w = "";
        loop: for (let i = 0; i < line.length; ++i) {
            switch (line[i]) {
                case "[":
                case "]":
                case "!":
                case "{":
                case "}":
                case ",":
                    if (w) {
                        words.push(w);
                        w = "";
                    }
                    words.push(line[i]);
                    break;
                case " ":
                case "\t":
                case "\r":
                case "\n":
                    if (w) {
                        words.push(w);
                        w = "";
                    }
                    break;
                case "/":
                    if (line[i + 1] == "/")
                        break loop;
                    break;
                case ";":
                    // drop the trailing comment
                    break loop;
                default:
                    w += line[i];
                    break;
            }
        }
        if (w) {
            words.push(w);
            w = "";
        }
        if (!words[0])
            return null;
        return words;
    }
    function parseString(s) {
        s = s.replace(/\\\\/g, "\\B") // don't get confused by double backslash
            .replace(/\\(['\?])/g, (f, q) => q) // these are not valid in JSON yet valid in C
            .replace(/\\[z0]/g, "\u0000") // \0 is valid in C 
            .replace(/\\x([0-9a-f][0-9a-f])/gi, (f, h) => "\\u00" + h)
            .replace(/\\B/g, "\\\\"); // undo anti-confusion above
        try {
            return JSON.parse(s);
        }
        catch (e) {
            return null;
        }
    }
    function emitErr(msg, tok) {
        return {
            stack: null,
            opcode: null,
            error: msg,
            errorAt: tok
        };
    }
    function expectError(ei, asm) {
        let b = new File(ei);
        b.emit(asm);
        if (b.errors.length == 0) {
            oops("ASMTEST: expecting error for: " + asm);
        }
        // console.log(b.errors[0].message)
    }
    function tohex(n) {
        if (n < 0 || n > 0xffff)
            return ("0x" + n.toString(16)).toLowerCase();
        else
            return ("0x" + ("000" + n.toString(16)).slice(-4)).toLowerCase();
    }
    function expect(ei, disasm) {
        let exp = [];
        let asm = disasm.replace(/^([0-9a-fA-F]{4,8})\s/gm, (w, n) => {
            exp.push(parseInt(n.slice(0, 4), 16));
            if (n.length == 8)
                exp.push(parseInt(n.slice(4, 8), 16));
            return "";
        });
        let b = new File(ei);
        b.throwOnError = true;
        b.disablePeepHole = true;
        b.emit(asm);
        if (b.errors.length > 0) {
            console.debug(b.errors[0].message);
            oops("ASMTEST: not expecting errors");
        }
        if (b.buf.length != exp.length)
            oops("ASMTEST: wrong buf len");
        for (let i = 0; i < exp.length; ++i) {
            if (b.buf[i] != exp[i])
                oops("ASMTEST: wrong buf content at " + i + " , exp:" + tohex(exp[i]) + ", got: " + tohex(b.buf[i]));
        }
    }

    /*! *****************************************************************************
    Copyright (c) Microsoft Corporation.

    Permission to use, copy, modify, and/or distribute this software for any
    purpose with or without fee is hereby granted.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
    REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
    INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
    LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
    OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
    PERFORMANCE OF THIS SOFTWARE.
    ***************************************************************************** */

    function __await(v) {
        return this instanceof __await ? (this.v = v, this) : new __await(v);
    }

    function __asyncGenerator(thisArg, _arguments, generator) {
        if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
        var g = generator.apply(thisArg, _arguments || []), i, q = [];
        return i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i;
        function verb(n) { if (g[n]) i[n] = function (v) { return new Promise(function (a, b) { q.push([n, v, a, b]) > 1 || resume(n, v); }); }; }
        function resume(n, v) { try { step(g[n](v)); } catch (e) { settle(q[0][3], e); } }
        function step(r) { r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r); }
        function fulfill(value) { resume("next", value); }
        function reject(value) { resume("throw", value); }
        function settle(f, v) { if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]); }
    }

    const asmDeps = {
        'softmax': ['expf_asm']
    };
    const asmFns = {
        "expf_asm": `
// based on https://stackoverflow.com/questions/29381117
expf_asm:
	vldr.32	s15, .L10
	vcmpe.f32	s0, s15
	vmrs	APSR_nzcv, FPSCR
	bmi	.L5
	vldr.32	s15, .L10+4
	vcmpe.f32	s0, s15
	vmrs	APSR_nzcv, FPSCR
	bgt	.L9
	vldr.32	s15, .L10+8
	vldr.32	s9, .L10+12
	vldr.32	s6, .L10+16
	vldr.32	s7, .L10+20
	vldr.32	s10, .L10+24
	vldr.32	s8, .L10+28
	vldr.32	s11, .L10+32
	vldr.32	s12, .L10+36
	vldr.32	s13, .L10+40
	vmul.f32	s15, s0, s15
	vmov.f32	s14, #1.0
	vadd.f32	s15, s15, s9
	vsub.f32	s15, s15, s9
	vfma.f32	s0, s15, s6
	vcvt.s32.f32	s9, s15
	vfma.f32	s0, s15, s7
	vmov.f32	s15, s10
	vfma.f32	s15, s8, s0
	vmov	r3, s9	// int
	vfma.f32	s11, s15, s0
	vfma.f32	s12, s11, s0
	vfma.f32	s13, s12, s0
	vmov.f32	s15, s13
	vmov.f32	s13, s14
	vfma.f32	s13, s15, s0
	vfma.f32	s14, s13, s0
	vmov	r2, s14	// int
	add	r3, r2, r3, lsl #23
	vmov	s0, r3	// int
	bx	lr
.L9:
	vldr.32	s15, .L10+44
	vmov.f32	s14, #1.0
	vdiv.f32	s0, s14, s15
	bx	lr
.L5:
	vldr.32	s0, .L10+44
	bx	lr
.L11:
	.align	2
.L10:
	.word	3265921024
	.word	1118699520
	.word	1069066811
	.word	1262485504
	.word	3207688704
	.word	3049242254
	.word	1007234926
	.word	984915968
	.word	1026207149
	.word	1042983464
	.word	1056964603
	.word	0
`,
        "softmax": `
softmax:
	cmp	r1, #1
	push	{r3, r4, r5, lr}
	vldr.32	s5, [r0]
	bls	.L13
	adds	r3, r0, #4
	add	r2, r0, r1, lsl #2
.L16:
	vldmia.32	r3!, {s15}
	vcmp.f32	s15, s5
	vmrs	APSR_nzcv, FPSCR
	it	gt
	vmovgt.f32	s5, s15
	cmp	r2, r3
	bne	.L16
.L17:
	movs	r4, #0
	vmov	s4, r4
	mov	r5, r0
.L19:
	vldr.32	s0, [r5]
	vsub.f32	s0, s0, s5
	bl	expf_asm
	adds	r4, #1
	cmp	r1, r4
	vadd.f32	s4, s4, s0
	vstmia.32	r5!, {s0}
	bhi	.L19
	movs	r3, #0
.L20:
	vldr.32	s14, [r0]
	vdiv.f32	s15, s14, s4
	adds	r3, #1
	cmp	r1, r3
	vstmia.32	r0!, {s15}
	bhi	.L20
	pop	{r3, r4, r5, pc}
.L13:
	cmp	r1, #0
	bne	.L17
	pop	{r3, r4, r5, pc}
`
    };

    ///<reference path="pxtpackage.d.ts" />
    const unrollLimit = 10;
    var OpCode;
    (function (OpCode) {
        OpCode[OpCode["comment"] = 0] = "comment";
        OpCode[OpCode["label"] = 1] = "label";
        OpCode[OpCode["repeat"] = 2] = "repeat";
        OpCode[OpCode["loadWeightAddr"] = 3] = "loadWeightAddr";
        OpCode[OpCode["loadDataAddr"] = 4] = "loadDataAddr";
        OpCode[OpCode["addPtr"] = 5] = "addPtr";
        OpCode[OpCode["loadFConst"] = 6] = "loadFConst";
        OpCode[OpCode["load"] = 7] = "load";
        OpCode[OpCode["store"] = 8] = "store";
        OpCode[OpCode["vmul"] = 9] = "vmul";
        OpCode[OpCode["vmax"] = 10] = "vmax";
        OpCode[OpCode["vadd"] = 11] = "vadd";
        OpCode[OpCode["vcvt"] = 12] = "vcvt";
        OpCode[OpCode["relu"] = 13] = "relu";
        OpCode[OpCode["fcall"] = 14] = "fcall";
    })(OpCode || (OpCode = {}));
    var Reg;
    (function (Reg) {
        Reg[Reg["S0"] = 0] = "S0";
        Reg[Reg["S1"] = 1] = "S1";
        Reg[Reg["S15"] = 15] = "S15";
        Reg[Reg["S31"] = 32] = "S31";
        Reg[Reg["InputPtr"] = 200] = "InputPtr";
        Reg[Reg["OutputPtr"] = 201] = "OutputPtr";
        Reg[Reg["KernelPtr"] = 202] = "KernelPtr";
        Reg[Reg["DataDescPtr"] = 203] = "DataDescPtr";
        Reg[Reg["Index0"] = 300] = "Index0";
        Reg[Reg["Tmp0"] = 400] = "Tmp0";
        Reg[Reg["Zero"] = 500] = "Zero";
        Reg[Reg["One"] = 501] = "One";
    })(Reg || (Reg = {}));
    var F16Mode;
    (function (F16Mode) {
        F16Mode[F16Mode["Off"] = 0] = "Off";
        F16Mode[F16Mode["On"] = 1] = "On";
        F16Mode[F16Mode["Even"] = 2] = "Even";
        F16Mode[F16Mode["Odd"] = 3] = "Odd";
    })(F16Mode || (F16Mode = {}));
    function assert$1(cond, msg = "assertion failed") {
        if (!cond) {
            debugger;
            throw new Error("ir: " + msg);
        }
    }
    function addParamBytes(mi, bytes) {
        assert$1((mi.weightPtr & (bytes.length - 1)) == 0);
        if (!mi.weightBuffer)
            mi.weightBuffer = new Uint8Array(128);
        const dstlen = mi.weightPtr + bytes.length;
        if (dstlen + 3 > mi.weightBuffer.length) {
            const buf = new Uint8Array(dstlen * 2);
            buf.set(mi.weightBuffer);
            mi.weightBuffer = buf;
        }
        mi.weightBuffer.set(bytes, mi.weightPtr);
        mi.weightPtr = dstlen;
    }
    function addFloat32(mi, v) {
        assert$1(v != null && !isNaN(v));
        mi.weightAsm += `.float ${v}\n`;
        const u = float32ToUInt32(v);
        addParamBytes(mi, [
            (u >> 0) & 0xff,
            (u >> 8) & 0xff,
            (u >> 16) & 0xff,
            (u >> 24) & 0xff,
        ]);
    }
    function addFloat16(mi, v) {
        assert$1(v != null && !isNaN(v));
        mi.weightAsm += `.float16 ${v}\n`;
        const u = float16toUInt16(v);
        addParamBytes(mi, [
            (u >> 0) & 0xff,
            (u >> 8) & 0xff,
        ]);
    }
    function alignWeights(mi) {
        while (mi.weightPtr & 3)
            addParamBytes(mi, [0]);
        mi.weightAsm += ".balign 4\n";
    }
    function addWeight(mi, v) {
        if (mi.opts.float16weights)
            addFloat16(mi, v);
        else
            addFloat32(mi, v);
    }
    function addBias(mi, v) {
        addFloat32(mi, v);
    }
    function weightOffset(mi) {
        assert$1((mi.weightPtr & 3) == 0);
        return mi.weightPtr >> 2;
    }
    function stringifyComment(msg) {
        if (!msg)
            return "";
        return "// " + msg.replace(/\n/g, "\n// ");
    }
    function indent(s) {
        return "  " + s.replace(/\n$/, "").replace(/\n/g, "\n  ") + "\n";
    }
    function numCycles(ops) {
        let cycles = 0;
        let prevDst = null;
        const addConst = (k) => k < (1 << 12) ? 1 : 2;
        for (const op of ops) {
            switch (op.opcode) {
                case OpCode.comment:
                case OpCode.label:
                    break;
                case OpCode.repeat:
                    cycles += (numCycles(op.body) + 4 + (op.isDef ? 1 : 0)) * op.num + 1;
                    break;
                case OpCode.loadWeightAddr:
                    cycles += 2 + addConst(op.num * 4);
                    break;
                case OpCode.loadDataAddr:
                    cycles += addConst(op.num * 4 + 8);
                    break;
                case OpCode.addPtr:
                    if (op.src == null)
                        cycles += addConst(op.num * 4);
                    else {
                        if (op.num != 1) {
                            if (op.src > Reg.Zero) {
                                if (op.src == Reg.Zero + 1) ;
                                else if (op.src == Reg.Zero + 2) {
                                    cycles++;
                                }
                                else {
                                    cycles += 2;
                                }
                            }
                            else {
                                cycles++;
                            }
                        }
                        cycles += 2;
                    }
                    if (op.num == 1)
                        cycles += 1;
                    else
                        cycles += 3;
                    break;
                case OpCode.loadFConst:
                    if (op.num == 0)
                        cycles += 2;
                    else if (op.num == 1)
                        cycles += 1;
                    else
                        cycles += 4; // ??
                    break;
                case OpCode.load:
                    cycles += 1 + op.num;
                    break;
                case OpCode.store:
                    cycles += 1 + op.num;
                    break;
                case OpCode.relu:
                    cycles += 6;
                    break;
                case OpCode.vmax:
                    cycles += 4;
                    if (op.src != op.dst)
                        cycles++;
                    break;
                case OpCode.vmul:
                case OpCode.vadd:
                    if (op.src === prevDst || op.srcAlt === prevDst)
                        cycles += 2;
                    else
                        cycles += 1;
                    prevDst = op.dst;
                    break;
                case OpCode.vcvt:
                    cycles += 1;
                    break;
                case OpCode.fcall:
                    if (op.fname == "softmax")
                        cycles += 200 + op.num * 150; // estimate
                    else
                        cycles += 500 + op.num * 500; // estimate
                    break;
                default:
                    throw new Error("bad op " + op.opcode);
            }
        }
        return cycles;
    }
    function toThumb(modelInfo, ops) {
        var _a;
        const weightAddrDO = 0;
        const zeroDO = 4;
        const descWords = 2;
        const usedFns = {};
        const hasTest = !!modelInfo.opts.testInput && !!modelInfo.opts.includeTest;
        let ind = "";
        const byteOffset = (n) => 4 * (n + descWords);
        const header = [
            "0x30470f62  // magic",
            "0x46344c4d  // more magic; ML4F",
            `_start_model-_header // header size`,
            `_end-_header // total size of compiled object`,
            `_weights-_header // offset of weights`,
            hasTest ? `_testInput-_header` : `0 // no tests`,
            hasTest ? `_testOutput-_header` : `0 // no tests`,
            `${byteOffset(modelInfo.arenaSize)} // arena size`,
            `${byteOffset(0)}  // offset of input data`,
            `1 // input type - float32`,
            `${byteOffset(modelInfo.outputOffset)}  // offset of output data`,
            `1 // output type - float32`,
        ];
        for (let i = 0; i < 4; ++i)
            header.push(`0 // padding`);
        addShape(modelInfo.inputShape, "input");
        addShape(modelInfo.outputShape, "output");
        let initCmt = "";
        while (((_a = ops[0]) === null || _a === void 0 ? void 0 : _a.opcode) == OpCode.comment) {
            const op = ops.shift();
            initCmt += stringifyComment(op.fname) + "\n";
        }
        let regAlloc = {};
        let resText = `${stringifyComment(modelInfo.stats)}
    .cpu cortex-m4
    .text
    .arch armv7e-m
    .syntax unified
    .thumb
    .thumb_func
    .fpu fpv4-sp-d16
// ABI: r0 -> points to magic, r1 -> points to RAM arena
_header:
`;
        for (const h of header)
            write(`.word ${h}`);
        let lblid = 0;
        // TODO use high registers for i/o/k ? these are used with 32 bit instructions anyways
        regAlloc[Reg.InputPtr] = 1;
        regAlloc[Reg.OutputPtr] = 2;
        regAlloc[Reg.KernelPtr] = 3;
        regAlloc[Reg.DataDescPtr] = 7;
        write(`_start_model:`);
        write(`push {r4,r5,r6,r7,r8,r9,r10,r11,r12,lr}`);
        write(`mov ${reg(Reg.DataDescPtr)}, r1`);
        write(`ldr r1, [r0, #4*4] // weight offset`);
        write(`adds r1, r0 // weight addr`);
        write(`str r1, [${reg(Reg.DataDescPtr)}, #${weightAddrDO}]`);
        write(`movs r1, #0`);
        write(`str r1, [${reg(Reg.DataDescPtr)}, #${zeroDO}]`);
        compiles(ops);
        write(`pop {r4,r5,r6,r7,r8,r9,r10,r11,r12,pc}`);
        for (const k of Object.keys(usedFns)) {
            for (const d of asmDeps[k] || [])
                usedFns[d] = true;
        }
        for (const k of Object.keys(usedFns)) {
            write(asmFns[k]);
        }
        write(".balign 4");
        //const u32 = new Uint32Array(modelInfo.weightBuffer.buffer)
        write(`_weights:\n${modelInfo.weightAsm}`);
        if (hasTest) {
            writeArray("_testInput", modelInfo.opts.testInput);
            writeArray("_testOutput", modelInfo.opts.testOutput);
        }
        write("_end:");
        return resText;
        function writeArray(lbl, vals) {
            write(`${lbl}:`);
            for (const w of vals)
                write(`.float ${w}`);
        }
        function addShape(shape, lbl) {
            for (const shp of shape)
                if (shp != null)
                    header.push(`${shp} // ${lbl} shape`);
            header.push(`0 // end of ${lbl} shape`);
        }
        function alloc(r, f) {
            assert$1(!regAlloc[r]);
            const copy = {};
            const used = {};
            for (const k of Object.keys(regAlloc)) {
                copy[k] = regAlloc[k];
                used[copy[k]] = true;
            }
            let all = -1;
            for (let i = 4; i <= 12; ++i) {
                if (!used[i]) {
                    all = i;
                    break;
                }
            }
            if (all < 0)
                oops("can't alloc " + r);
            regAlloc[r] = all;
            if (f) {
                const pind = ind;
                try {
                    ind += "    ";
                    f();
                }
                finally {
                    ind = pind;
                    regAlloc = copy;
                }
            }
        }
        function write(asm) {
            if (isFake(asm))
                oops("wrong reg: " + asm);
            resText += ind + asm + "\n";
        }
        function oops(msg) {
            debugger;
            throw new Error("internal thumb error: " + msg);
        }
        function reg(r) {
            if (r == null)
                return "<fake>";
            if (r <= Reg.S31)
                return "s" + (r - Reg.S0);
            if (r >= Reg.Zero)
                return "#" + (r - Reg.Zero);
            const id = regAlloc[r];
            if (id == undefined)
                return "<fake:" + regName(r) + ">";
            return "r" + id;
        }
        function isFake(r) {
            return r.indexOf("<fake") >= 0;
        }
        function isLowReg(reg) {
            return /^r[0-7]$/.test(reg);
        }
        function loadConst(dst, num) {
            // TODO?
            if (num <= 0xff && isLowReg(dst))
                write(`movs ${dst}, #${num}`);
            else
                write(`movw ${dst}, #${num}`);
        }
        function addConst(dst, src, num) {
            if (Math.abs(num) < (1 << 12)) {
                if (num < 0)
                    write(`subw ${dst}, ${src}, #${-num}`);
                else
                    write(`addw ${dst}, ${src}, #${num}`);
            }
            else {
                assert$1(src != dst);
                loadConst(dst, num);
                write(`adds ${dst}, ${src}, ${dst}`);
            }
        }
        function compiles(ops) {
            for (const op of ops)
                compile(op);
        }
        function range$1(op) {
            return "{" + range(op.num).map(k => reg(op.dst + k)).join(",") + "}";
        }
        function compile(op) {
            let dst = reg(op.dst);
            const src = reg(op.src);
            const srcAlt = reg(op.srcAlt);
            const incr = op.increment ? "!" : "";
            switch (op.opcode) {
                case OpCode.label:
                    write(`${op.fname}:`);
                    break;
                case OpCode.comment:
                    write(stringifyComment(op.fname));
                    break;
                case OpCode.repeat:
                    assert$1(op.num >= 1);
                    alloc(op.dst, () => {
                        dst = reg(op.dst);
                        const lbl = `.l.${lblid++}`;
                        loadConst(dst, op.isDef ? 0 : op.num);
                        write(`${lbl}:  // rep ${op.num}`);
                        compiles(op.body);
                        if (op.isDef) {
                            write(`adds ${dst}, #1`);
                            write(`cmp ${dst}, #${op.num}`);
                            write(`blt ${lbl}`);
                        }
                        else {
                            if (isLowReg(dst))
                                write(`subs ${dst}, #1`);
                            else
                                write(`subs ${dst}, ${dst}, #1`);
                            write(`bne ${lbl}`);
                        }
                    });
                    break;
                case OpCode.loadWeightAddr:
                    write(`ldr r0, [${reg(Reg.DataDescPtr)}, #${weightAddrDO}]`);
                    addConst(dst, "r0", op.num * 4);
                    break;
                case OpCode.loadDataAddr:
                    addConst(dst, reg(Reg.DataDescPtr), byteOffset(op.num));
                    break;
                case OpCode.addPtr:
                    if (isFake(dst) && op.isDef) {
                        alloc(op.dst);
                        dst = reg(op.dst);
                    }
                    if (op.src == null) {
                        addConst(dst, srcAlt, op.num * 4);
                    }
                    else {
                        if (op.num != 1) {
                            loadConst("r0", op.num * 4);
                            if (src[0] == '#') {
                                const n = +src.slice(1);
                                if (n == 0)
                                    loadConst("r0", 0);
                                else if (n == 1) ;
                                else if (n == 2) {
                                    write(`adds r0,r0`);
                                }
                                else {
                                    assert$1(dst != srcAlt);
                                    loadConst(dst, n);
                                    write(`muls r0, ${dst}`);
                                }
                            }
                            else {
                                write(`muls r0, ${src}`);
                            }
                        }
                        else {
                            if (src[0] == '#') {
                                const n = +src.slice(1);
                                loadConst("r0", n << 2);
                            }
                            else {
                                write(`lsls r0, ${src}, #2`);
                            }
                        }
                        write(`adds ${dst}, ${srcAlt}, r0`);
                    }
                    break;
                case OpCode.loadFConst:
                    if (op.num == 0.0)
                        write(`vldr ${dst}, [${reg(Reg.DataDescPtr)}, #${zeroDO}]`);
                    else
                        write(`vmov ${dst}, #${op.num}e+0`);
                    break;
                case OpCode.load:
                    assert$1(op.f16Mode != F16Mode.On);
                    write(`vldm ${src}${incr}, ${range$1(op)}`);
                    break;
                case OpCode.store:
                    write(`vstm ${src}${incr}, ${range$1(op)}`);
                    break;
                case OpCode.relu:
                    write(`ldr r0, [${dst}, #0]`);
                    // negative check on FP and int is the same
                    write(`cmp r0, #0`);
                    write(`it lt`);
                    // int 0 is same as 0.0f
                    // this could be movslt but GAS always assembles this as movw, so for bit-exactness we stick to movw
                    write(`movwlt r0, #0`);
                    write(`stm ${dst}!, {r0}`);
                    break;
                case OpCode.vmul:
                    write(`vmul.f32 ${dst}, ${src}, ${srcAlt}`);
                    break;
                case OpCode.vadd:
                    write(`vadd.f32 ${dst}, ${src}, ${srcAlt}`);
                    break;
                case OpCode.vcvt:
                    write(`${op.fname} ${dst}, ${src}`);
                    break;
                case OpCode.vmax:
                    assert$1(dst != srcAlt);
                    if (src != dst)
                        write(`vmov ${dst}, ${src}`);
                    write(`vcmp.f32 ${dst}, ${srcAlt}`);
                    write(`vmrs APSR_nzcv, FPSCR`);
                    write(`it mi`);
                    write(`vmovmi.f32 ${dst}, ${srcAlt}`);
                    break;
                case OpCode.fcall:
                    write(`mov r0, ${dst}`);
                    loadConst("r1", op.num);
                    write(`bl ${op.fname}`);
                    usedFns[op.fname] = true;
                    break;
                default:
                    oops("bad op " + op.opcode);
            }
        }
    }
    function toJS(modelInfo, op) {
        let r = "";
        if (op.opcode == OpCode.repeat) {
            const dst = regName(op.dst);
            r = `for (let ${dst} = 0; ${dst} < ${op.num}; ${dst}++) {\n${indent(toJSs(modelInfo, op.body))}}\n`;
        }
        else {
            r = stringify1(op);
        }
        if (r.indexOf("???") >= 0)
            oops("invalid register in: " + r);
        return r;
    }
    function stringify(op) {
        return op.map(stringify1).join("");
    }
    function stringify1(op) {
        const dst = op.dst == null ? null : regName(op.dst);
        const src = op.src == null ? null : regName(op.src);
        const srcAlt = op.srcAlt == null ? null : regName(op.srcAlt);
        switch (op.opcode) {
            case OpCode.label:
                return stringifyComment("label: " + op.fname) + "\n";
            case OpCode.comment:
                if (isBreak(op))
                    return "debugger\n";
                return stringifyComment(op.fname) + "\n";
            case OpCode.repeat:
                return `for (let ${dst} = 0; ${dst} < ${op.num}; ${dst}++) {\n${indent(stringify(op.body))}}\n`;
            case OpCode.loadWeightAddr:
                return `${dst} = weightOff + ${op.num}\n`;
            case OpCode.loadDataAddr:
                return `${dst} = dataOff + ${op.num}\n`;
            case OpCode.addPtr:
                if (op.src == null)
                    return `${dst} = ${srcAlt} + ${op.num}\n`;
                return `${dst} = ${srcAlt} + ${src}${op.num == 1 ? "" : " * " + op.num}\n`;
            case OpCode.loadFConst:
                return `${dst} = ${op.num}\n`;
            case OpCode.load: {
                let r = "";
                let dp = op.dst + 0;
                if (op.increment) {
                    for (let i = 0; i < op.num; ++i)
                        r += `${regName(dp++)} = ${op.fname || "mem"}[${src}++]\n`;
                }
                else {
                    for (let i = 0; i < op.num; ++i)
                        r += `${regName(dp++)} = mem[${src} + ${i}]\n`;
                }
                return r;
            }
            case OpCode.store: {
                let r = "";
                let dp = op.dst + 0;
                if (op.increment) {
                    for (let i = 0; i < op.num; ++i)
                        r += `mem[${src}++] = ${regName(dp++)}\n`;
                }
                else {
                    for (let i = 0; i < op.num; ++i)
                        r += `mem[${src} + ${i}] = ${regName(dp++)}\n`;
                }
                return r;
            }
            case OpCode.relu:
                return `if (mem[${dst}] < 0) mem[${dst}] = 0; ${dst}++\n`;
            case OpCode.vmul:
                return `${dst} = f32(${src} * ${srcAlt})\n`;
            case OpCode.vadd:
                return `${dst} = f32(${src} + ${srcAlt})\n`;
            case OpCode.vmax:
                return `${dst} = Math.max(${src}, ${srcAlt})\n`;
            case OpCode.fcall:
                return `${op.fname}(${dst}, ${op.num})\n`;
            case OpCode.vcvt:
                return `${dst} = rt.${op.fname.replace(/\./g, "_")}(${src})\n`;
            default:
                throw new Error("bad op " + op.opcode);
        }
    }
    function regName(r) {
        if (r <= Reg.S31)
            return "s" + (r - Reg.S0);
        if (r >= Reg.Zero)
            return "" + (r - Reg.Zero);
        if (r >= Reg.Tmp0)
            return "tmp" + (r - Reg.Tmp0);
        if (r >= Reg.Index0)
            return "idx" + (r - Reg.Index0);
        switch (r) {
            case Reg.InputPtr:
                return "input";
            case Reg.KernelPtr:
                return "kernel";
            case Reg.OutputPtr:
                return "output";
            default:
                return "???" + r;
        }
    }
    function toJSs(modelInfo, op) {
        return op.map(o => toJS(modelInfo, o)).join("");
    }
    let repIdx = 0;
    function repeatIdx(n, f) {
        const idx = Reg.Index0 + repIdx++;
        return {
            opcode: OpCode.repeat,
            dst: idx,
            num: n,
            body: f(idx),
            isDef: true
        };
    }
    function repeat(n, f) {
        const r = repeatIdx(n, f);
        r.isDef = false;
        return r;
    }
    function comment(str) {
        return {
            opcode: OpCode.comment,
            fname: str
        };
    }
    function label(name) {
        return {
            opcode: OpCode.label,
            fname: name
        };
    }
    function loadWeightAddr(dst, idx) {
        assert$1(idx >= 0);
        return {
            opcode: OpCode.loadWeightAddr,
            dst,
            num: idx
        };
    }
    function relaxWeights() {
        const r = addPtr(Reg.KernelPtr, null, 0);
        r.fname = "relax";
        return r;
    }
    function loadDataAddr(dst, idx) {
        assert$1(idx >= 0);
        return {
            opcode: OpCode.loadDataAddr,
            dst,
            num: idx
        };
    }
    function addPtr(dst, idx, mult = 1, base) {
        if (!base)
            base = dst;
        return {
            opcode: OpCode.addPtr,
            dst,
            src: idx,
            srcAlt: base,
            num: mult
        };
    }
    function load0(dst) {
        return {
            opcode: OpCode.loadFConst,
            dst,
            num: 0.0
        };
    }
    function load(dst, num, src, adv) {
        return {
            opcode: OpCode.load,
            dst,
            src,
            num: num,
            increment: adv
        };
    }
    function load16(dst, num, src) {
        return {
            opcode: OpCode.load,
            dst,
            src,
            num: num,
            increment: true,
            f16Mode: F16Mode.On,
        };
    }
    function loadWeight(mi, dst, num) {
        const src = Reg.KernelPtr;
        if (mi.opts.float16weights)
            return load16(dst, num, src);
        else
            return load(dst, num, src, true);
    }
    function store(dst, src, num, adv) {
        return {
            opcode: OpCode.store,
            src: dst,
            dst: src,
            num: num,
            increment: adv
        };
    }
    function relu(dst) {
        return {
            opcode: OpCode.relu,
            dst,
            increment: true
        };
    }
    function vmul(dst, a, b) {
        return {
            opcode: OpCode.vmul,
            dst,
            src: a,
            srcAlt: b,
        };
    }
    function vmax(dst, a, b) {
        if (b == dst)
            [a, b] = [b, a];
        return {
            opcode: OpCode.vmax,
            dst,
            src: a,
            srcAlt: b,
        };
    }
    function vadd(dst, a, b) {
        return {
            opcode: OpCode.vadd,
            dst,
            src: a,
            srcAlt: b,
        };
    }
    function vcvt(fname, dst, src) {
        return {
            opcode: OpCode.vcvt,
            dst,
            src,
            fname
        };
    }
    function fcall(name, dst, len) {
        return {
            opcode: OpCode.fcall,
            fname: name,
            dst,
            num: len,
        };
    }
    function flatten(...args) {
        const res = [];
        const add = (a) => {
            if (a)
                res.push(a);
        };
        for (const a of args) {
            if (Array.isArray(a)) {
                for (const b of a) {
                    if (Array.isArray(b)) {
                        for (const c of b)
                            add(c);
                    }
                    else {
                        add(b);
                    }
                }
            }
            else {
                add(a);
            }
        }
        return res;
    }
    function isRelax(op) {
        return (op.opcode == OpCode.addPtr && op.fname == "relax");
    }
    function isBreak(op) {
        return (op.opcode == OpCode.comment && op.fname == "BREAK");
    }
    function isOddF16(ops) {
        let cnt = 0;
        for (const op of ops) {
            if (op.opcode == OpCode.load && op.f16Mode)
                cnt += op.num;
            if (isRelax(op))
                cnt = (cnt + 1) & ~1;
        }
        return !!(cnt & 1);
    }
    function fixupAndMarkF16(ops) {
        function loop(ops, odd = false) {
            let cnt = odd ? 1 : 0;
            const isOdd = () => !!(cnt & 1);
            const res = [];
            for (let op of ops) {
                op = cloneOp(op);
                if (op.opcode == OpCode.repeat) {
                    if (op.num == 0)
                        continue;
                    const odd0 = isOdd();
                    const body0 = op.body;
                    const r = loop(body0, odd0);
                    op.body = r.ops;
                    if (r.odd != odd0) {
                        if (op.isDef) {
                            console.log(stringify([op]));
                            assert$1(false);
                        }
                        if (op.num == 1) {
                            pushRange(res, r.ops);
                            cnt++; // swap oddity
                        }
                        else {
                            const leftover = op.num & 1;
                            op.num >>= 1;
                            const r1 = loop(body0, r.odd);
                            assert$1(r1.odd == odd0);
                            op.body = r.ops.concat(r1.ops);
                            res.push(op);
                            if (leftover) {
                                const r2 = loop(body0, odd0);
                                pushRange(res, r2.ops);
                                cnt++;
                            }
                        }
                    }
                    else {
                        res.push(op);
                    }
                    continue;
                }
                res.push(op);
                if (op.opcode == OpCode.load && op.f16Mode) {
                    assert$1(op.f16Mode == F16Mode.On);
                    op.f16Mode = isOdd() ? F16Mode.Odd : F16Mode.Even;
                    cnt += op.num;
                }
                if (isRelax(op))
                    cnt = (cnt + 1) & ~1;
            }
            return { ops: res, odd: !!(cnt & 1) };
        }
        function expand(ops) {
            const res = [];
            for (let op of ops) {
                if (op.opcode == OpCode.repeat) {
                    assert$1(!isOddF16(op.body));
                    op.body = expand(op.body);
                    res.push(op);
                }
                else if (op.opcode == OpCode.load && op.f16Mode) {
                    let numLoad = 0;
                    let isBottom = false;
                    if (op.f16Mode == F16Mode.Odd) {
                        numLoad = (op.num >> 1) + 1;
                        res.push(addPtr(op.src, Reg.One, -1));
                        if (!(op.num & 1))
                            isBottom = true;
                    }
                    else if (op.f16Mode == F16Mode.Even) {
                        numLoad = (op.num + 1) >> 1;
                        if (op.num & 1)
                            isBottom = true;
                    }
                    else {
                        assert$1(false);
                    }
                    const ld = load(op.dst, numLoad, op.src, true);
                    ld.fname = "memU32";
                    res.push(ld);
                    let srcreg = op.dst + numLoad - 1;
                    for (let i = op.num - 1; i >= 0; --i) {
                        res.push(vcvt(isBottom ? "vcvtb.f32.f16" : "vcvtt.f32.f16", op.dst + i, srcreg));
                        if (isBottom)
                            srcreg--;
                        isBottom = !isBottom;
                    }
                }
                else {
                    res.push(op);
                }
            }
            return res;
        }
        return expand(loop(ops).ops);
    }
    function cloneOp(op) {
        return {
            opcode: op.opcode,
            dst: op.dst,
            src: op.src,
            srcAlt: op.srcAlt,
            isDef: op.isDef,
            f16Mode: op.f16Mode,
            increment: op.increment,
            num: op.num,
            body: op.body,
            fname: op.fname
        };
    }
    function optimize(ops, replMap = {}) {
        const repl = (r) => {
            if (!r)
                return r;
            if (replMap[r] != undefined)
                return replMap[r];
            return r;
        };
        const res = [];
        for (let op of ops) {
            op = cloneOp(op);
            op.dst = repl(op.dst);
            op.src = repl(op.src);
            op.srcAlt = repl(op.srcAlt);
            switch (op.opcode) {
                case OpCode.repeat:
                    if (op.num == 0) ;
                    else if (op.num == 1) {
                        replMap[op.dst] = Reg.Zero;
                        pushRange(res, optimize(op.body, replMap));
                    }
                    else {
                        op.body = optimize(op.body, replMap);
                        const stripLoop = op.num * op.body.length < unrollLimit * 2;
                        const canUnroll = !op.isDef && 2 * op.body.length < unrollLimit;
                        if (stripLoop) {
                            for (let i = 0; i < op.num; ++i) {
                                replMap[op.dst] = Reg.Zero + i;
                                // need to run optimize() again due to new replacement
                                pushRange(res, optimize(op.body, replMap));
                            }
                        }
                        else if (canUnroll) {
                            const unrollCnt = (unrollLimit / op.body.length) | 0;
                            const tmp = op.body.slice();
                            for (let i = 1; i < unrollCnt; ++i)
                                pushRange(op.body, tmp);
                            const newnum = (op.num / unrollCnt) | 0;
                            res.push(op);
                            const left = op.num - newnum * unrollCnt;
                            op.num = newnum;
                            for (let i = 0; i < left; ++i)
                                pushRange(res, tmp);
                        }
                        else {
                            res.push(op);
                        }
                    }
                    break;
                case OpCode.addPtr:
                    if (op.dst == op.srcAlt && (op.num == 0 || op.src == Reg.Zero)) ;
                    else
                        res.push(op);
                    break;
                default:
                    res.push(op);
            }
        }
        return res;
    }
    function reset() {
        repIdx = 0;
    }

    ///<reference path="pxtpackage.d.ts" />
    let inited$1 = false;
    const compilers = {
        Conv2D: { compile: compileConv, computePaddedInputShape: paddingConv },
        Conv1D: { compile: compileConv, computePaddedInputShape: paddingConv },
        MaxPooling1D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool },
        MaxPooling2D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool },
        Dense: { compile: compileDense },
        Dropout: {},
        Flatten: {},
        InputLayer: {},
        Reshape: {},
    };
    const numFPRegs = 32;
    const numTmpRegs = 8;
    function unsupported(msg) {
        debugger;
        throw new Error("Unsupported operator or config: " + msg);
    }
    function assert$2(cond, msg = "assertion failed") {
        if (!cond)
            unsupported(msg);
    }
    function getLayerInfo(l) {
        const ll = l;
        let r = ll.__ml4f_info;
        if (!r) {
            r = {
                layer: l
            };
            ll.__ml4f_info = r;
        }
        return r;
    }
    function validateConfig(info) {
        const config = info.layer.getConfig();
        if (info.model.opts.verbose)
            console.log(info.inputShape, info.outputShape, config);
        const is2D = info.inputShape.length == 4;
        if (is2D) {
            if (info.inputShape.length != 4 && info.inputShape.length != 3)
                unsupported("inputShape: " + info.inputShape.length);
            if (config.dataFormat != "channelsLast")
                unsupported("dataFormat: " + config.dataFormat);
        }
        else {
            if (info.inputShape.length != 3)
                unsupported("inputShape: " + info.inputShape.length);
        }
        if (config.dtype && config.dtype != "float32")
            unsupported("dtype: " + config.dtype);
    }
    function addActivation(res, info) {
        const config = info.layer.getConfig();
        const numoutp = shapeElts(info.outputShape);
        if (config.activation == "linear")
            return; // linear is identity
        res.push(loadDataAddr(Reg.OutputPtr, info.outputOff));
        if (config.activation == "relu")
            res.push(repeat(numoutp, () => [relu(Reg.OutputPtr)]));
        else if (config.activation == "softmax")
            res.push(fcall("softmax", Reg.OutputPtr, numoutp));
        else
            unsupported("activation: " + config.activation);
    }
    function paddingConv(info) {
        const config = info.layer.getConfig();
        const res = info.inputShape.slice();
        for (let i = 1; i <= config.kernelSize.length; ++i) {
            const str = config.strides[i - 1];
            const tmp = info.outputShape[i] * str + config.kernelSize[i - 1] - str;
            assert$2(tmp >= res[i]);
            res[i] = tmp;
        }
        return res;
    }
    function paddingPool(info) {
        const config = info.layer.getConfig();
        const res = info.inputShape.slice();
        for (let i = 1; i <= config.poolSize.length; ++i) {
            // TODO this may be wrong if config.poolSize != config.strides
            const tmp = info.outputShape[i] * config.strides[i - 1];
            if (tmp > res[i])
                res[i] = tmp;
        }
        return res;
    }
    function compileConv(info) {
        const config = info.layer.getConfig();
        const memRegs = numFPRegs >> 1;
        const flashRegs = numFPRegs >> 1;
        validateConfig(info);
        const is2D = config.kernelSize.length == 2;
        const weights0 = info.layer.weights[0].read().arraySync();
        const weights = (is2D ? weights0 : [weights0]);
        const fix1D = (a) => {
            a = a.slice();
            if (!is2D)
                a.unshift(1);
            return a;
        };
        const [kh, kw] = fix1D(config.kernelSize);
        const [strh, strw] = fix1D(config.strides);
        const [inph, inpw, inpch] = fix1D(info.inputShape.slice(1));
        const [outh, outw, outch] = fix1D(info.outputShape.slice(1));
        // padding not implemented yet
        assert$2(kh <= inph, "KH2");
        assert$2(kw <= inpw, "KW2");
        assert$2(weights.length == kh, "KH");
        assert$2(weights[0].length == kw, "KW");
        assert$2(weights[0][0].length == inpch, "CH");
        assert$2(weights[0][0][0].length == config.filters, "F");
        assert$2(outch == config.filters, "FF");
        const mi = info.model;
        const weightsIdx = weightOffset(mi);
        const bias = config.useBias ? info.layer.weights[1].read().arraySync() : null;
        for (let f = 0; f < config.filters; f++) {
            if (bias)
                addBias(mi, bias[f]);
            for (let y = 0; y < kh; y++) {
                for (let x = 0; x < kw; x++)
                    for (let c = 0; c < inpch; ++c)
                        addWeight(mi, weights[y][x][c][f]);
                alignWeights(mi);
            }
        }
        const res = [
            loadWeightAddr(Reg.KernelPtr, weightsIdx),
            repeatIdx(config.filters, filt => {
                const res = [];
                const setOutput = (res) => {
                    res.push(loadDataAddr(Reg.OutputPtr, info.outputOff));
                    res.push(addPtr(Reg.OutputPtr, filt));
                };
                // set bias
                setOutput(res);
                if (config.useBias)
                    res.push(load(Reg.S0, 1, Reg.KernelPtr, true));
                else
                    res.push(load0(Reg.S0));
                res.push(repeat(outw * outh, () => [
                    store(Reg.OutputPtr, Reg.S0, 1, false),
                    addPtr(Reg.OutputPtr, null, config.filters)
                ]));
                res.push(repeatIdx(kh, kline => {
                    const res = [];
                    const kernSz = kw * inpch;
                    let chunk = 0;
                    for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
                        chunk = kernSz - kernOff;
                        if (chunk > flashRegs)
                            chunk = flashRegs;
                        res.push(loadWeight(mi, memRegs, chunk));
                        res.push(loadDataAddr(Reg.InputPtr, info.inputOff + kernOff));
                        res.push(addPtr(Reg.InputPtr, kline, inpw * inpch));
                        setOutput(res);
                        const wSkip = strw * inpch;
                        const hSkip = strh * inpw * inpch;
                        res.push(repeat(outh, () => [repeat(outw, () => flatten(load(Reg.S0, chunk, Reg.InputPtr, true), addPtr(Reg.InputPtr, null, wSkip - chunk), range(chunk + 1).map(i => [
                                i < chunk ? vmul(i, i, i + memRegs) : null,
                                i >= 2 ? vadd(Reg.S0, Reg.S0, i - 1) : null
                            ]), load(Reg.S1, 1, Reg.OutputPtr, false), vadd(Reg.S0, Reg.S0, Reg.S1), store(Reg.OutputPtr, Reg.S0, 1, false), addPtr(Reg.OutputPtr, null, config.filters))),
                            addPtr(Reg.InputPtr, null, hSkip - outw * wSkip)]));
                    }
                    res.push(relaxWeights());
                    return res;
                }));
                res.push(relaxWeights());
                return res;
            })
        ];
        addActivation(res, info);
        return res;
    }
    function compileMaxPooling(info) {
        const config = info.layer.getConfig();
        const is2D = config.poolSize.length == 2;
        validateConfig(info);
        const fix1D = (a) => {
            a = a.slice();
            if (!is2D)
                a.unshift(1);
            return a;
        };
        const [kh, kw] = fix1D(config.poolSize);
        const [strh, strw] = fix1D(config.strides);
        const [inph, inpw, numch] = fix1D(info.inputShape.slice(1));
        const [outh, outw, outch] = fix1D(info.outputShape.slice(1));
        // padding not implemented yet
        assert$2(kh <= inph, "KH2");
        assert$2(kw <= inpw, "KW2");
        assert$2(numch == outch, "CH");
        if (kh - 1 > numTmpRegs)
            unsupported(`too high MaxPool2D area`);
        const lineW = inpw * numch;
        return [
            repeatIdx(numch, filt => {
                const res = [
                    loadDataAddr(Reg.OutputPtr, info.outputOff),
                    addPtr(Reg.OutputPtr, filt),
                    loadDataAddr(Reg.InputPtr, info.inputOff),
                    addPtr(Reg.InputPtr, filt),
                ];
                const ptrRegs = range(kh - 1).map(i => Reg.Tmp0 + i);
                ptrRegs.unshift(Reg.InputPtr);
                for (let i = 1; i < kh; ++i) {
                    const op = addPtr(ptrRegs[i], null, lineW * i, Reg.InputPtr);
                    op.isDef = true;
                    res.push(op);
                }
                res.push(repeat(outh, () => flatten(repeat(outw, () => {
                    const res = [];
                    for (let i = 0; i < kh; ++i) {
                        for (let j = 0; j < kw; ++j) {
                            const reg = i == 0 && j == 0 ? Reg.S0 : Reg.S1;
                            res.push(load(reg, 1, ptrRegs[i], true), addPtr(ptrRegs[i], null, numch - 1));
                            if (reg != Reg.S0)
                                res.push(vmax(Reg.S0, Reg.S0, reg));
                        }
                        res.push(addPtr(ptrRegs[i], null, (strw - kw) * numch));
                    }
                    res.push(store(Reg.OutputPtr, Reg.S0, 1, true), addPtr(Reg.OutputPtr, null, numch - 1));
                    return res;
                }), ptrRegs.map(r => addPtr(r, null, strh * lineW - outw * strw * numch)))));
                return res;
            })
        ];
    }
    function compileDense(info) {
        const config = info.layer.getConfig();
        const maxChunk = (numFPRegs >> 1) - 2;
        const memReg0 = Reg.S1;
        const flashReg0 = memReg0 + maxChunk;
        //if (info.model.opts.verbose)
        //    console.log(info.inputShape, info.outputShape, config)
        if (info.inputShape.length != 2)
            unsupported("inputShape: " + info.inputShape.length);
        if (config.dtype && config.dtype != "float32")
            unsupported("dtype: " + config.dtype);
        const weights = info.layer.weights[0].read().arraySync();
        //console.log(weights)
        const inpsize = info.inputShape[1];
        assert$2(weights.length == inpsize, "IH");
        assert$2(weights[0].length == config.units, "UN");
        const mi = info.model;
        const weightsIdx = weightOffset(mi);
        const bias = config.useBias ? info.layer.weights[1].read().arraySync() : null;
        //console.log(bias)
        for (let f = 0; f < config.units; f++) {
            if (bias)
                addBias(mi, bias[f]);
            for (let i = 0; i < inpsize; ++i)
                addWeight(mi, weights[i][f]);
            alignWeights(mi);
        }
        const res = [
            loadWeightAddr(Reg.KernelPtr, weightsIdx),
            loadDataAddr(Reg.OutputPtr, info.outputOff),
            repeat(config.units, () => {
                const res = [];
                // set bias
                if (config.useBias)
                    res.push(load(Reg.S0, 1, Reg.KernelPtr, true));
                else
                    res.push(load0(Reg.S0));
                res.push(loadDataAddr(Reg.InputPtr, info.inputOff));
                const addChunk = (len) => flatten(load(memReg0, len, Reg.InputPtr, true), loadWeight(mi, flashReg0, len), range(len + 1).map(i => [
                    i < len ? vmul(memReg0 + i, memReg0 + i, flashReg0 + i) : null,
                    i >= 1 ? vadd(Reg.S0, Reg.S0, memReg0 + i - 1) : null
                ]));
                const numRep = (inpsize / maxChunk) | 0;
                if (numRep > 0)
                    res.push(repeat(numRep, () => addChunk(maxChunk)));
                const left = inpsize - numRep * maxChunk;
                if (left > 0)
                    pushRange(res, addChunk(left));
                res.push(store(Reg.OutputPtr, Reg.S0, 1, true));
                res.push(relaxWeights());
                return res;
            })
        ];
        addActivation(res, info);
        return res;
    }
    function noop(info) {
        return [];
    }
    function shapeElts(shape) {
        let r = 1;
        for (const s of shape)
            if (s != null)
                r *= s;
        return r;
    }
    function fixupCompileInfo(info) {
        if (info.testable === undefined)
            info.testable = !!info.compile;
        if (!info.compile) {
            if (info.inPlace === undefined)
                info.inPlace = true;
            info.compile = noop;
        }
        if (!info.computePaddedInputShape)
            info.computePaddedInputShape = info => info.inputShape.slice();
    }
    function isInPlace(layer) {
        var _a;
        return !!((_a = compilers[layer.getClassName()]) === null || _a === void 0 ? void 0 : _a.inPlace);
    }
    function isTestable(layer) {
        var _a;
        return !!((_a = compilers[layer.getClassName()]) === null || _a === void 0 ? void 0 : _a.testable);
    }
    function shapeToString(shape) {
        return `[${shape.filter(x => x != null).join(",")}]`;
    }
    function assignLayerInfos(m, opts) {
        if (!inited$1) {
            inited$1 = true;
            Object.values(compilers).forEach(fixupCompileInfo);
        }
        reset();
        if (opts.verbose)
            m.summary();
        const inputShape = m.layers[0].batchInputShape;
        const modelInfo = {
            weightPtr: 0,
            weightBuffer: new Uint8Array(128),
            weightAsm: "",
            inputShape,
            outputShape: null,
            outputOffset: -1,
            arenaSize: -1,
            minArenaSize: -1,
            opts,
            stats: ""
        };
        let maxSize = [shapeElts(inputShape), 0];
        let currIdx = 0;
        let prev;
        let totalMax = maxSize[0];
        const recordMax = (n) => totalMax = Math.max(n, totalMax);
        for (const l of m.layers) {
            const info = getLayerInfo(l);
            info.model = modelInfo;
            if (prev) {
                info.inputShape = prev.outputShape;
            }
            else {
                info.inputShape = inputShape;
            }
            info.outputShape = l.computeOutputShape(info.inputShape);
            const comp = compilers[l.getClassName()];
            const paddedShape = comp ? comp.computePaddedInputShape(info) : info.inputShape.slice();
            info.inputOff = currIdx;
            info.rawInputShape = info.inputShape.slice();
            const paddedElts = shapeElts(paddedShape);
            const needsPadding = shapeElts(info.inputShape) != paddedElts;
            if (needsPadding) {
                currIdx = currIdx == 0 ? 1 : 0;
                info.rawInputOff = info.inputOff;
                info.inputOff = currIdx;
                info.inputShape = paddedShape;
                if (paddedElts > maxSize[currIdx])
                    maxSize[currIdx] = paddedElts;
                recordMax(paddedElts + shapeElts(info.rawInputShape));
            }
            else {
                info.rawInputOff = null;
            }
            const elts = shapeElts(info.outputShape);
            if (isInPlace(l)) {
                recordMax(shapeElts(info.inputShape));
                recordMax(shapeElts(info.outputShape));
            }
            else {
                recordMax(shapeElts(info.inputShape) + shapeElts(info.outputShape));
                currIdx = currIdx == 0 ? 1 : 0;
            }
            info.outputOff = currIdx;
            if (elts > maxSize[currIdx])
                maxSize[currIdx] = elts;
            prev = info;
        }
        modelInfo.outputShape = prev.outputShape;
        // TODO alignment?
        const midOff = maxSize[0];
        for (const l of m.layers) {
            const info = getLayerInfo(l);
            if (info.inputOff)
                info.inputOff = midOff;
            if (info.outputOff)
                info.outputOff = midOff;
            if (info.rawInputOff)
                info.rawInputOff = midOff;
            info.stats = { name: l.name };
        }
        const arenaSize = maxSize[0] + maxSize[1];
        modelInfo.arenaSize = arenaSize;
        modelInfo.minArenaSize = totalMax;
        if (arenaSize > totalMax * 1.2) {
            // TODO
            console.log("possible arena shrink with wiser allocation: " + (arenaSize / totalMax).toFixed(3) + "x");
        }
        return modelInfo;
    }
    function compilePadding(info) {
        const res = [];
        if (info.rawInputOff == null)
            return res;
        const [_batch0, inpy, inpx, numch] = info.rawInputShape;
        const [_batch1, outy, outx, outch] = info.inputShape;
        assert$2(numch == outch);
        const padx = outx - inpx;
        const x0 = padx >> 1;
        const x1 = padx - x0;
        const pady = outy - inpy;
        const y0 = pady >> 1;
        const y1 = pady - y0;
        const numZero = numFPRegs >> 1;
        const numData = numFPRegs - numZero;
        const dataReg = Reg.S0 + numZero;
        res.push(load0(Reg.S0));
        // this is slightly cheaper than loading zeros from memory
        for (let i = 1; i < numZero; ++i)
            res.push(vadd(Reg.S0 + i, Reg.S0, Reg.S0));
        res.push(loadDataAddr(Reg.InputPtr, info.rawInputOff));
        res.push(loadDataAddr(Reg.OutputPtr, info.inputOff));
        const topPad = y0 * outx + x0;
        const linePad = x1 + x0;
        const bottomPad = x1 + y1 * outx;
        res.push(...setZero(topPad));
        res.push(repeat(inpy - 1, () => flatten(copyOver(inpx), setZero(linePad))));
        res.push(...copyOver(inpx));
        res.push(...setZero(bottomPad));
        return res;
        function setZero(n) {
            const res = [];
            n *= numch;
            const leftover = n % numZero;
            const reps = (n - leftover) / numZero;
            if (reps)
                res.push(repeat(reps, () => [
                    store(Reg.OutputPtr, Reg.S0, numZero, true)
                ]));
            if (leftover)
                res.push(store(Reg.OutputPtr, Reg.S0, leftover, true));
            return res;
        }
        function copyOver(n) {
            const res = [];
            n *= numch;
            const leftover = n % numData;
            const reps = (n - leftover) / numData;
            if (reps)
                res.push(repeat(reps, () => [
                    load(dataReg, numData, Reg.InputPtr, true),
                    store(Reg.OutputPtr, dataReg, numData, true)
                ]));
            if (leftover) {
                res.push(load(dataReg, leftover, Reg.InputPtr, true), store(Reg.OutputPtr, dataReg, leftover, true));
            }
            return res;
        }
    }
    function optimizeWithComment(opts, opcodes, stats) {
        if (opts.float16weights)
            opcodes = fixupAndMarkF16(opcodes);
        const c0 = numCycles(opcodes);
        if (opts.optimize)
            opcodes = optimize(opcodes);
        const c1 = numCycles(opcodes);
        stats.unoptimizedCycles += c0;
        stats.optimizedCycles += c1;
        const optRate = 100 * (c0 - c1) / c0;
        const optinfo = c0 ? `${c1} cycles (${optRate.toFixed(1)}% opt)` : "(no computation)";
        if (c0)
            opcodes.unshift(comment(optinfo));
        return { opcodes, optinfo };
    }
    function statsShape(shape) {
        return shape.filter(x => x != null);
    }
    function compileModelCore(m, opts) {
        const modelInfo = assignLayerInfos(m, opts);
        if (opts.optimize === undefined)
            opts.optimize = true;
        const ops = [];
        const layerStats = [];
        const layer0 = getLayerInfo(m.layers[0]);
        const layerN = getLayerInfo(m.layers[m.layers.length - 1]);
        const totalStats = {
            name: "TOTAL",
            inputShape: statsShape(layer0.rawInputShape || layer0.inputShape),
            outputShape: statsShape(layerN.outputShape),
            arenaBytes: 0,
            codeBytes: 0,
            weightBytes: 0,
            unoptimizedCycles: 0,
            optimizedCycles: 0
        };
        for (const l of m.layers) {
            const info = getLayerInfo(l);
            info.stats.unoptimizedCycles = 0;
            info.stats.optimizedCycles = 0;
            info.stats.arenaBytes = 0;
            info.stats.inputShape = statsShape(info.rawInputShape || info.inputShape);
            info.stats.outputShape = statsShape(info.outputShape);
            const statsIdx = layerStats.length;
            layerStats.push(info.stats);
            ops.push([label("begin_" + statsIdx)]);
            if (info.rawInputOff != null) {
                const tmp = optimizeWithComment(opts, compilePadding(info), info.stats);
                ops.push(tmp.opcodes);
                info.stats.arenaBytes = (shapeElts(info.rawInputShape) + shapeElts(info.inputShape)) << 2;
                info.stats.hasPadding = true;
            }
            const cinfo = compilers[l.getClassName()];
            if (cinfo) {
                const size0 = weightOffset(modelInfo);
                const tmp = optimizeWithComment(opts, cinfo.compile(info), info.stats);
                info.stats.weightBytes = (weightOffset(modelInfo) - size0) << 2;
                const shapeinfo = `data: ${shapeToString(info.inputShape)}@${info.inputOff} => ${shapeToString(info.outputShape)}@${info.outputOff}`;
                const infostr = `Layer: ${l.getClassName()}; ${shapeinfo}`;
                tmp.opcodes.unshift(comment(infostr));
                if (opts.verbose)
                    console.log(infostr + " " + tmp.optinfo);
                ops.push(tmp.opcodes);
            }
            else
                unsupported("layer: " + l.getClassName());
            if (info.stats.unoptimizedCycles)
                info.stats.arenaBytes = Math.max(info.stats.arenaBytes, (shapeElts(info.inputShape) + shapeElts(info.outputShape)) << 2);
            totalStats.unoptimizedCycles += info.stats.unoptimizedCycles;
            ops.push([label("end_" + statsIdx)]);
        }
        let flat = flatten(ops);
        const lastInfo = getLayerInfo(m.layers[m.layers.length - 1]);
        modelInfo.outputOffset = lastInfo.outputOff;
        const cycles = numCycles(flat);
        const cycleinfo = `total cycles: ${cycles} (${(cycles / 84000).toFixed(3)}ms at 84MHz)`;
        modelInfo.stats = cycleinfo;
        totalStats.optimizedCycles = cycles;
        if (opts.verbose)
            console.log(modelInfo.stats);
        modelInfo.weightBuffer = modelInfo.weightBuffer.slice(0, modelInfo.weightPtr);
        const js = `
${stringifyComment(modelInfo.stats)}
((weights, mkRuntime) => {
    "use strict";
    const weightOff = ${modelInfo.arenaSize}
    const dataOff = 0
    const mem = new Float32Array(weightOff + ${weightOffset(modelInfo)})
    mem.fill(1000.2342)
    new Uint8Array(mem.buffer).set(weights, weightOff << 2)
    const memU32 = new Uint32Array(mem.buffer)
    const rt = mkRuntime(mem)
    const { softmax, f32 } = rt
    return (inputs => {
        if (inputs.length != ${shapeElts(getLayerInfo(m.layers[0]).rawInputShape)})
            throw new Error("invalid input size")
        mem.set(inputs, dataOff)
        let input, output, kernel
        let ${range(numTmpRegs).map(r => "tmp" + r).join(", ")}
        let ${range(numFPRegs).map(r => "s" + r).join(", ")}

${toJSs(modelInfo, flat)}
        
        return mem.slice(${lastInfo.outputOff}, ${lastInfo.outputOff + shapeElts(lastInfo.outputShape)})
    })
})
`;
        const execute = (eval(js))(modelInfo.weightBuffer, mkRuntime);
        let thumb = "";
        if (opts.includeTest && opts.testOutput && opts.testOutputFromJS) {
            // If requested, embed the output from JS code as reference in Thumb code
            // This is important for float16 - the JS and Thumb should be equivalent
            // but the TF.JS may be further out, as it only does float32
            const prev = opts.testOutput;
            opts.testOutput = execute(opts.testInput);
            thumb = toThumb(modelInfo, flat);
            opts.testOutput = prev;
        }
        else {
            thumb = toThumb(modelInfo, flat);
        }
        const res = {
            execute: execute,
            js,
            thumb,
            machineCode: null,
            options: opts,
            memInfo: null,
            timeInfo: modelInfo.stats,
            stats: {
                total: totalStats,
                layers: layerStats,
            }
        };
        return res;
    }
    function mkRuntime(mem) {
        return {
            softmax: (ptr, len) => {
                let max = mem[ptr];
                for (let i = 1; i < len; ++i)
                    max = Math.max(mem[ptr + i], max);
                let sum = 0;
                for (let i = 0; i < len; ++i)
                    sum += (mem[ptr + i] = Math.exp(mem[ptr + i] - max));
                for (let i = 0; i < len; ++i)
                    mem[ptr + i] /= sum;
            },
            f32: (v) => {
                const arr = new Float32Array(1);
                arr[0] = v;
                return arr[0];
            },
            vcvtb_f32_f16: (v) => float16AsUintToFloat(v & 0xffff),
            vcvtt_f32_f16: (v) => float16AsUintToFloat((v >> 16) & 0xffff),
        };
    }
    /**
     * Split model into single-layer models for testing.
     */
    function partialModels(m, opts) {
        var _a;
        return __asyncGenerator(this, arguments, function* partialModels_1() {
            let mod;
            yield __await(m.save({
                save: m => {
                    mod = m;
                    const res = {
                        modelArtifactsInfo: {
                            dateSaved: new Date(),
                            modelTopologyType: "JSON"
                        }
                    };
                    return Promise.resolve(res);
                }
            }));
            delete mod.weightData;
            delete mod.weightSpecs;
            const cfg = (_a = mod.modelTopology) === null || _a === void 0 ? void 0 : _a.config;
            const layersJson = (cfg === null || cfg === void 0 ? void 0 : cfg.layers) || [];
            for (let i = 0; i < m.layers.length; ++i) {
                const layerJson = layersJson[i];
                const layer = m.layers[i];
                const info = getLayerInfo(layer);
                if ((layerJson === null || layerJson === void 0 ? void 0 : layerJson.class_name) != layer.getClassName())
                    throw new Error("invalid serialization");
                if (!isTestable(layer))
                    continue;
                const lcfg = layerJson.config;
                lcfg.batch_input_shape = info.inputShape;
                cfg.layers = [layerJson];
                const copy = yield __await(tf.loadLayersModel({ load: () => Promise.resolve(mod) }));
                console.log(`testing ${layer.getClassName()}: ${shapeToString(info.inputShape)} => ${shapeToString(info.outputShape)}...`);
                yield yield __await(copy);
                layerJson.config.batch_input_shape = info.inputShape;
                // also test it without activation
                if (lcfg.activation) {
                    lcfg.activation = null;
                    const withoutAct = yield __await(tf.loadLayersModel({ load: () => Promise.resolve(mod) }));
                    console.log(`also with no activation...`);
                    yield yield __await(withoutAct);
                }
            }
        });
    }

    /* Docs:
     *
     * Thumb 16-bit Instruction Set Quick Reference Card
     *   http://infocenter.arm.com/help/topic/com.arm.doc.qrc0006e/QRC0006_UAL16.pdf
     *
     * ARMv6-M Architecture Reference Manual (bit encoding of instructions)
     *   http://ecee.colorado.edu/ecen3000/labs/lab3/files/DDI0419C_arm_architecture_v6m_reference_manual.pdf
     *
     * The ARM-THUMB Procedure Call Standard
     *   http://www.cs.cornell.edu/courses/cs414/2001fa/armcallconvention.pdf
     *
     * Cortex-M0 Technical Reference Manual: 3.3. Instruction set summary (cycle counts)
     *   http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0432c/CHDCICDF.html  // M0
     *   http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0484c/CHDCICDF.html  // M0+
     */
    const thumbRegs = {
        "r0": 0,
        "r1": 1,
        "r2": 2,
        "r3": 3,
        "r4": 4,
        "r5": 5,
        "r6": 6,
        "r7": 7,
        "r8": 8,
        "r9": 9,
        "r10": 10,
        "r11": 11,
        "r12": 12,
        "sp": 13,
        "r13": 13,
        "lr": 14,
        "r14": 14,
        "pc": 15,
        "r15": 15,
    };
    const armConditions = {
        "eq": 0,
        "ne": 1,
        "cs": 2,
        "hs": 2,
        "cc": 3,
        "lo": 3,
        "mi": 4,
        "pl": 5,
        "vs": 6,
        "vc": 7,
        "hi": 8,
        "ls": 9,
        "ge": 10,
        "lt": 11,
        "gt": 12,
        "le": 13,
        "": 14,
        "al": 14,
    };
    let fpRegs;
    class ThumbProcessor extends AbstractProcessor {
        constructor() {
            super();
            this.runtimeIsARM = false;
            if (!fpRegs) {
                fpRegs = {};
                for (let i = 0; i < 32; ++i)
                    fpRegs["s" + i] = i;
            }
            const allConds = (f, inclAl = false) => {
                for (const k of Object.keys(armConditions))
                    if (armConditions[k] != 14 || inclAl)
                        f(k, armConditions[k]);
            };
            // Registers
            // $r0 - bits 2:1:0
            // $r1 - bits 5:4:3
            // $r2 - bits 7:2:1:0
            // $r3 - bits 6:5:4:3
            // $r4 - bits 8:7:6
            // $r5 - bits 10:9:8
            this.addEnc("$r0", "R0-7", v => this.inrange(7, v, v));
            this.addEnc("$r1", "R0-7", v => this.inrange(7, v, v << 3));
            this.addEnc("$r2", "R0-15", v => this.inrange(15, v, (v & 7) | ((v & 8) << 4)));
            this.addEnc("$r3", "R0-15", v => this.inrange(15, v, v << 3));
            this.addEnc("$r4", "R0-7", v => this.inrange(7, v, v << 6));
            this.addEnc("$r5", "R0-7", v => this.inrange(7, v, v << 8));
            // this for setting both $r0 and $r1 (two argument adds and subs)
            this.addEnc("$r01", "R0-7", v => this.inrange(7, v, (v | v << 3)));
            // Immdiates:
            // $i0 - bits 7-0
            // $i1 - bits 7-0 * 4
            // $i2 - bits 6-0 * 4
            // $i3 - bits 8-6
            // $i4 - bits 10-6
            // $i5 - bits 10-6 * 4
            // $i6 - bits 10-6, 0 is 32
            // $i7 - bits 10-6 * 2
            this.addEnc("$i0", "#0-255", v => this.inrange(255, v, v));
            this.addEnc("$i1", "#0-1020", v => this.inrange(255, v / 4, v >> 2));
            this.addEnc("$i2", "#0-510", v => this.inrange(127, v / 4, v >> 2));
            this.addEnc("$i3", "#0-7", v => this.inrange(7, v, v << 6));
            this.addEnc("$i4", "#0-31", v => this.inrange(31, v, v << 6));
            this.addEnc("$i5", "#0-124", v => this.inrange(31, v / 4, (v >> 2) << 6));
            this.addEnc("$i6", "#1-32", v => v == 0 ? null : v == 32 ? 0 : this.inrange(31, v, v << 6));
            this.addEnc("$i7", "#0-62", v => this.inrange(31, v / 2, (v >> 1) << 6));
            this.addEnc("$i32", "#0-2^32", v => 1);
            this.addEnc("$rl0", "{R0-7,...}", v => this.inrange(255, v, v));
            this.addEnc("$rl1", "{LR,R0-7,...}", v => (v & 0x4000) ? this.inrange(255, (v & ~0x4000), 0x100 | (v & 0xff)) : this.inrange(255, v, v));
            this.addEnc("$rl2", "{PC,R0-7,...}", v => (v & 0x8000) ? this.inrange(255, (v & ~0x8000), 0x100 | (v & 0xff)) : this.inrange(255, v, v));
            this.addEnc("$la", "LABEL", v => this.inrange(255, v / 4, v >> 2)).isWordAligned = true;
            this.addEnc("$lb", "LABEL", v => this.inrangeSigned(127, v / 2, v >> 1));
            this.addEnc("$lb11", "LABEL", v => this.inrangeSigned(1023, v / 2, v >> 1));
            //this.addInst("nop",                   0xbf00, 0xffff);  // we use mov r8,r8 as gcc
            this.addInst("adcs  $r0, $r1", 0x4140, 0xffc0);
            this.addInst("add   $r2, $r3", 0x4400, 0xff00);
            this.addInst("add   $r5, pc, $i1", 0xa000, 0xf800);
            this.addInst("add   $r5, sp, $i1", 0xa800, 0xf800);
            this.addInst("add   sp, $i2", 0xb000, 0xff80).canBeShared = true;
            this.addInst("adds  $r0, $r1, $i3", 0x1c00, 0xfe00);
            this.addInst("adds  $r0, $r1, $r4", 0x1800, 0xfe00);
            this.addInst("adds  $r01, $r4", 0x1800, 0xfe00);
            this.addInst("adds  $r5, $i0", 0x3000, 0xf800);
            this.addInst("adr   $r5, $la", 0xa000, 0xf800);
            this.addInst("ands  $r0, $r1", 0x4000, 0xffc0);
            this.addInst("asrs  $r0, $r1", 0x4100, 0xffc0);
            this.addInst("asrs  $r0, $r1, $i6", 0x1000, 0xf800);
            this.addInst("bics  $r0, $r1", 0x4380, 0xffc0);
            this.addInst("bkpt  $i0", 0xbe00, 0xff00);
            this.addInst("blx   $r3", 0x4780, 0xff87);
            this.addInst("bx    $r3", 0x4700, 0xff80);
            this.addInst("cmn   $r0, $r1", 0x42c0, 0xffc0);
            this.addInst("cmp   $r0, $r1", 0x4280, 0xffc0);
            this.addInst("cmp   $r2, $r3", 0x4500, 0xff00);
            this.addInst("cmp   $r5, $i0", 0x2800, 0xf800);
            this.addInst("eors  $r0, $r1", 0x4040, 0xffc0);
            this.addInst("ldmia $r5!, $rl0", 0xc800, 0xf800);
            this.addInst("ldmia $r5, $rl0", 0xc800, 0xf800);
            this.addInst("ldr   $r0, [$r1, $i5]", 0x6800, 0xf800); // this is used for debugger breakpoint - cannot be shared
            this.addInst("ldr   $r0, [$r1, $r4]", 0x5800, 0xfe00);
            this.addInst("ldr   $r5, [pc, $i1]", 0x4800, 0xf800);
            this.addInst("ldr   $r5, $la", 0x4800, 0xf800);
            this.addInst("ldr   $r5, [sp, $i1]", 0x9800, 0xf800).canBeShared = true;
            this.addInst("ldr   $r5, [sp]", 0x9800, 0xf800).canBeShared = true;
            this.addInst("ldrb  $r0, [$r1, $i4]", 0x7800, 0xf800);
            this.addInst("ldrb  $r0, [$r1, $r4]", 0x5c00, 0xfe00);
            this.addInst("ldrh  $r0, [$r1, $i7]", 0x8800, 0xf800);
            this.addInst("ldrh  $r0, [$r1, $r4]", 0x5a00, 0xfe00);
            this.addInst("ldrsb $r0, [$r1, $r4]", 0x5600, 0xfe00);
            this.addInst("ldrsh $r0, [$r1, $r4]", 0x5e00, 0xfe00);
            this.addInst("lsls  $r0, $r1", 0x4080, 0xffc0);
            this.addInst("lsls  $r0, $r1, $i4", 0x0000, 0xf800);
            this.addInst("lsrs  $r0, $r1", 0x40c0, 0xffc0);
            this.addInst("lsrs  $r0, $r1, $i6", 0x0800, 0xf800);
            //this.addInst("mov   $r0, $r1", 0x4600, 0xffc0);
            this.addInst("mov   $r2, $r3", 0x4600, 0xff00);
            this.addInst("movs  $r0, $r1", 0x0000, 0xffc0);
            this.addInst("movs  $r5, $i0", 0x2000, 0xf800);
            this.addInst("muls  $r0, $r1", 0x4340, 0xffc0);
            this.addInst("mvns  $r0, $r1", 0x43c0, 0xffc0);
            this.addInst("negs  $r0, $r1", 0x4240, 0xffc0);
            this.addInst("nop", 0x46c0, 0xffff); // mov r8, r8
            this.addInst("orrs  $r0, $r1", 0x4300, 0xffc0);
            this.addInst("pop   $rl2", 0xbc00, 0xfe00);
            this.addInst("push  $rl1", 0xb400, 0xfe00);
            this.addInst("rev   $r0, $r1", 0xba00, 0xffc0);
            this.addInst("rev16 $r0, $r1", 0xba40, 0xffc0);
            this.addInst("revsh $r0, $r1", 0xbac0, 0xffc0);
            this.addInst("rors  $r0, $r1", 0x41c0, 0xffc0);
            this.addInst("sbcs  $r0, $r1", 0x4180, 0xffc0);
            this.addInst("sev", 0xbf40, 0xffff);
            this.addInst("stm   $r5!, $rl0", 0xc000, 0xf800);
            this.addInst("stmia $r5!, $rl0", 0xc000, 0xf800); // alias for stm
            this.addInst("stmea $r5!, $rl0", 0xc000, 0xf800); // alias for stm
            this.addInst("str   $r0, [$r1, $i5]", 0x6000, 0xf800).canBeShared = true;
            this.addInst("str   $r0, [$r1]", 0x6000, 0xf800).canBeShared = true;
            this.addInst("str   $r0, [$r1, $r4]", 0x5000, 0xfe00);
            this.addInst("str   $r5, [sp, $i1]", 0x9000, 0xf800).canBeShared = true;
            this.addInst("str   $r5, [sp]", 0x9000, 0xf800).canBeShared = true;
            this.addInst("strb  $r0, [$r1, $i4]", 0x7000, 0xf800);
            this.addInst("strb  $r0, [$r1, $r4]", 0x5400, 0xfe00);
            this.addInst("strh  $r0, [$r1, $i7]", 0x8000, 0xf800);
            this.addInst("strh  $r0, [$r1, $r4]", 0x5200, 0xfe00);
            this.addInst("sub   sp, $i2", 0xb080, 0xff80);
            this.addInst("subs  $r0, $r1, $i3", 0x1e00, 0xfe00);
            this.addInst("subs  $r0, $r1, $r4", 0x1a00, 0xfe00);
            this.addInst("subs  $r01, $r4", 0x1a00, 0xfe00);
            this.addInst("subs  $r5, $i0", 0x3800, 0xf800);
            this.addInst("svc   $i0", 0xdf00, 0xff00);
            this.addInst("sxtb  $r0, $r1", 0xb240, 0xffc0);
            this.addInst("sxth  $r0, $r1", 0xb200, 0xffc0);
            this.addInst("tst   $r0, $r1", 0x4200, 0xffc0);
            this.addInst("udf   $i0", 0xde00, 0xff00);
            this.addInst("uxtb  $r0, $r1", 0xb2c0, 0xffc0);
            this.addInst("uxth  $r0, $r1", 0xb280, 0xffc0);
            this.addInst("wfe", 0xbf20, 0xffff);
            this.addInst("wfi", 0xbf30, 0xffff);
            this.addInst("yield", 0xbf10, 0xffff);
            this.addInst("cpsid i", 0xb672, 0xffff);
            this.addInst("cpsie i", 0xb662, 0xffff);
            allConds((cond, id) => this.addInst(`b${cond} $lb`, 0xd000 | (id << 8), 0xff00));
            this.addInst("b     $lb11", 0xe000, 0xf800);
            this.addInst("bal   $lb11", 0xe000, 0xf800);
            // handled specially - 32 bit instruction
            this.addInst("bl    $lb", 0xf000, 0xf800);
            // this is normally emitted as 'b' but will be emitted as 'bl' if needed
            this.addInst("bb    $lb", 0xe000, 0xf800);
            // this will emit as PC-relative LDR or ADDS
            this.addInst("ldlit   $r5, $i32", 0x4800, 0xf800);
            // 32 bit encodings
            this.addEnc("$RL0", "{R0-15,...}", v => this.inrange(0xffff, v, v));
            this.addEnc("$R0", "R0-15", v => this.inrange(15, v, v << 8)); // 8-11
            this.addEnc("$R1", "R0-15", v => this.inrange(15, v, v << 16)); // 16-19
            this.addEnc("$R2", "R0-15", v => this.inrange(15, v, v << 12)); // 12-15
            this.addEnc("$R3", "R0-15", v => this.inrange(15, v, v << 0)); // 0-3
            this.addEnc("$I0", "#0-4095", v => this.inrange(4095, v, (v & 0xff) | ((v & 0x700) << 4) | ((v & 0x800) << 15)));
            this.addEnc("$I1", "#0-4095", v => this.inrange(4095, v, v));
            this.addEnc("$I2", "#0-65535", v => this.inrange(0xffff, v, (v & 0xff) | ((v & 0x700) << 4) | ((v & 0x800) << 15) | ((v & 0xf000) << 4)));
            this.addEnc("$I3", "#0-31", v => this.inrange(31, v, ((v & 3) << 6) | ((v >> 2) << 12)));
            this.addEnc("$LB", "LABEL", v => {
                const q = ((v >> 1) & 0x7ff)
                    | (((v >> 12) & 0x3f) << 16)
                    | (((v >> 18) & 0x1) << 13)
                    | (((v >> 19) & 0x1) << 11)
                    | (((v >> 20) & 0x1) << 26);
                if (this.inrangeSigned((1 << 20) - 1, v / 2, q) == null)
                    return null;
                return q;
            });
            this.addEnc("$S0", "S0-31", v => this.inrange(31, v, ((v >> 1) << 0) | ((v & 1) << 5))); // 0-3 + 5
            this.addEnc("$S1", "S0-31", v => this.inrange(31, v, ((v >> 1) << 12) | ((v & 1) << 22))); // 12-15 + 22
            this.addEnc("$S2", "S0-31", v => this.inrange(31, v, ((v >> 1) << 16) | ((v & 1) << 7))); // 16-19 + 7
            this.addEnc("$SL0", "{S0-S31}", v => {
                v |= 0;
                if (!v)
                    return null;
                let reg0 = 0;
                while (reg0 < 32 && 0 == (v & (1 << reg0)))
                    reg0++;
                v >>>= reg0;
                if (!v)
                    return null;
                let num = 0;
                while (v & 1) {
                    v >>= 1;
                    num++;
                }
                if (v)
                    return null; // non-consecutive
                v = reg0;
                // console.log(v0.toString(16), v, num)
                return ((v >> 1) << 12) | ((v & 1) << 22) | num;
            });
            this.addInst32("push  $RL0", 0xe92d0000, 0xffff0000);
            this.addInst32("pop   $RL0", 0xe8bd0000, 0xffff0000);
            this.addInst32("addw  $R0, $R1, $I0", 0xf2000000, 0xfbf08000);
            this.addInst32("subw  $R0, $R1, $I0", 0xf2a00000, 0xfbf08000);
            this.addInst32("ldr   $R2, [$R1, $I1]", 0xf8d00000, 0xfff00000);
            this.addInst32("str   $R2, [$R1, $I1]", 0xf8c00000, 0xfff00000);
            this.addInst32("movw  $R0, $I2", 0xf2400000, 0xfbf08000);
            this.addInst32("add   $R0, $R1, $R3, lsl $I3", 0xeb000000, 0xfff08000);
            // encoding $i0 is only a subset of allowed constants
            this.addInst32("subs  $R0, $R1, $i0", 0xf1b00000, 0xfff08000);
            this.addInst32("sub   $R0, $R1, $i0", 0xf1a00000, 0xfff08000);
            this.addInst32("adds  $R0, $R1, $i0", 0xf1100000, 0xfff08000);
            this.addInst32("add   $R0, $R1, $i0", 0xf1000000, 0xfff08000);
            allConds((cond, id) => this.addInst32(`b${cond} $LB`, 0xf0008000 | (id << 22), 0xfbc0d000), true);
            allConds((cond, id) => this.addInst(`it ${cond}`, 0xbf08 | (id << 4), 0xffff), true);
            this.addInst32("vabs.f32     $S1, $S0", 0xeeb00ac0, 0xffbf0fd0);
            this.addInst32("vadd.f32     $S1, $S2, $S0", 0xee300a00, 0xffb00f50);
            this.addInst32("vmul.f32     $S1, $S2, $S0", 0xee200a00, 0xffb00f50);
            this.addInst32("vcmpe.f32    $S1, #0.0", 0xeeb50ac0, 0xffbf0ff0);
            this.addInst32("vcmpe.f32    $S1, $S0", 0xeeb40ac0, 0xffbf0fd0);
            this.addInst32("vcmp.f32     $S1, #0.0", 0xeeb50a40, 0xffbf0ff0);
            this.addInst32("vcmp.f32     $S1, $S0", 0xeeb40a40, 0xffbf0fd0);
            this.addInst32("vdiv.f32     $S1, $S2, $S0", 0xee800a00, 0xffb00f50);
            this.addInst32("vfma.f32     $S1, $S2, $S0", 0xeea00a00, 0xffb00f50);
            this.addInst32("vfms.f32     $S1, $S2, $S0", 0xeea00a40, 0xffb00f50);
            this.addInst32("vfnma.f32    $S1, $S2, $S0", 0xee900a40, 0xffb00f50);
            this.addInst32("vfnms.f32    $S1, $S2, $S0", 0xee900a00, 0xffb00f50);
            this.addInst32("vmla.f32     $S1, $S2, $S0", 0xe2000d10, 0xffb00f10);
            this.addInst32("vmls.f32     $S1, $S2, $S0", 0xe2200d10, 0xffb00f10);
            this.addInst32("vneg.f32     $S1, $S0", 0xeeb10a40, 0xffbf0fd0);
            this.addInst32("vsqrt.f32    $S1, $S0", 0xeeb10ac0, 0xffbf0fd0);
            this.addInst32("vsub.f32     $S1, $S2, $S0", 0xee300a40, 0xffb00f50);
            this.addInst32("vstmdb       $R1!, $SL0", 0xed200a00, 0xffb00f00);
            this.addInst32("vstmia       $R1!, $SL0", 0xeca00a00, 0xffb00f00);
            this.addInst32("vstmia       $R1, $SL0", 0xec800a00, 0xffb00f00);
            this.addInst32("vstm         $R1!, $SL0", 0xeca00a00, 0xffb00f00);
            this.addInst32("vstm         $R1, $SL0", 0xec800a00, 0xffb00f00);
            this.addInst32("vldmdb       $R1!, $SL0", 0xed300a00, 0xffb00f00);
            this.addInst32("vldmia       $R1!, $SL0", 0xecb00a00, 0xffb00f00);
            this.addInst32("vldmia       $R1, $SL0", 0xec900a00, 0xffb00f00);
            this.addInst32("vldm         $R1!, $SL0", 0xecb00a00, 0xffb00f00);
            this.addInst32("vldm         $R1, $SL0", 0xec900a00, 0xffb00f00);
            this.addInst32("vldr         $S1, [$R1, $i1]", 0xed900a00, 0xffb00f00);
            this.addInst32("vstr         $S1, [$R1, $i1]", 0xed800a00, 0xffb00f00);
            this.addInst32("vldr         $S1, [$R1]", 0xed900a00, 0xffb00f00);
            this.addInst32("vmrs         APSR_nzcv, fpscr", 0xeef1fa10, 0xffffffff);
            this.addInst32("vmrs         APSR_nzcv, FPSCR", 0xeef1fa10, 0xffffffff);
            this.addInst32("vmov.f32     $S1, $S0", 0xeeb00a40, 0xffbf0fd0);
            this.addInst32("vmov         $S2, $R2", 0xee000a10, 0xfff00f7f);
            this.addInst32("vmov         $R2, $S2", 0xee100a10, 0xfff00f7f);
            this.addInst32("vldr         $S1, $la", 0xed9f0a00, 0xffbf0f00);
            this.addInst32("vmov.f32     $S1, #1.0", 0xeeb70a00, 0xffbf0ff0);
            this.addInst32("vcvt.s32.f32 $S1, $S0", 0xeebd0ac0, 0xffbf0fd0);
            this.addInst32("vcvtb.f32.f16 $S1, $S0", 0xeeb20a40, 0xffbf0fd0);
            this.addInst32("vcvtt.f32.f16 $S1, $S0", 0xeeb20ac0, 0xffbf0fd0);
            this.addInst32("vcvtb.f16.f32 $S1, $S0", 0xeeb30a40, 0xffbf0fd0);
            this.addInst32("vcvtt.f16.f32 $S1, $S0", 0xeeb30ac0, 0xffbf0fd0);
            /*
            vmsr
            vpush
            vpop
            vrint
            vsel
            */
        }
        stripCondition(name) {
            if (name.length >= 5) {
                const dot = name.indexOf(".");
                let suff = "";
                let force = false;
                if (dot > 0) {
                    suff = name.slice(dot);
                    name = name.slice(0, dot);
                    if (suff == ".32") {
                        force = true;
                        suff = "";
                    }
                }
                if (armConditions[name.slice(-2)])
                    return name.slice(0, -2) + suff;
                if (force)
                    return name;
            }
            return null;
        }
        toFnPtr(v, baseOff, lbl) {
            if (this.runtimeIsARM && /::/.test(lbl))
                return (v + baseOff) & ~1;
            return (v + baseOff) | 1;
        }
        wordSize() {
            return 4;
        }
        is32bit(i) {
            return i.name == "bl" || i.name == "bb" || i.is32bit;
        }
        postProcessAbsAddress(f, v) {
            // Thumb addresses have last bit set, but we are ourselves always
            // in Thumb state, so to go to ARM state, we signal that with that last bit
            v ^= 1;
            v -= f.baseOffset;
            return v;
        }
        emit32(v0, v, actual) {
            let isBLX = v % 2 ? true : false;
            if (isBLX) {
                v = (v + 1) & ~3;
            }
            let off = v >> 1;
            assert(off != null);
            // Range is +-4M (i.e., 2M instructions)
            if ((off | 0) != off ||
                !(-2 * 1024 * 1024 < off && off < 2 * 1024 * 1024))
                return emitErr("jump out of range", actual);
            // note that off is already in instructions, not bytes
            let imm11 = off & 0x7ff;
            let imm10 = (off >> 11) & 0x3ff;
            return {
                opcode: (off & 0xf0000000) ? (0xf400 | imm10) : (0xf000 | imm10),
                opcode2: isBLX ? (0xe800 | imm11) : (0xf800 | imm11),
                stack: 0,
                numArgs: [v],
                labelName: actual
            };
        }
        expandLdlit(f) {
            let nextGoodSpot;
            let needsJumpOver = false;
            let outlines = [];
            let values = {};
            let seq = 1;
            for (let i = 0; i < f.lines.length; ++i) {
                let line = f.lines[i];
                outlines.push(line);
                if (line.type == "instruction" && line.instruction && line.instruction.name == "ldlit") {
                    if (!nextGoodSpot) {
                        let limit = line.location + 900; // leave some space - real limit is 1020
                        let j = i + 1;
                        for (; j < f.lines.length; ++j) {
                            if (f.lines[j].location > limit)
                                break;
                            let op = f.lines[j].getOp();
                            if (op == "b" || op == "bb" || (op == "pop" && f.lines[j].words[2] == "pc"))
                                nextGoodSpot = f.lines[j];
                        }
                        if (nextGoodSpot) {
                            needsJumpOver = false;
                        }
                        else {
                            needsJumpOver = true;
                            while (--j > i) {
                                if (f.lines[j].type == "instruction") {
                                    nextGoodSpot = f.lines[j];
                                    break;
                                }
                            }
                        }
                    }
                    let reg = line.words[1];
                    // make sure the key in values[] below doesn't look like integer
                    // we rely on Object.keys() returning stuff in insertion order, and integers mess with it
                    // see https://www.ecma-international.org/ecma-262/6.0/#sec-ordinary-object-internal-methods-and-internal-slots-ownpropertykeys
                    // or possibly https://www.stefanjudis.com/today-i-learned/property-order-is-predictable-in-javascript-objects-since-es2015/
                    let v = "#" + line.words[3];
                    let lbl = lookup(values, v);
                    if (!lbl) {
                        lbl = "_ldlit_" + ++seq;
                        values[v] = lbl;
                    }
                    line.update(`ldr ${reg}, ${lbl}`);
                }
                if (line === nextGoodSpot) {
                    nextGoodSpot = null;
                    let txtLines = [];
                    let jmplbl = "_jmpwords_" + ++seq;
                    if (needsJumpOver)
                        txtLines.push("bb " + jmplbl);
                    txtLines.push(".balign 4");
                    for (let v of Object.keys(values)) {
                        let lbl = values[v];
                        txtLines.push(lbl + ": .word " + v.slice(1));
                    }
                    if (needsJumpOver)
                        txtLines.push(jmplbl + ":");
                    for (let t of txtLines) {
                        f.buildLine(t, outlines);
                        let ll = outlines[outlines.length - 1];
                        ll.scope = line.scope;
                        ll.lineNo = line.lineNo;
                    }
                    values = {};
                }
            }
            f.lines = outlines;
        }
        getAddressFromLabel(f, i, s, wordAligned = false) {
            let l = f.lookupLabel(s);
            if (l == null)
                return null;
            let pc = f.location() + 4;
            if (wordAligned)
                pc = pc & 0xfffffffc;
            return l - pc;
        }
        isPop(opcode) {
            return opcode == 0xbc00;
        }
        isPush(opcode) {
            return opcode == 0xb400;
        }
        isAddSP(opcode) {
            return opcode == 0xb000;
        }
        isSubSP(opcode) {
            return opcode == 0xb080;
        }
        peephole(ln, lnNext, lnNext2) {
            let lb11 = this.encoders["$lb11"];
            let lb = this.encoders["$lb"];
            // +/-8 bytes is because the code size can slightly change due to .balign directives
            // inserted by literal generation code; see https://github.com/Microsoft/pxt-adafruit/issues/514
            // Most likely 4 would be enough, but we play it safe
            function fits(enc, ln) {
                return (enc.encode(ln.numArgs[0] + 8) != null &&
                    enc.encode(ln.numArgs[0] - 8) != null &&
                    enc.encode(ln.numArgs[0]) != null);
            }
            let lnop = ln.getOp();
            let isSkipBranch = false;
            if (lnop == "bne" || lnop == "beq") {
                if (lnNext.getOp() == "b" && ln.numArgs[0] == 0)
                    isSkipBranch = true;
                if (lnNext.getOp() == "bb" && ln.numArgs[0] == 2)
                    isSkipBranch = true;
            }
            if (lnop == "bb" && fits(lb11, ln)) {
                // RULE: bb .somewhere -> b .somewhere (if fits)
                ln.update("b " + ln.words[1]);
            }
            else if (lnop == "b" && ln.numArgs[0] == -2) {
                // RULE: b .somewhere; .somewhere: -> .somewhere:
                ln.update("");
            }
            else if (lnop == "bne" && isSkipBranch && fits(lb, lnNext)) {
                // RULE: bne .next; b .somewhere; .next: -> beq .somewhere
                ln.update("beq " + lnNext.words[1]);
                lnNext.update("");
            }
            else if (lnop == "beq" && isSkipBranch && fits(lb, lnNext)) {
                // RULE: beq .next; b .somewhere; .next: -> bne .somewhere
                ln.update("bne " + lnNext.words[1]);
                lnNext.update("");
            }
            else if (lnop == "push" && ln.numArgs[0] == 0x4000 && lnNext.getOp() == "push" && !(lnNext.numArgs[0] & 0x4000)) {
                // RULE: push {lr}; push {X, ...} -> push {lr, X, ...}
                ln.update(lnNext.text.replace("{", "{lr, "));
                lnNext.update("");
            }
            else if (lnop == "pop" && lnNext.getOp() == "pop" && lnNext.numArgs[0] == 0x8000) {
                // RULE: pop {X, ...}; pop {pc} -> push {X, ..., pc}
                ln.update(ln.text.replace("}", ", pc}"));
                lnNext.update("");
            }
            else if (lnop == "push" && lnNext.getOp() == "pop" && ln.numArgs[0] == lnNext.numArgs[0]) {
                // RULE: push {X}; pop {X} -> nothing
                assert(ln.numArgs[0] > 0);
                ln.update("");
                lnNext.update("");
            }
            else if (lnop == "push" && lnNext.getOp() == "pop" &&
                ln.words.length == 4 &&
                lnNext.words.length == 4) {
                // RULE: push {rX}; pop {rY} -> mov rY, rX
                assert(ln.words[1] == "{");
                ln.update("mov " + lnNext.words[2] + ", " + ln.words[2]);
                lnNext.update("");
            }
            else if (lnNext2 && ln.getOpExt() == "movs $r5, $i0" && lnNext.getOpExt() == "mov $r0, $r1" &&
                ln.numArgs[0] == lnNext.numArgs[1] &&
                clobbersReg(lnNext2, ln.numArgs[0])) {
                // RULE: movs rX, #V; mov rY, rX; clobber rX -> movs rY, #V
                ln.update("movs r" + lnNext.numArgs[0] + ", #" + ln.numArgs[1]);
                lnNext.update("");
            }
            else if (lnop == "pop" && singleReg(ln) >= 0 && lnNext.getOp() == "push" &&
                singleReg(ln) == singleReg(lnNext)) {
                // RULE: pop {rX}; push {rX} -> ldr rX, [sp, #0]
                ln.update("ldr r" + singleReg(ln) + ", [sp, #0]");
                lnNext.update("");
            }
            else if (lnop == "push" && lnNext.getOpExt() == "ldr $r5, [sp, $i1]" &&
                singleReg(ln) == lnNext.numArgs[0] && lnNext.numArgs[1] == 0) {
                // RULE: push {rX}; ldr rX, [sp, #0] -> push {rX}
                lnNext.update("");
            }
            else if (lnNext2 && lnop == "push" && singleReg(ln) >= 0 && preservesReg(lnNext, singleReg(ln)) &&
                lnNext2.getOp() == "pop" && singleReg(ln) == singleReg(lnNext2)) {
                // RULE: push {rX}; movs rY, #V; pop {rX} -> movs rY, #V (when X != Y)
                ln.update("");
                lnNext2.update("");
            }
        }
        registerNo(actual, enc) {
            if (!actual)
                return null;
            actual = actual.toLowerCase();
            let map = thumbRegs;
            if (enc.name[1] == "S") {
                map = fpRegs;
            }
            const r = map[actual];
            if (r === undefined)
                return null;
            return r;
        }
        testAssembler() {
            expectError(this, "lsl r0, r0, #8");
            //assembler.expectError(this, "push {pc,lr}");
            expectError(this, "push {r17}");
            expectError(this, "mov r0, r1 foo");
            expectError(this, "movs r14, #100");
            expectError(this, "push {r0");
            expectError(this, "push lr,r0}");
            //assembler.expectError(this, "pop {lr,r0}");
            expectError(this, "b #+11");
            expectError(this, "b #+10240000");
            expectError(this, "bne undefined_label");
            expectError(this, ".foobar");
            expect(this, "0200      lsls    r0, r0, #8\n" +
                "b500      push    {lr}\n" +
                "2064      movs    r0, #100        ; 0x64\n" +
                "b401      push    {r0}\n" +
                "bc08      pop     {r3}\n" +
                "b501      push    {r0, lr}\n" +
                "bd20      pop {r5, pc}\n" +
                "bc01      pop {r0}\n" +
                "4770      bx      lr\n" +
                "0000      .balign 4\n" +
                "e6c0      .word   -72000\n" +
                "fffe\n");
            expect(this, "4291      cmp     r1, r2\n" +
                "d100      bne     l6\n" +
                "e000      b       l8\n" +
                "1840  l6: adds    r0, r0, r1\n" +
                "4718  l8: bx      r3\n");
            expect(this, "          @stackmark base\n" +
                "b403      push    {r0, r1}\n" +
                "          @stackmark locals\n" +
                "9801      ldr     r0, [sp, locals@1]\n" +
                "b401      push    {r0}\n" +
                "9802      ldr     r0, [sp, locals@1]\n" +
                "bc01      pop     {r0}\n" +
                "          @stackempty locals\n" +
                "9901      ldr     r1, [sp, locals@1]\n" +
                "9102      str     r1, [sp, base@0]\n" +
                "          @stackempty locals\n" +
                "b002      add     sp, #8\n" +
                "          @stackempty base\n");
            expect(this, "b090      sub sp, #4*16\n" +
                "b010      add sp, #4*16\n");
            expect(this, "6261      .string \"abc\"\n" +
                "0063      \n");
            expect(this, "6261      .string \"abcde\"\n" +
                "6463      \n" +
                "0065      \n");
            expect(this, "3042      adds r0, 0x42\n" +
                "1c0d      adds r5, r1, #0\n" +
                "d100      bne #0\n" +
                "2800      cmp r0, #0\n" +
                "6b28      ldr r0, [r5, #48]\n" +
                "0200      lsls r0, r0, #8\n" +
                "2063      movs r0, 0x63\n" +
                "4240      negs r0, r0\n" +
                "46c0      nop\n" +
                "b500      push {lr}\n" +
                "b401      push {r0}\n" +
                "b402      push {r1}\n" +
                "b404      push {r2}\n" +
                "b408      push {r3}\n" +
                "b520      push {r5, lr}\n" +
                "bd00      pop {pc}\n" +
                "bc01      pop {r0}\n" +
                "bc02      pop {r1}\n" +
                "bc04      pop {r2}\n" +
                "bc08      pop {r3}\n" +
                "bd20      pop {r5, pc}\n" +
                "9003      str r0, [sp, #4*3]\n");
        }
    }
    // if true then instruction doesn't write r<n> and doesn't read/write memory
    function preservesReg(ln, n) {
        if (ln.getOpExt() == "movs $r5, $i0" && ln.numArgs[0] != n)
            return true;
        return false;
    }
    function clobbersReg(ln, n) {
        // TODO add some more
        if (ln.getOp() == "pop" && ln.numArgs[0] & (1 << n))
            return true;
        return false;
    }
    function singleReg(ln) {
        assert(ln.getOp() == "push" || ln.getOp() == "pop");
        let k = 0;
        let ret = -1;
        let v = ln.numArgs[0];
        while (v > 0) {
            if (v & 1) {
                if (ret == -1)
                    ret = k;
                else
                    ret = -2;
            }
            v >>= 1;
            k++;
        }
        if (ret >= 0)
            return ret;
        else
            return -1;
    }

    const epsF32 = 0.00002;
    const epsF16 = 0.0045;
    function mkProcessorFile() {
        const b = new File(new ThumbProcessor());
        b.ei.testAssembler(); // just in case
        b.disablePeepHole = true;
        b.lookupExternalLabel = _name => null;
        b.normalizeExternalLabel = s => s;
        b.throwOnError = true;
        return b;
    }
    function throwAssemblerErrors(b) {
        if (b.errors.length > 0) {
            throw new Error(b.errors[0].message);
        }
    }
    function assemble(src) {
        const procFile = mkProcessorFile();
        procFile.emit(src);
        throwAssemblerErrors(procFile);
        const binary = new Uint8Array(procFile.buf.length << 1);
        for (let i = 0; i < procFile.buf.length; ++i) {
            binary[i << 1] = procFile.buf[i] & 0xff;
            binary[(i << 1) + 1] = (procFile.buf[i] >> 8) & 0xff;
        }
        return { binary, procFile };
    }
    function randomTensor(shape, mult = 1) {
        shape = shape.map(s => s == null ? 1 : s);
        const num = shapeElts(shape);
        return tf.tidy(() => tf.tensor(range(num).map(_ => mult * randomSFloat())).reshape(shape));
    }
    function setRandomWeights(l) {
        for (const w of l.weights) {
            const mult = 1;
            w.write(randomTensor(w.shape, mult));
        }
    }
    function isNear(a, b, eps) {
        const diff = Math.abs(a - b);
        if (diff < eps)
            return true;
        if (diff / (Math.abs(a) + Math.abs(b)) < eps)
            return true;
        return false;
    }
    function optionsWithTestData(m, opts) {
        opts = flatClone(opts);
        let count = 0;
        let maxMul = 0;
        while (true) {
            const randomInput = randomTensor(m.inputs[0].shape);
            const resTensor = m.predict(randomInput);
            const res = resTensor.flatten().arraySync();
            let sum = 0;
            let mul = 1;
            for (const r of res) {
                sum += r;
                mul *= r;
            }
            const isSoftmax = Math.abs(sum - 1) < 0.1;
            if (!isSoftmax) {
                save();
                break;
            }
            if (mul > maxMul) {
                maxMul = mul;
                save();
            }
            if (count++ > (opts.includeTest ? 1000 : 100) || maxMul > 0.1) {
                if (!mul)
                    save();
                break;
            }
            function save() {
                opts.testInput = randomInput.flatten().arraySync();
                opts.testOutput = res;
            }
        }
        return opts;
    }
    function compileModel(m, opts) {
        const cres = compileModelCore(m, opts);
        const ares = assemble(cres.thumb);
        cres.machineCode = ares.binary;
        let idx = 0;
        for (const st of cres.stats.layers) {
            st.codeBytes = ares.procFile.lookupLabel("end_" + idx) - ares.procFile.lookupLabel("begin_" + idx);
            idx++;
        }
        const st = getStatsFromBin(cres.machineCode, cres.stats.total);
        cres.memInfo = st.info;
        return cres;
    }
    async function compileModelAndFullValidate(m, opts) {
        assignLayerInfos(m, opts);
        const optsPart = flatClone(opts);
        optsPart.includeTest = false;
        console.log("Validating partial models...");
        const iter = partialModels(m, optsPart);
        while (true) {
            const m = (await iter.next()).value;
            if (!m)
                break;
            for (const l of m.layers)
                setRandomWeights(l);
            compileAndTest(m, optsPart);
        }
        console.log("Compiling full model...");
        // also test the top-level one again
        return compileAndTest(m, opts);
    }
    function validateCompilation(cres) {
        const opts = cres.options;
        const res = opts.testOutput;
        const res2 = cres.execute(opts.testInput);
        if (cres.options.verbose)
            console.log("Test output", res2);
        let numerr = 0;
        for (let i = 0; i < res2.length; ++i) {
            if (!isNear(res[i], res2[i], opts.float16weights ? epsF16 : epsF32)) {
                console.log(`at ${i} ${res[i]} - ${res2[i]} = ${res[i] - res2[i]}`);
                numerr++;
                if (numerr > 5)
                    break;
            }
        }
        if (numerr)
            throw new Error("mismatch");
    }
    function compileAndTest(m, options) {
        let cres;
        try {
            options = optionsWithTestData(m, options);
            cres = compileModel(m, options);
            validateCompilation(cres);
            return cres;
        }
        catch (e) {
            if (options.info)
                console.log(options.info);
            if (!cres || !options.verbose) {
                options.verbose = true;
                cres = compileModelCore(m, options);
            }
            console.log(cres.js);
            console.log("Failing model: ", m.name);
            throw e;
        }
    }
    function readU32(bin, off) {
        return (bin[off] | (bin[off + 1] << 8) | (bin[off + 2] << 16) | (bin[off + 3] << 24)) >>> 0;
    }
    function readU32s(bin) {
        const res = [];
        for (let i = 0; i < bin.length; i += 4) {
            res.push(readU32(bin, i));
        }
        return res;
    }
    function getStatsFromBin(bin, stats) {
        let [magic0, magic1, hdSize, totalSize, weightsOff, testInpOff, testOutOff, arenaSize] = readU32s(bin.slice(0, 64));
        if (magic0 != 0x30470f62)
            return null;
        const modelSize = testInpOff || totalSize;
        const codeSize = weightsOff - hdSize;
        const codePerc = codeSize * 100 / modelSize;
        const testSize = totalSize - modelSize;
        function sz(n) {
            return (n / 1024).toFixed(2) + "k";
        }
        const info = `model: ${sz(modelSize)}; ` +
            `code: ${sz(codeSize)} (${codePerc.toFixed(1)}%); ` +
            `arena: ${sz(arenaSize)}; test ${sz(testSize)}`;
        if (stats) {
            stats.arenaBytes = arenaSize;
            stats.codeBytes = codeSize;
            stats.weightBytes = modelSize - codeSize;
        }
        return {
            info,
            modelSize,
            codeSize,
            testSize,
            totalSize,
            arenaSize
        };
    }
    function loadTfjsModelJSON(modelJSON) {
        var _a, _b;
        // remove regularizers, as we're not going to train the model, and unknown regularizers
        // cause it to fail to load
        const cfg = (_b = (_a = modelJSON.modelTopology) === null || _a === void 0 ? void 0 : _a.model_config) === null || _b === void 0 ? void 0 : _b.config;
        for (const layer of (cfg === null || cfg === void 0 ? void 0 : cfg.layers) || []) {
            const layerConfig = layer === null || layer === void 0 ? void 0 : layer.config;
            if (layerConfig) {
                layerConfig.bias_regularizer = null;
                layerConfig.activity_regularizer = null;
                layerConfig.bias_constraint = null;
            }
        }
        const model = {
            modelTopology: modelJSON.modelTopology,
            format: modelJSON.format,
            generatedBy: modelJSON.generatedBy,
            convertedBy: modelJSON.convertedBy,
            trainingConfig: modelJSON.trainingConfig,
            userDefinedMetadata: modelJSON.userDefinedMetadata
        };
        return model;
    }
    function loadFlatJSONModel(preModel) {
        if (!preModel.modelJSON)
            return null;
        let modelJSON;
        if (typeof preModel.modelJSON == "string")
            modelJSON = JSON.parse(preModel.modelJSON);
        else
            modelJSON = preModel.modelJSON;
        const model = loadTfjsModelJSON(modelJSON);
        const arr = preModel.weights;
        if (Array.isArray(arr)) {
            model.weightData = new Uint32Array(arr).buffer;
            model.weightSpecs = modelJSON.weightSpecs;
        }
        return model;
    }

    function logThumb(cres) {
        let str = cres.thumb;
        function hex2(n) {
            return ("0" + n.toString(16)).slice(-2);
        }
        str += "// BUF: ";
        for (const v of cres.machineCode)
            str += hex2(v);
        console.log(str);
        console.log(cres.memInfo);
        console.log(cres.timeInfo);
    }
    async function runBrowser(seed) {
        tf.setBackend('cpu');
        const t0 = Date.now();
        seedRandom(seed || 220);
        testFloatConv();
        // const m = await tf.loadLayersModel("./models/gestures.tfjsmodel.json")
        const sample = sampleModel("oneD");
        const float16weights = true;
        const optimize = false;
        const opts = { verbose: true, float16weights, optimize };
        logThumb(compileAndTest(sample, opts));
        await testAllModels({ verbose: false, optimize });
        console.log(Date.now() - t0 + "ms");
    }
    function getSampleModels() {
        return {
            id: [tf.layers.inputLayer({
                    inputShape: [10, 3, 1]
                })],
            conv2d: [tf.layers.conv2d({
                    inputShape: [50, 3, 1],
                    kernelSize: [4, 4],
                    filters: 16,
                    strides: [1, 1],
                    padding: 'same',
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling'
                })],
            dense: [
                tf.layers.flatten({
                    inputShape: [10, 3, 1],
                }),
                tf.layers.dense({
                    units: 5,
                    activation: "softmax",
                })
            ],
            padding: [
                tf.layers.inputLayer({
                    inputShape: [50, 3, 1]
                }),
                tf.layers.conv2d({
                    filters: 16,
                    kernelSize: 4,
                    strides: 1,
                    padding: "same",
                    activation: "relu"
                })
            ],
            dspDense: [
                tf.layers.inputLayer({ inputShape: [33] }),
                tf.layers.dense({ units: 20, activation: "relu" }),
                tf.layers.dense({ units: 10, activation: "relu" }),
                tf.layers.dense({ units: 3, activation: "softmax" }),
            ],
            noDsp: [
                tf.layers.inputLayer({ inputShape: [150] }),
                tf.layers.reshape({ targetShape: [50, 3, 1] }),
                tf.layers.conv2d({ filters: 16, kernelSize: 4, strides: 1, padding: "same", activation: "relu" }),
                tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: "same" }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.conv2d({ filters: 16, kernelSize: 2, strides: 1, padding: "same", activation: "relu" }),
                tf.layers.maxPooling2d({ poolSize: 2, strides: 2, padding: "same" }),
                tf.layers.flatten(),
                tf.layers.dense({ units: 30, activation: "relu" }),
                tf.layers.dense({ units: 3, activation: "softmax" }),
            ],
            tfjsGest: [
                tf.layers.conv2d({
                    inputShape: [50, 3, 1],
                    kernelSize: [4, 3],
                    filters: 16,
                    strides: [1, 1],
                    padding: 'valid',
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling'
                }),
                tf.layers.maxPooling2d({ poolSize: [2, 1], strides: [2, 1] }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.conv2d({
                    kernelSize: [2, 1],
                    filters: 16,
                    strides: 1,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling'
                }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.conv2d({
                    kernelSize: [2, 1],
                    filters: 16,
                    strides: 1,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling'
                }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.flatten(),
                tf.layers.dense({
                    units: 4,
                    kernelInitializer: 'varianceScaling',
                    activation: 'softmax'
                })
            ],
            microSpeech: [
                tf.layers.conv2d({
                    inputShape: [49, 40, 1],
                    kernelSize: [10, 8],
                    filters: 8,
                    padding: "same",
                    activation: "relu",
                    strides: 2,
                }),
                tf.layers.flatten(),
                tf.layers.dense({
                    units: 4,
                    kernelInitializer: 'varianceScaling',
                    activation: 'softmax'
                })
            ],
            oneD: [
                tf.layers.conv1d({
                    inputShape: [50, 4],
                    kernelSize: [4],
                    strides: 1,
                    filters: 16,
                    activation: 'relu'
                }),
                tf.layers.maxPooling1d({ poolSize: [2] }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.conv1d({
                    kernelSize: [2],
                    strides: 1,
                    filters: 16,
                    activation: 'relu'
                }),
                tf.layers.maxPooling1d({ poolSize: [2] }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.conv1d({
                    kernelSize: [2],
                    strides: 1,
                    filters: 16,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.1 }),
                tf.layers.flatten(),
                tf.layers.dense({
                    units: 3,
                    activation: "softmax",
                })
            ]
        };
    }
    let _models;
    function allSampleModels() {
        if (!_models)
            _models = getSampleModels();
        return Object.keys(_models).map(sampleModel);
    }
    function sampleModel(id) {
        const model = tf.sequential();
        model.name = id;
        if (!_models)
            _models = getSampleModels();
        const layers = _models[id];
        if (!layers) {
            let msg = `no such model ${id}; options:\n`;
            for (const name of Object.keys(_models)) {
                msg += `- ${name}: ${_models[name].length} layer(s)\n`;
            }
            throw new Error(msg);
        }
        for (const l of layers)
            model.add(l);
        // make sure weights are deterministic
        for (const l of model.layers)
            setRandomWeights(l);
        return model;
    }
    async function testAllModels(opts) {
        const t0 = Date.now();
        opts = flatClone(opts);
        for (const m of allSampleModels()) {
            console.log(`***\n*** ${m.name}\n***`);
            console.log(opts.float16weights ? "--- F16" : "--- F32");
            await compileModelAndFullValidate(m, opts);
            opts.float16weights = !opts.float16weights;
            console.log(opts.float16weights ? "--- F16" : "--- F32");
            await compileModelAndFullValidate(m, opts);
        }
        console.log(`\n*** All OK (${Date.now() - t0}ms)\n`);
    }
    function flattenSample(s) {
        const res = [];
        const rec = (v) => {
            if (Array.isArray(v))
                v.forEach(rec);
            else if (typeof v == "number")
                res.push(v);
            else
                throw new Error("invalid input");
        };
        rec(s);
        return res;
    }
    function argmax(r) {
        let maxI = 0;
        let max = r[0];
        for (let i = 1; i < r.length; ++i) {
            if (r[i] > max) {
                max = r[i];
                maxI = i;
            }
        }
        return maxI;
    }
    function evalModel(cres, data) {
        let numOK = 0;
        const dim = data.y[0].length;
        const confusion = range(dim).map(_ => range(dim).map(_ => 0));
        for (let i = 0; i < data.x.length; ++i) {
            const predProb = cres.execute(flattenSample(data.x[i]));
            const pred = argmax(predProb);
            const ok = argmax(data.y[i]);
            confusion[pred][ok]++;
            if (pred == ok)
                numOK++;
        }
        let r = "";
        r += `Accuracy: ${(numOK / data.x.length).toFixed(4)}\n`;
        for (let i = 0; i < dim; i++) {
            for (let j = 0; j < dim; j++) {
                r += ("     " + confusion[i][j]).slice(-5);
            }
            r += "\n";
        }
        return r;
    }

    exports.AbstractProcessor = AbstractProcessor;
    exports.File = File;
    exports.Instruction = Instruction;
    exports.Line = Line;
    exports.VMFile = VMFile;
    exports.allSampleModels = allSampleModels;
    exports.asmDeps = asmDeps;
    exports.asmFns = asmFns;
    exports.assemble = assemble;
    exports.assert = assert;
    exports.assignLayerInfos = assignLayerInfos;
    exports.compileAndTest = compileAndTest;
    exports.compileModel = compileModel;
    exports.compileModelAndFullValidate = compileModelAndFullValidate;
    exports.compileModelCore = compileModelCore;
    exports.concat = concat;
    exports.concatArrayLike = concatArrayLike;
    exports.debug = debug;
    exports.emitErr = emitErr;
    exports.endsWith = endsWith;
    exports.evalModel = evalModel;
    exports.expect = expect;
    exports.expectError = expectError;
    exports.flatClone = flatClone;
    exports.float16AsUintToFloat = float16AsUintToFloat;
    exports.float16toUInt16 = float16toUInt16;
    exports.float32ToUInt32 = float32ToUInt32;
    exports.getStatsFromBin = getStatsFromBin;
    exports.iterMap = iterMap;
    exports.lf = lf;
    exports.loadFlatJSONModel = loadFlatJSONModel;
    exports.loadTfjsModelJSON = loadTfjsModelJSON;
    exports.lookup = lookup;
    exports.mapMap = mapMap;
    exports.oops = oops;
    exports.optionsWithTestData = optionsWithTestData;
    exports.partialModels = partialModels;
    exports.pushRange = pushRange;
    exports.randomInclusive = randomInclusive;
    exports.randomPermute = randomPermute;
    exports.randomPick = randomPick;
    exports.randomSFloat = randomSFloat;
    exports.randomUFloat = randomUFloat;
    exports.randomUint32 = randomUint32;
    exports.range = range;
    exports.runBrowser = runBrowser;
    exports.sampleModel = sampleModel;
    exports.seedRandom = seedRandom;
    exports.setRandomWeights = setRandomWeights;
    exports.shapeElts = shapeElts;
    exports.startsWith = startsWith;
    exports.testAllModels = testAllModels;
    exports.testFloatConv = testFloatConv;
    exports.tohex = tohex;
    exports.userError = userError;
    exports.validateCompilation = validateCompilation;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=ml4f.js.map
