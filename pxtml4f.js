(() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined")
      return require.apply(this, arguments);
    throw new Error('Dynamic require of "' + x + '" is not supported');
  });
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));

  // src/driver.ts
  var tf2 = __toESM(__require("@tensorflow/tfjs"));

  // src/float16.ts
  var basetable = new Uint16Array(512);
  var shifttable = new Uint8Array(512);
  var mantissatable = new Uint32Array(2048);
  var offsettable = new Uint16Array(64);
  var exponenttable = new Uint32Array(64);
  var inited = false;
  function init() {
    inited = true;
    for (let i = 0; i < 256; ++i) {
      const e = i - 127;
      if (e < -24) {
        basetable[i | 0] = 0;
        basetable[i | 256] = 32768;
        shifttable[i | 0] = 24;
        shifttable[i | 256] = 24;
      } else if (e < -14) {
        basetable[i | 0] = 1024 >> -e - 14;
        basetable[i | 256] = 1024 >> -e - 14 | 32768;
        shifttable[i | 0] = -e - 1;
        shifttable[i | 256] = -e - 1;
      } else if (e <= 15) {
        basetable[i | 0] = e + 15 << 10;
        basetable[i | 256] = e + 15 << 10 | 32768;
        shifttable[i | 0] = 13;
        shifttable[i | 256] = 13;
      } else if (e < 128) {
        basetable[i | 0] = 31744;
        basetable[i | 256] = 64512;
        shifttable[i | 0] = 24;
        shifttable[i | 256] = 24;
      } else {
        basetable[i | 0] = 31744;
        basetable[i | 256] = 64512;
        shifttable[i | 0] = 13;
        shifttable[i | 256] = 13;
      }
    }
    for (let i = 1; i < 2048; ++i) {
      if (i < 1024)
        mantissatable[i] = convertmantissa(i);
      else
        mantissatable[i] = 939524096 + (i - 1024 << 13);
    }
    exponenttable[32] = 2147483648;
    exponenttable[31] = 1199570944;
    exponenttable[63] = 3347054592;
    for (let i = 1; i <= 30; ++i)
      exponenttable[i] = i << 23;
    for (let i = 33; i <= 62; ++i)
      exponenttable[i] = 2147483648 + (i - 32 << 23);
    for (let i = 1; i < offsettable.length; ++i)
      offsettable[i] = 1024;
    offsettable[32] = 0;
    function convertmantissa(i) {
      let m = i << 13;
      let e = 0;
      while (!(m & 8388608)) {
        e -= 8388608;
        m <<= 1;
      }
      m &= ~8388608;
      e += 947912704;
      return (m | e) >>> 0;
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
    return basetable[f >> 23 & 511] | (f & 8388607) >> shifttable[f >> 23 & 511];
  }
  function float16AsUintToFloat(h) {
    if (!inited)
      init();
    const tmp = mantissatable[offsettable[h >> 10] + (h & 1023)] + exponenttable[h >> 10];
    const buf = new Uint32Array(1);
    buf[0] = tmp;
    return new Float32Array(buf.buffer)[0];
  }

  // src/util.ts
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
  function mapMap(m, f) {
    let r = {};
    Object.keys(m).forEach((k) => r[k] = f(k, m[k]));
    return r;
  }
  function pushRange(trg, src) {
    for (let i = 0; i < src.length; ++i)
      trg.push(src[i]);
  }
  function range(len) {
    let r = [];
    for (let i = 0; i < len; ++i)
      r.push(i);
    return r;
  }
  var seed = 13 * 16777619;
  function randomUint32() {
    let x = seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    seed = x;
    return x;
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

  // src/assembler.ts
  var debug = false;
  function lf(fmt, ...args) {
    return fmt.replace(/{(\d+)}/g, (match, index) => args[+index]);
  }
  var badNameError = emitErr("opcode name doesn't match", "<name>");
  var Instruction = class {
    constructor(ei, format, opcode, mask, is32bit) {
      this.opcode = opcode;
      this.mask = mask;
      this.is32bit = is32bit;
      this.canBeShared = false;
      assert((opcode & mask) == opcode);
      this.ei = ei;
      this.code = format.replace(/\s+/g, " ");
      this.friendlyFmt = format.replace(/\$\w+/g, (m) => {
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
            if (this.ei.isPush(this.opcode))
              stack++;
            else if (this.ei.isPop(this.opcode))
              stack--;
          } else if (enc.isImmediate) {
            actual = actual.replace(/^#/, "");
            v = ln.bin.parseOneInt(actual);
            if (v == null) {
              return emitErr("expecting number", actual);
            } else {
              if (this.ei.isAddSP(this.opcode))
                stack = -(v / this.ei.wordSize());
              else if (this.ei.isSubSP(this.opcode))
                stack = v / this.ei.wordSize();
            }
          } else if (enc.isRegList) {
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
              if (v & 1 << no)
                return emitErr("duplicate register name", actual);
              v |= 1 << no;
              if (this.ei.isPush(this.opcode))
                stack++;
              else if (this.ei.isPop(this.opcode))
                stack--;
              if (tokens[j] == ",")
                j++;
            }
            actual = tokens[j++];
          } else if (enc.isLabel) {
            actual = actual.replace(/^#/, "");
            if (/^[+-]?\d+$/.test(actual)) {
              v = parseInt(actual, 10);
              labelName = "rel" + v;
            } else if (/^0x[0-9a-fA-F]+$/.test(actual)) {
              v = parseInt(actual, 16);
              labelName = "abs" + v;
            } else {
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
                  v = 8;
              }
              v += lbloff;
            }
            if (isSpecial32) {
              bit32_value = v;
              bit32_actual = actual;
              continue;
            }
          } else {
            oops();
          }
          if (v == null)
            return emitErr("didn't understand it", actual);
          numArgs.push(v);
          v = enc.encode(v);
          if (v == null)
            return emitErr("argument out of range or mis-aligned", actual);
          assert((r & v) == 0);
          r |= v;
        } else if (formal == actual) {
        } else {
          return emitErr("expecting " + formal, actual);
        }
      }
      if (tokens[j])
        return emitErr("trailing tokens", tokens[j]);
      if (isSpecial32)
        return this.ei.emit32(r, bit32_value, ln.bin.normalizeExternalLabel(bit32_actual));
      if (this.is32bit)
        return {
          opcode: r >> 16 & 65535 | 32768,
          opcode2: r >> 0 & 65535,
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
  };
  var Line = class {
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
  };
  var File = class {
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
      assert(0 <= op && op <= 65535);
      this.buf.push(op);
    }
    emitOpCode(op) {
      this.emitShort(op);
    }
    location() {
      return this.buf.length * 2;
    }
    pc() {
      return this.location() + this.baseOffset;
    }
    parseOneInt(s) {
      if (!s)
        return null;
      if (/^\d+$/.test(s))
        return parseInt(s, 10);
      const minP = s.indexOf("-");
      if (minP > 0)
        return this.parseOneInt(s.slice(0, minP)) - this.parseOneInt(s.slice(minP + 1));
      let mul = 1;
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
      } else if (s[0] == "+") {
        s = s.slice(1);
      }
      if (/^\d+$/.test(s))
        return mul * parseInt(s, 10);
      if (endsWith(s, "|1")) {
        return this.parseOneInt(s.slice(0, s.length - 2)) | 1;
      }
      if (endsWith(s, "-1")) {
        return this.parseOneInt(s.slice(0, s.length - 2)) - 1;
      }
      if (endsWith(s, "+1")) {
        return this.parseOneInt(s.slice(0, s.length - 2)) + 1;
      }
      let shm = /(.*)>>(\d+)$/.exec(s);
      if (shm) {
        let left = this.parseOneInt(shm[1]);
        let mask = this.baseOffset & ~16777215;
        left &= ~mask;
        return left >> parseInt(shm[2]);
      }
      let v = null;
      if (s[0] == "0") {
        if (s[1] == "x" || s[1] == "X") {
          let m = /^0x([a-f0-9]+)$/i.exec(s);
          if (m)
            v = parseInt(m[1], 16);
        } else if (s[1] == "b" || s[1] == "B") {
          let m = /^0b([01]+)$/i.exec(s);
          if (m)
            v = parseInt(m[1], 2);
        }
      }
      if (s.indexOf("@") >= 0) {
        let m = /^(\w+)@(-?\d+)$/.exec(s);
        if (m) {
          if (mul != 1)
            this.directiveError(lf("multiplication not supported with saved stacks"));
          if (this.stackpointers.hasOwnProperty(m[1])) {
            v = this.ei.wordSize() * this.ei.computeStackOffset(m[1], this.stack - this.stackpointers[m[1]] + parseInt(m[2]));
          } else
            this.directiveError(lf("saved stack not found"));
        }
        m = /^(.*)@(hi|lo|fn)$/.exec(s);
        if (m && this.looksLikeLabel(m[1])) {
          v = this.lookupLabel(m[1], true);
          if (v != null) {
            if (m[2] == "fn") {
              v = this.ei.toFnPtr(v, this.baseOffset, m[1]);
            } else {
              v >>= 1;
              if (0 <= v && v <= 65535) {
                if (m[2] == "hi")
                  v = v >> 8 & 255;
                else if (m[2] == "lo")
                  v = v & 255;
                else
                  oops();
              } else {
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
      } else if (this.lookupExternalLabel) {
        v = this.lookupExternalLabel(name);
        if (v != null) {
          v = this.ei.postProcessAbsAddress(this, v);
        }
      }
      if (v == null && this.equs.hasOwnProperty(scoped)) {
        v = this.equs[scoped];
      }
      if (v == null && direct) {
        if (this.finalEmit) {
          this.directiveError(lf("unknown label: {0}", name));
        } else
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
        hints
      };
      this.errors.push(err);
      if (this.throwOnError)
        throw new Error(err.message);
    }
    directiveError(msg) {
      this.pushError(msg);
    }
    emitString(l, utf16 = false) {
      function byteAt(s2, i) {
        return (s2.charCodeAt(i) || 0) & 255;
      }
      let m = /^\s*([\w\.]+\s*:\s*)?.\w+\s+(".*")\s*$/.exec(l);
      let s;
      if (!m || null == (s = parseString(m[2]))) {
        this.directiveError(lf("expecting string"));
      } else {
        this.align(2);
        if (utf16) {
          for (let i = 0; i < s.length; i++) {
            this.emitShort(s.charCodeAt(i));
          }
        } else {
          for (let i = 0; i < s.length + 1; i += 2) {
            this.emitShort(byteAt(s, i + 1) << 8 | byteAt(s, i));
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
        } else
          nums.push(n);
        if (words[0] == ",") {
          words.shift();
          if (words[0] == null)
            break;
        } else if (words[0] == null) {
          break;
        } else {
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
        let f = nums[1] & 255;
        f = f | f << 8;
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
        if (0 <= n0 && n1 <= 255 && 0 <= n1 && n0 <= 255)
          this.emitShort(n0 & 255 | (n1 & 255) << 8);
        else
          this.directiveError(lf("expecting uint8"));
      }
    }
    emitHex(words) {
      words.slice(1).forEach((w) => {
        if (w == ",")
          return;
        if (w.length % 4 != 0)
          this.directiveError(".hex needs an even number of bytes");
        else if (!/^[a-f0-9]+$/i.test(w))
          this.directiveError(".hex needs a hex number");
        else
          for (let i = 0; i < w.length; i += 4) {
            let n = parseInt(w.slice(i, i + 4), 16);
            n = (n & 255) << 8 | n >> 8 & 255;
            this.emitShort(n);
          }
      });
    }
    emitFloats(words) {
      words.slice(1).forEach((w) => {
        if (w == ",")
          return;
        const v = parseFloat(w);
        if (isNaN(v))
          this.directiveError("invalid .float");
        const n = float32ToUInt32(v);
        this.emitShort(n & 65535);
        this.emitShort(n >> 16 & 65535);
      });
    }
    emitFloats16(words) {
      words.slice(1).forEach((w) => {
        if (w == ",")
          return;
        const v = parseFloat(w);
        if (isNaN(v))
          this.directiveError("invalid .float16");
        const n = float16toUInt16(v);
        this.emitShort(n & 65535);
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
            } else {
              this.directiveError(lf("expecting 1, 2, 3 or 4 (for 2, 4, 8, or 16 byte alignment)"));
            }
          } else
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
            } else {
              this.directiveError(lf("expecting 2, 4, 8, or 16"));
            }
          } else
            this.directiveError(lf("expecting number"));
          break;
        case ".p2align":
          expectOne();
          num0 = this.parseOneInt(words[1]);
          if (num0 != null) {
            this.align(1 << num0);
          } else
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
          this.parseNumbers(words).forEach((n) => {
            if (-32768 <= n && n <= 65535)
              this.emitShort(n & 65535);
            else
              this.directiveError(lf("expecting int16"));
          });
          break;
        case ".word":
        case ".4bytes":
        case ".long":
          this.parseNumbers(words).forEach((n) => {
            if (-2147483648 <= n && n <= 4294967295) {
              this.emitShort(n & 65535);
              this.emitShort(n >> 16 & 65535);
            } else {
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
          const nums = this.parseNumbers(words.slice(words[2] == "," || words[2] == "=" ? 2 : 1));
          if (nums.length != 1)
            this.directiveError(lf("expecting one value"));
          if (this.equs[words[1]] !== void 0 && this.equs[words[1]] != nums[0])
            this.directiveError(lf("redefinition of {0}", words[1]));
          this.equs[words[1]] = nums[0];
          break;
        case ".startaddr":
          if (this.location())
            this.directiveError(lf(".startaddr can be only be specified at the beginning of the file"));
          expectOne();
          this.baseOffset = this.parseOneInt(words[1]);
          break;
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
          words = words.filter((x) => x != ",");
          words.shift();
          let sz = this.parseOneInt(words[1]);
          let align = 0;
          if (words[2])
            align = this.parseOneInt(words[2]);
          else
            align = 4;
          let val = this.lookupLabel(words[0]);
          if (val == null) {
            if (!this.commPtr) {
              this.commPtr = this.lookupExternalLabel("_pxt_comm_base") || 0;
              if (!this.commPtr)
                this.directiveError(lf("PXT_COMM_BASE not defined"));
            }
            while (this.commPtr & align - 1)
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
        case ".globl":
        case ".local":
          break;
        case "@":
          break;
        default:
          if (/^\.cfi_/.test(words[0])) {
          } else {
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
        possibilities.forEach((i) => {
          let err = i.emit(ln);
          hints += lf("   Maybe: {0} ({1} at '{2}')\n", i.toString(), err.error, err.errorAt);
        });
      }
      this.pushError(lf("assembly error"), hints);
    }
    buildLine(tx, lst) {
      let mkLine = (tx2) => {
        let l2 = new Line(this, tx2);
        l2.scope = this.scope;
        l2.lineNo = this.currLineNo;
        lst.push(l2);
        return l2;
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
          } else {
            return;
          }
        }
      }
      let c0 = w0.charAt(0);
      if (c0 == "." || c0 == "@") {
        l.type = "directive";
        if (l.words[0] == "@scope")
          this.handleDirective(l);
      } else {
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
      text.split(/\r?\n/).forEach((tx) => {
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
      this.lines.forEach((l) => {
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
          } else {
            if (this.labels.hasOwnProperty(lblname))
              this.directiveError(lf("label redefinition"));
            else if (this.inlineMode && /^_/.test(lblname))
              this.directiveError(lf("labels starting with '_' are reserved for the compiler"));
            else {
              this.labels[lblname] = this.location();
            }
          }
          l.location = this.location();
        } else if (l.type == "directive") {
          this.handleDirective(l);
        } else if (l.type == "instruction") {
          this.handleInstruction(l);
        } else if (l.type == "empty") {
        } else {
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
      let totalSize = lenTotal + this.baseOffset & 16777215;
      if (flashSize && totalSize > flashSize)
        userError(lf("program too big by {0} bytes!", totalSize - flashSize));
      flashSize = flashSize || 128 * 1024;
      let totalInfo = lf(
        "; total bytes: {0} ({1}% of {2}k flash with {3} free)",
        totalSize,
        (100 * totalSize / flashSize).toFixed(1),
        (flashSize / 1024).toFixed(1),
        flashSize - totalSize
      );
      let res = lf(
        "; generated code sizes (bytes): {0} (incl. {1} user, {2} helpers, {3} vtables, {4} lits); src size {5}\n",
        lenAllCode,
        lenCode,
        lenHelpers,
        lenVtables,
        lenLiterals,
        lenTotal - lenAllCode
      ) + lf(
        "; assembly: {0} lines; density: {1} bytes/stmt; ({2} stmts)\n",
        this.lines.length,
        Math.round(100 * lenCode / numStmts) / 100,
        numStmts
      ) + totalInfo + "\n" + this.stats + "\n\n";
      let skipOne = false;
      this.lines.forEach((ln, i) => {
        if (ln.words[0] == "_stored_program") {
          res += '_stored_program: .string "..."\n';
          skipOne = true;
          return;
        }
        if (skipOne) {
          skipOne = false;
          return;
        }
        let text = ln.text;
        if (clean) {
          if (ln.words[0] == "@stackempty" && this.lines[i - 1].text == ln.text)
            return;
          text = text.replace(/; WAS: .*/, "");
          if (!text.trim())
            return;
        }
        if (debug) {
          if (ln.type == "label" || ln.type == "instruction")
            text += ` 	; 0x${(ln.location + this.baseOffset).toString(16)}`;
        }
        res += text + "\n";
      });
      return res;
    }
    peepHole() {
      let mylines = this.lines.filter((l) => l.type != "empty");
      for (let i = 0; i < mylines.length; ++i) {
        let ln = mylines[i];
        if (/^user/.test(ln.scope))
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
  };
  var AbstractProcessor = class {
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
        isLabel: /^\$l[a-z]/i.test(n)
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
      let mask = max << 1 | 1;
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
      const high = 2147483648;
      assert(!!(code & high));
      assert(!!(mask & high));
      code &= ~high;
      mask &= ~high;
      return this.addInst(name, code, mask, true);
    }
  };
  function tokenize(line) {
    let words = [];
    let w = "";
    loop:
      for (let i = 0; i < line.length; ++i) {
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
          case "	":
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
    s = s.replace(/\\\\/g, "\\B").replace(/\\(['\?])/g, (f, q) => q).replace(/\\[z0]/g, "\0").replace(/\\x([0-9a-f][0-9a-f])/gi, (f, h) => "\\u00" + h).replace(/\\B/g, "\\\\");
    try {
      return JSON.parse(s);
    } catch (e) {
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
  }
  function tohex(n) {
    if (n < 0 || n > 65535)
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

  // src/thumb.ts
  var thumbRegs = {
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
    "r15": 15
  };
  var armConditions = {
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
    "al": 14
  };
  var fpRegs;
  var ThumbProcessor = class extends AbstractProcessor {
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
      this.addEnc("$r0", "R0-7", (v) => this.inrange(7, v, v));
      this.addEnc("$r1", "R0-7", (v) => this.inrange(7, v, v << 3));
      this.addEnc("$r2", "R0-15", (v) => this.inrange(15, v, v & 7 | (v & 8) << 4));
      this.addEnc("$r3", "R0-15", (v) => this.inrange(15, v, v << 3));
      this.addEnc("$r4", "R0-7", (v) => this.inrange(7, v, v << 6));
      this.addEnc("$r5", "R0-7", (v) => this.inrange(7, v, v << 8));
      this.addEnc("$r01", "R0-7", (v) => this.inrange(7, v, v | v << 3));
      this.addEnc("$i0", "#0-255", (v) => this.inrange(255, v, v));
      this.addEnc("$i1", "#0-1020", (v) => this.inrange(255, v / 4, v >> 2));
      this.addEnc("$i2", "#0-510", (v) => this.inrange(127, v / 4, v >> 2));
      this.addEnc("$i3", "#0-7", (v) => this.inrange(7, v, v << 6));
      this.addEnc("$i4", "#0-31", (v) => this.inrange(31, v, v << 6));
      this.addEnc("$i5", "#0-124", (v) => this.inrange(31, v / 4, v >> 2 << 6));
      this.addEnc("$i6", "#1-32", (v) => v == 0 ? null : v == 32 ? 0 : this.inrange(31, v, v << 6));
      this.addEnc("$i7", "#0-62", (v) => this.inrange(31, v / 2, v >> 1 << 6));
      this.addEnc("$i32", "#0-2^32", (v) => 1);
      this.addEnc("$rl0", "{R0-7,...}", (v) => this.inrange(255, v, v));
      this.addEnc("$rl1", "{LR,R0-7,...}", (v) => v & 16384 ? this.inrange(255, v & ~16384, 256 | v & 255) : this.inrange(255, v, v));
      this.addEnc("$rl2", "{PC,R0-7,...}", (v) => v & 32768 ? this.inrange(255, v & ~32768, 256 | v & 255) : this.inrange(255, v, v));
      this.addEnc("$la", "LABEL", (v) => this.inrange(255, v / 4, v >> 2)).isWordAligned = true;
      this.addEnc("$lb", "LABEL", (v) => this.inrangeSigned(127, v / 2, v >> 1));
      this.addEnc("$lb11", "LABEL", (v) => this.inrangeSigned(1023, v / 2, v >> 1));
      this.addInst("adcs  $r0, $r1", 16704, 65472);
      this.addInst("add   $r2, $r3", 17408, 65280);
      this.addInst("add   $r5, pc, $i1", 40960, 63488);
      this.addInst("add   $r5, sp, $i1", 43008, 63488);
      this.addInst("add   sp, $i2", 45056, 65408).canBeShared = true;
      this.addInst("adds  $r0, $r1, $i3", 7168, 65024);
      this.addInst("adds  $r0, $r1, $r4", 6144, 65024);
      this.addInst("adds  $r01, $r4", 6144, 65024);
      this.addInst("adds  $r5, $i0", 12288, 63488);
      this.addInst("adr   $r5, $la", 40960, 63488);
      this.addInst("ands  $r0, $r1", 16384, 65472);
      this.addInst("asrs  $r0, $r1", 16640, 65472);
      this.addInst("asrs  $r0, $r1, $i6", 4096, 63488);
      this.addInst("bics  $r0, $r1", 17280, 65472);
      this.addInst("bkpt  $i0", 48640, 65280);
      this.addInst("blx   $r3", 18304, 65415);
      this.addInst("bx    $r3", 18176, 65408);
      this.addInst("cmn   $r0, $r1", 17088, 65472);
      this.addInst("cmp   $r0, $r1", 17024, 65472);
      this.addInst("cmp   $r2, $r3", 17664, 65280);
      this.addInst("cmp   $r5, $i0", 10240, 63488);
      this.addInst("eors  $r0, $r1", 16448, 65472);
      this.addInst("ldmia $r5!, $rl0", 51200, 63488);
      this.addInst("ldmia $r5, $rl0", 51200, 63488);
      this.addInst("ldr   $r0, [$r1, $i5]", 26624, 63488);
      this.addInst("ldr   $r0, [$r1, $r4]", 22528, 65024);
      this.addInst("ldr   $r5, [pc, $i1]", 18432, 63488);
      this.addInst("ldr   $r5, $la", 18432, 63488);
      this.addInst("ldr   $r5, [sp, $i1]", 38912, 63488).canBeShared = true;
      this.addInst("ldr   $r5, [sp]", 38912, 63488).canBeShared = true;
      this.addInst("ldrb  $r0, [$r1, $i4]", 30720, 63488);
      this.addInst("ldrb  $r0, [$r1, $r4]", 23552, 65024);
      this.addInst("ldrh  $r0, [$r1, $i7]", 34816, 63488);
      this.addInst("ldrh  $r0, [$r1, $r4]", 23040, 65024);
      this.addInst("ldrsb $r0, [$r1, $r4]", 22016, 65024);
      this.addInst("ldrsh $r0, [$r1, $r4]", 24064, 65024);
      this.addInst("lsls  $r0, $r1", 16512, 65472);
      this.addInst("lsls  $r0, $r1, $i4", 0, 63488);
      this.addInst("lsrs  $r0, $r1", 16576, 65472);
      this.addInst("lsrs  $r0, $r1, $i6", 2048, 63488);
      this.addInst("mov   $r2, $r3", 17920, 65280);
      this.addInst("movs  $r0, $r1", 0, 65472);
      this.addInst("movs  $r5, $i0", 8192, 63488);
      this.addInst("muls  $r0, $r1", 17216, 65472);
      this.addInst("mvns  $r0, $r1", 17344, 65472);
      this.addInst("negs  $r0, $r1", 16960, 65472);
      this.addInst("nop", 18112, 65535);
      this.addInst("orrs  $r0, $r1", 17152, 65472);
      this.addInst("pop   $rl2", 48128, 65024);
      this.addInst("push  $rl1", 46080, 65024);
      this.addInst("rev   $r0, $r1", 47616, 65472);
      this.addInst("rev16 $r0, $r1", 47680, 65472);
      this.addInst("revsh $r0, $r1", 47808, 65472);
      this.addInst("rors  $r0, $r1", 16832, 65472);
      this.addInst("sbcs  $r0, $r1", 16768, 65472);
      this.addInst("sev", 48960, 65535);
      this.addInst("stm   $r5!, $rl0", 49152, 63488);
      this.addInst("stmia $r5!, $rl0", 49152, 63488);
      this.addInst("stmea $r5!, $rl0", 49152, 63488);
      this.addInst("str   $r0, [$r1, $i5]", 24576, 63488).canBeShared = true;
      this.addInst("str   $r0, [$r1]", 24576, 63488).canBeShared = true;
      this.addInst("str   $r0, [$r1, $r4]", 20480, 65024);
      this.addInst("str   $r5, [sp, $i1]", 36864, 63488).canBeShared = true;
      this.addInst("str   $r5, [sp]", 36864, 63488).canBeShared = true;
      this.addInst("strb  $r0, [$r1, $i4]", 28672, 63488);
      this.addInst("strb  $r0, [$r1, $r4]", 21504, 65024);
      this.addInst("strh  $r0, [$r1, $i7]", 32768, 63488);
      this.addInst("strh  $r0, [$r1, $r4]", 20992, 65024);
      this.addInst("sub   sp, $i2", 45184, 65408);
      this.addInst("subs  $r0, $r1, $i3", 7680, 65024);
      this.addInst("subs  $r0, $r1, $r4", 6656, 65024);
      this.addInst("subs  $r01, $r4", 6656, 65024);
      this.addInst("subs  $r5, $i0", 14336, 63488);
      this.addInst("svc   $i0", 57088, 65280);
      this.addInst("sxtb  $r0, $r1", 45632, 65472);
      this.addInst("sxth  $r0, $r1", 45568, 65472);
      this.addInst("tst   $r0, $r1", 16896, 65472);
      this.addInst("udf   $i0", 56832, 65280);
      this.addInst("uxtb  $r0, $r1", 45760, 65472);
      this.addInst("uxth  $r0, $r1", 45696, 65472);
      this.addInst("wfe", 48928, 65535);
      this.addInst("wfi", 48944, 65535);
      this.addInst("yield", 48912, 65535);
      this.addInst("cpsid i", 46706, 65535);
      this.addInst("cpsie i", 46690, 65535);
      allConds((cond, id) => this.addInst(`b${cond} $lb`, 53248 | id << 8, 65280));
      this.addInst("b     $lb11", 57344, 63488);
      this.addInst("bal   $lb11", 57344, 63488);
      this.addInst("bl    $lb", 61440, 63488);
      this.addInst("bb    $lb", 57344, 63488);
      this.addInst("ldlit   $r5, $i32", 18432, 63488);
      this.addEnc("$RL0", "{R0-15,...}", (v) => this.inrange(65535, v, v));
      this.addEnc("$R0", "R0-15", (v) => this.inrange(15, v, v << 8));
      this.addEnc("$R1", "R0-15", (v) => this.inrange(15, v, v << 16));
      this.addEnc("$R2", "R0-15", (v) => this.inrange(15, v, v << 12));
      this.addEnc("$R3", "R0-15", (v) => this.inrange(15, v, v << 0));
      this.addEnc("$I0", "#0-4095", (v) => this.inrange(4095, v, v & 255 | (v & 1792) << 4 | (v & 2048) << 15));
      this.addEnc("$I1", "#0-4095", (v) => this.inrange(4095, v, v));
      this.addEnc("$I2", "#0-65535", (v) => this.inrange(
        65535,
        v,
        v & 255 | (v & 1792) << 4 | (v & 2048) << 15 | (v & 61440) << 4
      ));
      this.addEnc("$I3", "#0-31", (v) => this.inrange(31, v, (v & 3) << 6 | v >> 2 << 12));
      this.addEnc("$LB", "LABEL", (v) => {
        const q = v >> 1 & 2047 | (v >> 12 & 63) << 16 | (v >> 18 & 1) << 13 | (v >> 19 & 1) << 11 | (v >> 20 & 1) << 26;
        if (this.inrangeSigned((1 << 20) - 1, v / 2, q) == null)
          return null;
        return q;
      });
      this.addEnc("$S0", "S0-31", (v) => this.inrange(31, v, v >> 1 << 0 | (v & 1) << 5));
      this.addEnc("$S1", "S0-31", (v) => this.inrange(31, v, v >> 1 << 12 | (v & 1) << 22));
      this.addEnc("$S2", "S0-31", (v) => this.inrange(31, v, v >> 1 << 16 | (v & 1) << 7));
      this.addEnc(
        "$SL0",
        "{S0-S31}",
        (v) => {
          v |= 0;
          const v0 = v;
          if (!v)
            return null;
          let reg0 = 0;
          while (reg0 < 32 && 0 == (v & 1 << reg0))
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
            return null;
          v = reg0;
          return v >> 1 << 12 | (v & 1) << 22 | num;
        }
      );
      this.addInst32("push  $RL0", 3912040448, 4294901760);
      this.addInst32("pop   $RL0", 3904700416, 4294901760);
      this.addInst32("addw  $R0, $R1, $I0", 4060086272, 4226842624);
      this.addInst32("subw  $R0, $R1, $I0", 4070572032, 4226842624);
      this.addInst32("ldr   $R2, [$R1, $I1]", 4174381056, 4293918720);
      this.addInst32("str   $R2, [$R1, $I1]", 4173332480, 4293918720);
      this.addInst32("movw  $R0, $I2", 4064280576, 4226842624);
      this.addInst32("add   $R0, $R1, $R3, lsl $I3", 3942645760, 4293951488);
      this.addInst32("subs  $R0, $R1, $i0", 4054843392, 4293951488);
      this.addInst32("sub   $R0, $R1, $i0", 4053794816, 4293951488);
      this.addInst32("adds  $R0, $R1, $i0", 4044357632, 4293951488);
      this.addInst32("add   $R0, $R1, $i0", 4043309056, 4293951488);
      allConds((cond, id) => this.addInst32(`b${cond} $LB`, 4026564608 | id << 22, 4223717376), true);
      allConds((cond, id) => this.addInst(`it ${cond}`, 48904 | id << 4, 65535), true);
      this.addInst32("vabs.f32     $S1, $S0", 4004514496, 4290711504);
      this.addInst32("vadd.f32     $S1, $S2, $S0", 3996125696, 4289728336);
      this.addInst32("vmul.f32     $S1, $S2, $S0", 3995077120, 4289728336);
      this.addInst32("vcmpe.f32    $S1, #0.0", 4004842176, 4290711536);
      this.addInst32("vcmpe.f32    $S1, $S0", 4004776640, 4290711504);
      this.addInst32("vcmp.f32     $S1, #0.0", 4004842048, 4290711536);
      this.addInst32("vcmp.f32     $S1, $S0", 4004776512, 4290711504);
      this.addInst32("vdiv.f32     $S1, $S2, $S0", 4001368576, 4289728336);
      this.addInst32("vfma.f32     $S1, $S2, $S0", 4003465728, 4289728336);
      this.addInst32("vfms.f32     $S1, $S2, $S0", 4003465792, 4289728336);
      this.addInst32("vfnma.f32    $S1, $S2, $S0", 4002417216, 4289728336);
      this.addInst32("vfnms.f32    $S1, $S2, $S0", 4002417152, 4289728336);
      this.addInst32("vmla.f32     $S1, $S2, $S0", 3791654160, 4289728272);
      this.addInst32("vmls.f32     $S1, $S2, $S0", 3793751312, 4289728272);
      this.addInst32("vneg.f32     $S1, $S0", 4004579904, 4290711504);
      this.addInst32("vsqrt.f32    $S1, $S0", 4004580032, 4290711504);
      this.addInst32("vsub.f32     $S1, $S2, $S0", 3996125760, 4289728336);
      this.addInst32("vstmdb       $R1!, $SL0", 3978299904, 4289728256);
      this.addInst32("vstmia       $R1!, $SL0", 3969911296, 4289728256);
      this.addInst32("vstmia       $R1, $SL0", 3967814144, 4289728256);
      this.addInst32("vstm         $R1!, $SL0", 3969911296, 4289728256);
      this.addInst32("vstm         $R1, $SL0", 3967814144, 4289728256);
      this.addInst32("vldmdb       $R1!, $SL0", 3979348480, 4289728256);
      this.addInst32("vldmia       $R1!, $SL0", 3970959872, 4289728256);
      this.addInst32("vldmia       $R1, $SL0", 3968862720, 4289728256);
      this.addInst32("vldm         $R1!, $SL0", 3970959872, 4289728256);
      this.addInst32("vldm         $R1, $SL0", 3968862720, 4289728256);
      this.addInst32("vldr         $S1, [$R1, $i1]", 3985639936, 4289728256);
      this.addInst32("vstr         $S1, [$R1, $i1]", 3984591360, 4289728256);
      this.addInst32("vldr         $S1, [$R1]", 3985639936, 4289728256);
      this.addInst32("vmrs         APSR_nzcv, fpscr", 4008835600, 4294967295);
      this.addInst32("vmrs         APSR_nzcv, FPSCR", 4008835600, 4294967295);
      this.addInst32("vmov.f32     $S1, $S0", 4004514368, 4290711504);
      this.addInst32("vmov         $S2, $R2", 3992979984, 4293922687);
      this.addInst32("vmov         $R2, $S2", 3994028560, 4293922687);
      this.addInst32("vldr         $S1, $la", 3986622976, 4290711296);
      this.addInst32("vmov.f32     $S1, #1.0", 4004973056, 4290711536);
      this.addInst32("vcvt.s32.f32 $S1, $S0", 4005366464, 4290711504);
      this.addInst32("vcvtb.f32.f16 $S1, $S0", 4004645440, 4290711504);
      this.addInst32("vcvtt.f32.f16 $S1, $S0", 4004645568, 4290711504);
      this.addInst32("vcvtb.f16.f32 $S1, $S0", 4004710976, 4290711504);
      this.addInst32("vcvtt.f16.f32 $S1, $S0", 4004711104, 4290711504);
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
        return v + baseOff & ~1;
      return v + baseOff | 1;
    }
    wordSize() {
      return 4;
    }
    is32bit(i) {
      return i.name == "bl" || i.name == "bb" || i.is32bit;
    }
    postProcessAbsAddress(f, v) {
      v ^= 1;
      v -= f.baseOffset;
      return v;
    }
    emit32(v0, v, actual) {
      let isBLX = v % 2 ? true : false;
      if (isBLX) {
        v = v + 1 & ~3;
      }
      let off = v >> 1;
      assert(off != null);
      if ((off | 0) != off || !(-2 * 1024 * 1024 < off && off < 2 * 1024 * 1024))
        return emitErr("jump out of range", actual);
      let imm11 = off & 2047;
      let imm10 = off >> 11 & 1023;
      return {
        opcode: off & 4026531840 ? 62464 | imm10 : 61440 | imm10,
        opcode2: isBLX ? 59392 | imm11 : 63488 | imm11,
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
            let limit = line.location + 900;
            let j = i + 1;
            for (; j < f.lines.length; ++j) {
              if (f.lines[j].location > limit)
                break;
              let op = f.lines[j].getOp();
              if (op == "b" || op == "bb" || op == "pop" && f.lines[j].words[2] == "pc")
                nextGoodSpot = f.lines[j];
            }
            if (nextGoodSpot) {
              needsJumpOver = false;
            } else {
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
        pc = pc & 4294967292;
      return l - pc;
    }
    isPop(opcode) {
      return opcode == 48128;
    }
    isPush(opcode) {
      return opcode == 46080;
    }
    isAddSP(opcode) {
      return opcode == 45056;
    }
    isSubSP(opcode) {
      return opcode == 45184;
    }
    peephole(ln, lnNext, lnNext2) {
      let lb11 = this.encoders["$lb11"];
      let lb = this.encoders["$lb"];
      function fits(enc, ln2) {
        return enc.encode(ln2.numArgs[0] + 8) != null && enc.encode(ln2.numArgs[0] - 8) != null && enc.encode(ln2.numArgs[0]) != null;
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
        ln.update("b " + ln.words[1]);
      } else if (lnop == "b" && ln.numArgs[0] == -2) {
        ln.update("");
      } else if (lnop == "bne" && isSkipBranch && fits(lb, lnNext)) {
        ln.update("beq " + lnNext.words[1]);
        lnNext.update("");
      } else if (lnop == "beq" && isSkipBranch && fits(lb, lnNext)) {
        ln.update("bne " + lnNext.words[1]);
        lnNext.update("");
      } else if (lnop == "push" && ln.numArgs[0] == 16384 && lnNext.getOp() == "push" && !(lnNext.numArgs[0] & 16384)) {
        ln.update(lnNext.text.replace("{", "{lr, "));
        lnNext.update("");
      } else if (lnop == "pop" && lnNext.getOp() == "pop" && lnNext.numArgs[0] == 32768) {
        ln.update(ln.text.replace("}", ", pc}"));
        lnNext.update("");
      } else if (lnop == "push" && lnNext.getOp() == "pop" && ln.numArgs[0] == lnNext.numArgs[0]) {
        assert(ln.numArgs[0] > 0);
        ln.update("");
        lnNext.update("");
      } else if (lnop == "push" && lnNext.getOp() == "pop" && ln.words.length == 4 && lnNext.words.length == 4) {
        assert(ln.words[1] == "{");
        ln.update("mov " + lnNext.words[2] + ", " + ln.words[2]);
        lnNext.update("");
      } else if (lnNext2 && ln.getOpExt() == "movs $r5, $i0" && lnNext.getOpExt() == "mov $r0, $r1" && ln.numArgs[0] == lnNext.numArgs[1] && clobbersReg(lnNext2, ln.numArgs[0])) {
        ln.update("movs r" + lnNext.numArgs[0] + ", #" + ln.numArgs[1]);
        lnNext.update("");
      } else if (lnop == "pop" && singleReg(ln) >= 0 && lnNext.getOp() == "push" && singleReg(ln) == singleReg(lnNext)) {
        ln.update("ldr r" + singleReg(ln) + ", [sp, #0]");
        lnNext.update("");
      } else if (lnop == "push" && lnNext.getOpExt() == "ldr $r5, [sp, $i1]" && singleReg(ln) == lnNext.numArgs[0] && lnNext.numArgs[1] == 0) {
        lnNext.update("");
      } else if (lnNext2 && lnop == "push" && singleReg(ln) >= 0 && preservesReg(lnNext, singleReg(ln)) && lnNext2.getOp() == "pop" && singleReg(ln) == singleReg(lnNext2)) {
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
      if (r === void 0)
        return null;
      return r;
    }
    testAssembler() {
      expectError(this, "lsl r0, r0, #8");
      expectError(this, "push {r17}");
      expectError(this, "mov r0, r1 foo");
      expectError(this, "movs r14, #100");
      expectError(this, "push {r0");
      expectError(this, "push lr,r0}");
      expectError(this, "b #+11");
      expectError(this, "b #+10240000");
      expectError(this, "bne undefined_label");
      expectError(this, ".foobar");
      expect(
        this,
        "0200      lsls    r0, r0, #8\nb500      push    {lr}\n2064      movs    r0, #100        ; 0x64\nb401      push    {r0}\nbc08      pop     {r3}\nb501      push    {r0, lr}\nbd20      pop {r5, pc}\nbc01      pop {r0}\n4770      bx      lr\n0000      .balign 4\ne6c0      .word   -72000\nfffe\n"
      );
      expect(
        this,
        "4291      cmp     r1, r2\nd100      bne     l6\ne000      b       l8\n1840  l6: adds    r0, r0, r1\n4718  l8: bx      r3\n"
      );
      expect(
        this,
        "          @stackmark base\nb403      push    {r0, r1}\n          @stackmark locals\n9801      ldr     r0, [sp, locals@1]\nb401      push    {r0}\n9802      ldr     r0, [sp, locals@1]\nbc01      pop     {r0}\n          @stackempty locals\n9901      ldr     r1, [sp, locals@1]\n9102      str     r1, [sp, base@0]\n          @stackempty locals\nb002      add     sp, #8\n          @stackempty base\n"
      );
      expect(
        this,
        "b090      sub sp, #4*16\nb010      add sp, #4*16\n"
      );
      expect(
        this,
        '6261      .string "abc"\n0063      \n'
      );
      expect(
        this,
        '6261      .string "abcde"\n6463      \n0065      \n'
      );
      expect(
        this,
        "3042      adds r0, 0x42\n1c0d      adds r5, r1, #0\nd100      bne #0\n2800      cmp r0, #0\n6b28      ldr r0, [r5, #48]\n0200      lsls r0, r0, #8\n2063      movs r0, 0x63\n4240      negs r0, r0\n46c0      nop\nb500      push {lr}\nb401      push {r0}\nb402      push {r1}\nb404      push {r2}\nb408      push {r3}\nb520      push {r5, lr}\nbd00      pop {pc}\nbc01      pop {r0}\nbc02      pop {r1}\nbc04      pop {r2}\nbc08      pop {r3}\nbd20      pop {r5, pc}\n9003      str r0, [sp, #4*3]\n"
      );
    }
  };
  function preservesReg(ln, n) {
    if (ln.getOpExt() == "movs $r5, $i0" && ln.numArgs[0] != n)
      return true;
    return false;
  }
  function clobbersReg(ln, n) {
    if (ln.getOp() == "pop" && ln.numArgs[0] & 1 << n)
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

  // src/compiler.ts
  var tf = __toESM(__require("@tensorflow/tfjs"));

  // src/library.ts
  var asmDeps = {
    "softmax": ["expf_asm"]
  };
  var asmFns = {
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

  // src/ir.ts
  var unrollLimit = 10;
  function assert2(cond, msg = "assertion failed") {
    if (!cond) {
      debugger;
      throw new Error("ir: " + msg);
    }
  }
  function addParamBytes(mi, bytes) {
    assert2((mi.weightPtr & bytes.length - 1) == 0);
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
    assert2(v != null && !isNaN(v));
    mi.weightAsm += `.float ${v}
`;
    const u = float32ToUInt32(v);
    addParamBytes(mi, [
      u >> 0 & 255,
      u >> 8 & 255,
      u >> 16 & 255,
      u >> 24 & 255
    ]);
  }
  function addFloat16(mi, v) {
    assert2(v != null && !isNaN(v));
    mi.weightAsm += `.float16 ${v}
`;
    const u = float16toUInt16(v);
    addParamBytes(mi, [
      u >> 0 & 255,
      u >> 8 & 255
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
    assert2((mi.weightPtr & 3) == 0);
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
    const addConst = (k) => k < 1 << 12 ? 1 : 2;
    for (const op of ops) {
      switch (op.opcode) {
        case 0 /* comment */:
        case 1 /* label */:
          break;
        case 2 /* repeat */:
          cycles += (numCycles(op.body) + 4 + (op.isDef ? 1 : 0)) * op.num + 1;
          break;
        case 3 /* loadWeightAddr */:
          cycles += 2 + addConst(op.num * 4);
          break;
        case 4 /* loadDataAddr */:
          cycles += addConst(op.num * 4 + 8);
          break;
        case 5 /* addPtr */:
          if (op.src == null)
            cycles += addConst(op.num * 4);
          else {
            if (op.num != 1) {
              if (op.src > 500 /* Zero */) {
                if (op.src == 500 /* Zero */ + 1) {
                } else if (op.src == 500 /* Zero */ + 2) {
                  cycles++;
                } else {
                  cycles += 2;
                }
              } else {
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
        case 6 /* loadFConst */:
          if (op.num == 0)
            cycles += 2;
          else if (op.num == 1)
            cycles += 1;
          else
            cycles += 6;
          break;
        case 7 /* load */:
          cycles += 1 + op.num;
          break;
        case 8 /* store */:
          cycles += 1 + op.num;
          break;
        case 13 /* relu */:
          cycles += 6;
          break;
        case 10 /* vmax */:
          cycles += 4;
          if (op.src != op.dst)
            cycles++;
          break;
        case 9 /* vmul */:
        case 11 /* vadd */:
          if (op.src === prevDst || op.srcAlt === prevDst)
            cycles += 2;
          else
            cycles += 1;
          prevDst = op.dst;
          break;
        case 12 /* vcvt */:
          cycles += 1;
          break;
        case 14 /* fcall */:
          if (op.fname == "softmax")
            cycles += 200 + op.num * 150;
          else
            cycles += 500 + op.num * 500;
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
      `1 // output type - float32`
    ];
    for (let i = 0; i < 4; ++i)
      header.push(`0 // padding`);
    addShape(modelInfo.inputShape, "input");
    addShape(modelInfo.outputShape, "output");
    let initCmt = "";
    while (((_a = ops[0]) == null ? void 0 : _a.opcode) == 0 /* comment */) {
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
    regAlloc[200 /* InputPtr */] = 1;
    regAlloc[201 /* OutputPtr */] = 2;
    regAlloc[202 /* KernelPtr */] = 3;
    regAlloc[203 /* DataDescPtr */] = 7;
    write(`_start_model:`);
    write(`push {r4,r5,r6,r7,r8,r9,r10,r11,r12,lr}`);
    write(`mov ${reg(203 /* DataDescPtr */)}, r1`);
    write(`ldr r1, [r0, #4*4] // weight offset`);
    write(`adds r1, r0 // weight addr`);
    write(`str r1, [${reg(203 /* DataDescPtr */)}, #${weightAddrDO}]`);
    write(`movs r1, #0`);
    write(`str r1, [${reg(203 /* DataDescPtr */)}, #${zeroDO}]`);
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
    write(`_weights:
${modelInfo.weightAsm}`);
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
      assert2(!regAlloc[r]);
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
        oops2("can't alloc " + r);
      regAlloc[r] = all;
      if (f) {
        const pind = ind;
        try {
          ind += "    ";
          f();
        } finally {
          ind = pind;
          regAlloc = copy;
        }
      }
    }
    function write(asm) {
      if (isFake(asm))
        oops2("wrong reg: " + asm);
      resText += ind + asm + "\n";
    }
    function oops2(msg) {
      debugger;
      throw new Error("internal thumb error: " + msg);
    }
    function reg(r) {
      if (r == null)
        return "<fake>";
      if (r <= 32 /* S31 */)
        return "s" + (r - 0 /* S0 */);
      if (r >= 500 /* Zero */)
        return "#" + (r - 500 /* Zero */);
      const id = regAlloc[r];
      if (id == void 0)
        return "<fake:" + regName(r) + ">";
      return "r" + id;
    }
    function isFake(r) {
      return r.indexOf("<fake") >= 0;
    }
    function isLowReg(reg2) {
      return /^r[0-7]$/.test(reg2);
    }
    function loadConst(dst, num) {
      if (num <= 255 && isLowReg(dst))
        write(`movs ${dst}, #${num}`);
      else if (num <= 65535)
        write(`movw ${dst}, #${num}`);
      else {
        const lbl = `${lblid++}`;
        write(`ldr ${dst}, .c.${lbl}`);
        write(`b .s.${lbl}`);
        write(`.balign 4`);
        write(`.c.${lbl}: .word ${num}`);
        write(`.s.${lbl}:`);
      }
    }
    function addConst(dst, src, num) {
      if (Math.abs(num) < 1 << 12) {
        if (num == 0)
          write(`mov ${dst}, ${src}`);
        else if (num < 0)
          write(`subw ${dst}, ${src}, #${-num}`);
        else
          write(`addw ${dst}, ${src}, #${num}`);
      } else {
        if (src == dst) {
          const tmp = dst == "r0" ? "r1" : "r0";
          write(`push {${tmp}}`);
          loadConst(tmp, num);
          write(`adds ${dst}, ${dst}, ${tmp}`);
          write(`pop {${tmp}}`);
        } else {
          loadConst(dst, num);
          write(`adds ${dst}, ${src}, ${dst}`);
        }
      }
    }
    function compiles(ops2) {
      for (const op of ops2)
        compile(op);
    }
    function range2(op) {
      return "{" + range(op.num).map((k) => reg(op.dst + k)).join(",") + "}";
    }
    function compile(op) {
      let dst = reg(op.dst);
      const src = reg(op.src);
      const srcAlt = reg(op.srcAlt);
      const incr = op.increment ? "!" : "";
      switch (op.opcode) {
        case 1 /* label */:
          write(`${op.fname}:`);
          break;
        case 0 /* comment */:
          write(stringifyComment(op.fname));
          break;
        case 2 /* repeat */:
          assert2(op.num >= 1);
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
            } else {
              if (isLowReg(dst))
                write(`subs ${dst}, #1`);
              else
                write(`subs ${dst}, ${dst}, #1`);
              write(`bne ${lbl}`);
            }
          });
          break;
        case 3 /* loadWeightAddr */:
          write(`ldr r0, [${reg(203 /* DataDescPtr */)}, #${weightAddrDO}]`);
          addConst(dst, "r0", op.num * 4);
          break;
        case 4 /* loadDataAddr */:
          addConst(dst, reg(203 /* DataDescPtr */), byteOffset(op.num));
          break;
        case 5 /* addPtr */:
          if (isFake(dst) && op.isDef) {
            alloc(op.dst);
            dst = reg(op.dst);
          }
          if (op.src == null) {
            addConst(dst, srcAlt, op.num * 4);
          } else {
            if (op.num != 1) {
              loadConst("r0", op.num * 4);
              if (src[0] == "#") {
                const n = +src.slice(1);
                if (n == 0)
                  loadConst("r0", 0);
                else if (n == 1) {
                } else if (n == 2) {
                  write(`adds r0,r0`);
                } else {
                  assert2(dst != srcAlt);
                  loadConst(dst, n);
                  write(`muls r0, ${dst}`);
                }
              } else {
                write(`muls r0, ${src}`);
              }
            } else {
              if (src[0] == "#") {
                const n = +src.slice(1);
                loadConst("r0", n << 2);
              } else {
                write(`lsls r0, ${src}, #2`);
              }
            }
            write(`adds ${dst}, ${srcAlt}, r0`);
          }
          break;
        case 6 /* loadFConst */:
          if (op.num == 0)
            write(`vldr ${dst}, [${reg(203 /* DataDescPtr */)}, #${zeroDO}]`);
          else if (op.num == Number.NEGATIVE_INFINITY) {
            write(`movw r0, #0xff80`);
            write(`lsls r0, r0, #16`);
            write(`vmov ${dst}, r0`);
          } else {
            const tmp = float32ToUInt32(op.num);
            loadConst("r0", tmp);
            write(`vmov ${dst}, r0`);
          }
          break;
        case 7 /* load */:
          assert2(op.f16Mode != 1 /* On */);
          write(`vldm ${src}${incr}, ${range2(op)}`);
          break;
        case 8 /* store */:
          write(`vstm ${src}${incr}, ${range2(op)}`);
          break;
        case 13 /* relu */:
          write(`ldr r0, [${dst}, #0]`);
          write(`cmp r0, #0`);
          write(`it lt`);
          write(`movwlt r0, #0`);
          write(`stm ${dst}!, {r0}`);
          break;
        case 9 /* vmul */:
          write(`vmul.f32 ${dst}, ${src}, ${srcAlt}`);
          break;
        case 11 /* vadd */:
          write(`vadd.f32 ${dst}, ${src}, ${srcAlt}`);
          break;
        case 12 /* vcvt */:
          write(`${op.fname} ${dst}, ${src}`);
          break;
        case 10 /* vmax */:
          assert2(dst != srcAlt);
          if (src != dst)
            write(`vmov ${dst}, ${src}`);
          write(`vcmp.f32 ${dst}, ${srcAlt}`);
          write(`vmrs APSR_nzcv, FPSCR`);
          write(`it mi`);
          write(`vmovmi.f32 ${dst}, ${srcAlt}`);
          break;
        case 14 /* fcall */:
          write(`mov r0, ${dst}`);
          loadConst("r1", op.num);
          write(`bl ${op.fname}`);
          usedFns[op.fname] = true;
          break;
        default:
          oops2("bad op " + op.opcode);
      }
    }
  }
  function toJS(modelInfo, op) {
    let r = "";
    if (op.opcode == 2 /* repeat */) {
      const dst = regName(op.dst);
      r = `for (let ${dst} = 0; ${dst} < ${op.num}; ${dst}++) {
${indent(toJSs(modelInfo, op.body))}}
`;
    } else {
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
      case 1 /* label */:
        return stringifyComment("label: " + op.fname) + "\n";
      case 0 /* comment */:
        if (isBreak(op))
          return "debugger\n";
        return stringifyComment(op.fname) + "\n";
      case 2 /* repeat */:
        return `for (let ${dst} = 0; ${dst} < ${op.num}; ${dst}++) {
${indent(stringify(op.body))}}
`;
      case 3 /* loadWeightAddr */:
        return `${dst} = weightOff + ${op.num}
`;
      case 4 /* loadDataAddr */:
        return `${dst} = dataOff + ${op.num}
`;
      case 5 /* addPtr */:
        if (op.src == null)
          return `${dst} = ${srcAlt} + ${op.num}
`;
        return `${dst} = ${srcAlt} + ${src}${op.num == 1 ? "" : " * " + op.num}
`;
      case 6 /* loadFConst */:
        return `${dst} = ${op.num}
`;
      case 7 /* load */: {
        let r = "";
        let dp = op.dst + 0;
        if (op.increment) {
          for (let i = 0; i < op.num; ++i)
            r += `${regName(dp++)} = ${op.fname || "mem"}[${src}++]
`;
        } else {
          for (let i = 0; i < op.num; ++i)
            r += `${regName(dp++)} = mem[${src} + ${i}]
`;
        }
        return r;
      }
      case 8 /* store */: {
        let r = "";
        let dp = op.dst + 0;
        if (op.increment) {
          for (let i = 0; i < op.num; ++i)
            r += `mem[${src}++] = ${regName(dp++)}
`;
        } else {
          for (let i = 0; i < op.num; ++i)
            r += `mem[${src} + ${i}] = ${regName(dp++)}
`;
        }
        return r;
      }
      case 13 /* relu */:
        return `if (mem[${dst}] < 0) mem[${dst}] = 0; ${dst}++
`;
      case 9 /* vmul */:
        return `${dst} = f32(${src} * ${srcAlt})
`;
      case 11 /* vadd */:
        return `${dst} = f32(${src} + ${srcAlt})
`;
      case 10 /* vmax */:
        return `${dst} = Math.max(${src}, ${srcAlt})
`;
      case 14 /* fcall */:
        return `${op.fname}(${dst}, ${op.num})
`;
      case 12 /* vcvt */:
        return `${dst} = rt.${op.fname.replace(/\./g, "_")}(${src})
`;
      default:
        throw new Error("bad op " + op.opcode);
    }
  }
  function regName(r) {
    if (r <= 32 /* S31 */)
      return "s" + (r - 0 /* S0 */);
    if (r >= 500 /* Zero */)
      return "" + (r - 500 /* Zero */);
    if (r >= 400 /* Tmp0 */)
      return "tmp" + (r - 400 /* Tmp0 */);
    if (r >= 300 /* Index0 */)
      return "idx" + (r - 300 /* Index0 */);
    switch (r) {
      case 200 /* InputPtr */:
        return "input";
      case 202 /* KernelPtr */:
        return "kernel";
      case 201 /* OutputPtr */:
        return "output";
      default:
        return "???" + r;
    }
  }
  function toJSs(modelInfo, op) {
    return op.map((o) => toJS(modelInfo, o)).join("");
  }
  var repIdx = 0;
  function repeatIdx(n, f) {
    const idx = 300 /* Index0 */ + repIdx++;
    return {
      opcode: 2 /* repeat */,
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
      opcode: 0 /* comment */,
      fname: str
    };
  }
  function label(name) {
    return {
      opcode: 1 /* label */,
      fname: name
    };
  }
  function loadWeightAddr(dst, idx) {
    assert2(idx >= 0);
    return {
      opcode: 3 /* loadWeightAddr */,
      dst,
      num: idx
    };
  }
  function relaxWeights() {
    const r = addPtr(202 /* KernelPtr */, null, 0);
    r.fname = "relax";
    return r;
  }
  function loadDataAddr(dst, idx) {
    assert2(idx >= 0);
    return {
      opcode: 4 /* loadDataAddr */,
      dst,
      num: idx
    };
  }
  function addPtr(dst, idx, mult = 1, base) {
    if (!base)
      base = dst;
    return {
      opcode: 5 /* addPtr */,
      dst,
      src: idx,
      srcAlt: base,
      num: mult
    };
  }
  function load0(dst) {
    return {
      opcode: 6 /* loadFConst */,
      dst,
      num: 0
    };
  }
  function loadLit(dst, num) {
    return {
      opcode: 6 /* loadFConst */,
      dst,
      num
    };
  }
  function loadMInf(dst) {
    return {
      opcode: 6 /* loadFConst */,
      dst,
      num: Number.NEGATIVE_INFINITY
    };
  }
  function load(dst, num, src, adv) {
    return {
      opcode: 7 /* load */,
      dst,
      src,
      num,
      increment: adv
    };
  }
  function load16(dst, num, src) {
    return {
      opcode: 7 /* load */,
      dst,
      src,
      num,
      increment: true,
      f16Mode: 1 /* On */
    };
  }
  function loadWeight(mi, dst, num) {
    const src = 202 /* KernelPtr */;
    if (mi.opts.float16weights)
      return load16(dst, num, src);
    else
      return load(dst, num, src, true);
  }
  function store(dst, src, num, adv) {
    return {
      opcode: 8 /* store */,
      src: dst,
      dst: src,
      num,
      increment: adv
    };
  }
  function relu(dst) {
    return {
      opcode: 13 /* relu */,
      dst,
      increment: true
    };
  }
  function vmul(dst, a, b) {
    return {
      opcode: 9 /* vmul */,
      dst,
      src: a,
      srcAlt: b
    };
  }
  function vmax(dst, a, b) {
    if (b == dst)
      [a, b] = [b, a];
    return {
      opcode: 10 /* vmax */,
      dst,
      src: a,
      srcAlt: b
    };
  }
  function vadd(dst, a, b) {
    return {
      opcode: 11 /* vadd */,
      dst,
      src: a,
      srcAlt: b
    };
  }
  function vcvt(fname, dst, src) {
    return {
      opcode: 12 /* vcvt */,
      dst,
      src,
      fname
    };
  }
  function fcall(name, dst, len) {
    return {
      opcode: 14 /* fcall */,
      fname: name,
      dst,
      num: len
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
          } else {
            add(b);
          }
        }
      } else {
        add(a);
      }
    }
    return res;
  }
  function isRelax(op) {
    return op.opcode == 5 /* addPtr */ && op.fname == "relax";
  }
  function isBreak(op) {
    return op.opcode == 0 /* comment */ && op.fname == "BREAK";
  }
  function isOddF16(ops) {
    let cnt = 0;
    for (const op of ops) {
      if (op.opcode == 7 /* load */ && op.f16Mode)
        cnt += op.num;
      if (isRelax(op))
        cnt = cnt + 1 & ~1;
    }
    return !!(cnt & 1);
  }
  function fixupAndMarkF16(ops) {
    function loop(ops2, odd = false) {
      let cnt = odd ? 1 : 0;
      const isOdd = () => !!(cnt & 1);
      const res = [];
      for (let op of ops2) {
        op = cloneOp(op);
        if (op.opcode == 2 /* repeat */) {
          if (op.num == 0)
            continue;
          const odd0 = isOdd();
          const body0 = op.body;
          const r = loop(body0, odd0);
          op.body = r.ops;
          if (r.odd != odd0) {
            if (op.isDef) {
              console.log(stringify([op]));
              assert2(false);
            }
            if (op.num == 1) {
              pushRange(res, r.ops);
              cnt++;
            } else {
              const leftover = op.num & 1;
              op.num >>= 1;
              const r1 = loop(body0, r.odd);
              assert2(r1.odd == odd0);
              op.body = r.ops.concat(r1.ops);
              res.push(op);
              if (leftover) {
                const r2 = loop(body0, odd0);
                pushRange(res, r2.ops);
                cnt++;
              }
            }
          } else {
            res.push(op);
          }
          continue;
        }
        res.push(op);
        if (op.opcode == 7 /* load */ && op.f16Mode) {
          assert2(op.f16Mode == 1 /* On */);
          op.f16Mode = isOdd() ? 3 /* Odd */ : 2 /* Even */;
          cnt += op.num;
        }
        if (isRelax(op))
          cnt = cnt + 1 & ~1;
      }
      return { ops: res, odd: !!(cnt & 1) };
    }
    function expand(ops2) {
      const res = [];
      for (let op of ops2) {
        if (op.opcode == 2 /* repeat */) {
          assert2(!isOddF16(op.body));
          op.body = expand(op.body);
          res.push(op);
        } else if (op.opcode == 7 /* load */ && op.f16Mode) {
          let numLoad = 0;
          let isBottom = false;
          if (op.f16Mode == 3 /* Odd */) {
            numLoad = (op.num >> 1) + 1;
            res.push(addPtr(op.src, 501 /* One */, -1));
            if (!(op.num & 1))
              isBottom = true;
          } else if (op.f16Mode == 2 /* Even */) {
            numLoad = op.num + 1 >> 1;
            if (op.num & 1)
              isBottom = true;
          } else {
            assert2(false);
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
        } else {
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
      if (replMap[r] != void 0)
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
        case 2 /* repeat */:
          if (op.num == 0) {
          } else if (op.num == 1) {
            replMap[op.dst] = 500 /* Zero */;
            pushRange(res, optimize(op.body, replMap));
          } else {
            op.body = optimize(op.body, replMap);
            const stripLoop = op.num * op.body.length < unrollLimit * 2;
            const canUnroll = !op.isDef && 2 * op.body.length < unrollLimit;
            if (stripLoop) {
              for (let i = 0; i < op.num; ++i) {
                replMap[op.dst] = 500 /* Zero */ + i;
                pushRange(res, optimize(op.body, replMap));
              }
            } else if (canUnroll) {
              const unrollCnt = unrollLimit / op.body.length | 0;
              const tmp = op.body.slice();
              for (let i = 1; i < unrollCnt; ++i)
                pushRange(op.body, tmp);
              const newnum = op.num / unrollCnt | 0;
              res.push(op);
              const left = op.num - newnum * unrollCnt;
              op.num = newnum;
              for (let i = 0; i < left; ++i)
                pushRange(res, tmp);
            } else {
              res.push(op);
            }
          }
          break;
        case 5 /* addPtr */:
          if (op.dst == op.srcAlt && (op.num == 0 || op.src == 500 /* Zero */)) {
          } else
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

  // src/runtime.ts
  function mkRuntime(mem) {
    return {
      softmax: (ptr, len) => {
        let max = mem[ptr];
        for (let i = 1; i < len; ++i)
          max = Math.max(mem[ptr + i], max);
        let sum = 0;
        for (let i = 0; i < len; ++i)
          sum += mem[ptr + i] = Math.exp(mem[ptr + i] - max);
        for (let i = 0; i < len; ++i)
          mem[ptr + i] /= sum;
      },
      f32: (v) => {
        const arr = new Float32Array(1);
        arr[0] = v;
        return arr[0];
      },
      vcvtb_f32_f16: (v) => float16AsUintToFloat(v & 65535),
      vcvtt_f32_f16: (v) => float16AsUintToFloat(v >> 16 & 65535)
    };
  }

  // src/compiler.ts
  var inited2 = false;
  var compilers = {
    Conv2D: { compile: compileConv, computePaddedInputShape: paddingConv },
    Conv1D: { compile: compileConv, computePaddedInputShape: paddingConv },
    DepthwiseConv2D: { compile: compileDepthConv, computePaddedInputShape: paddingConv },
    DepthwiseConv1D: { compile: compileDepthConv, computePaddedInputShape: paddingConv },
    MaxPooling1D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool, needsMInfPadding: true },
    MaxPooling2D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool, needsMInfPadding: true },
    AveragePooling1D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool },
    AveragePooling2D: { compile: compileMaxPooling, computePaddedInputShape: paddingPool },
    Dense: { compile: compileDense },
    Activation: { compile: compileActivation, inPlace: true },
    Softmax: { compile: compileSoftmax, inPlace: true },
    BatchNormalization: { compile: compileBatchNorm, inPlace: true },
    Dropout: {},
    Flatten: {},
    InputLayer: {},
    Reshape: {}
  };
  var numFPRegs = 32;
  var numTmpRegs = 6;
  function unsupported(msg) {
    debugger;
    throw new Error("Unsupported operator or config: " + msg);
  }
  function assert3(cond, msg = "assertion failed") {
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
    } else {
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
      return;
    res.push(loadDataAddr(201 /* OutputPtr */, info.outputOff));
    if (config.activation == "relu")
      res.push(repeat(numoutp, () => [relu(201 /* OutputPtr */)]));
    else if (config.activation == "softmax")
      res.push(fcall("softmax", 201 /* OutputPtr */, numoutp));
    else
      unsupported("activation: " + config.activation);
  }
  function addSoftmax(res, info) {
    const numoutp = shapeElts(info.outputShape);
    res.push(loadDataAddr(201 /* OutputPtr */, info.outputOff));
    res.push(fcall("softmax", 201 /* OutputPtr */, numoutp));
  }
  function paddingConv(info) {
    const config = info.layer.getConfig();
    const res = info.inputShape.slice();
    for (let i = 1; i <= config.kernelSize.length; ++i) {
      const str = config.strides[i - 1];
      const tmp = info.outputShape[i] * str + config.kernelSize[i - 1] - str;
      assert3(tmp + str - 1 >= res[i], `${tmp} >= ${res[i]}`);
      if (tmp > res[i])
        res[i] = tmp;
    }
    return res;
  }
  function paddingPool(info) {
    const config = info.layer.getConfig();
    const res = info.inputShape.slice();
    for (let i = 1; i <= config.poolSize.length; ++i) {
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
    const weights = is2D ? weights0 : [weights0];
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
    assert3(kh <= inph, "KH2");
    assert3(kw <= inpw, "KW2");
    assert3(weights.length == kh, "KH");
    assert3(weights[0].length == kw, "KW");
    assert3(weights[0][0].length == inpch, "CH");
    assert3(weights[0][0][0].length == config.filters, "F");
    assert3(outch == config.filters, "FF");
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
      loadWeightAddr(202 /* KernelPtr */, weightsIdx),
      repeatIdx(config.filters, (filt) => {
        const res2 = [];
        const setOutput = (res3) => {
          res3.push(loadDataAddr(201 /* OutputPtr */, info.outputOff));
          res3.push(addPtr(201 /* OutputPtr */, filt));
        };
        setOutput(res2);
        if (config.useBias)
          res2.push(load(0 /* S0 */, 1, 202 /* KernelPtr */, true));
        else
          res2.push(load0(0 /* S0 */));
        res2.push(
          repeat(outw * outh, () => [
            store(201 /* OutputPtr */, 0 /* S0 */, 1, false),
            addPtr(201 /* OutputPtr */, null, config.filters)
          ])
        );
        res2.push(repeatIdx(kh, (kline) => {
          const res3 = [];
          const kernSz = kw * inpch;
          let chunk = 0;
          for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
            chunk = kernSz - kernOff;
            if (chunk > flashRegs)
              chunk = flashRegs;
            res3.push(loadWeight(mi, memRegs, chunk));
            res3.push(loadDataAddr(200 /* InputPtr */, info.inputOff + kernOff));
            res3.push(addPtr(200 /* InputPtr */, kline, inpw * inpch));
            setOutput(res3);
            const wSkip = strw * inpch;
            const hSkip = strh * inpw * inpch;
            res3.push(repeat(outh, () => [
              repeat(outw, () => flatten(
                load(0 /* S0 */, chunk, 200 /* InputPtr */, true),
                addPtr(200 /* InputPtr */, null, wSkip - chunk),
                range(chunk + 1).map((i) => [
                  i < chunk ? vmul(i, i, i + memRegs) : null,
                  i >= 2 ? vadd(0 /* S0 */, 0 /* S0 */, i - 1) : null
                ]),
                load(1 /* S1 */, 1, 201 /* OutputPtr */, false),
                vadd(0 /* S0 */, 0 /* S0 */, 1 /* S1 */),
                store(201 /* OutputPtr */, 0 /* S0 */, 1, false),
                addPtr(201 /* OutputPtr */, null, config.filters)
              )),
              addPtr(200 /* InputPtr */, null, hSkip - outw * wSkip)
            ]));
          }
          res3.push(relaxWeights());
          return res3;
        }));
        res2.push(relaxWeights());
        return res2;
      })
    ];
    addActivation(res, info);
    return res;
  }
  function compileDepthConv(info) {
    const config = info.layer.getConfig();
    const flashRegOff = 2;
    const flashRegs = numFPRegs - flashRegOff;
    validateConfig(info);
    const is2D = config.kernelSize.length == 2;
    const weights0 = info.layer.weights[0].read().arraySync();
    const weights = is2D ? weights0 : [weights0];
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
    assert3(kh <= inph, "KH2");
    assert3(kw <= inpw, "KW2");
    assert3(weights.length == kh, "KH");
    assert3(weights[0].length == kw, "KW");
    assert3(weights[0][0].length == inpch, "CH");
    assert3(weights[0][0][0].length == config.depthMultiplier, "F");
    assert3(outch == config.depthMultiplier * inpch, "FF");
    const mi = info.model;
    const weightsIdx = weightOffset(mi);
    const bias = config.useBias ? info.layer.weights[1].read().arraySync() : null;
    if (bias)
      unsupported("bias in depthwise");
    for (let q = 0; q < config.depthMultiplier; q++) {
      if (bias)
        addBias(mi, bias[q]);
      for (let k = 0; k < inpch; ++k) {
        for (let y = 0; y < kh; y++) {
          for (let x = 0; x < kw; x++)
            addWeight(mi, weights[y][x][k][q]);
        }
        alignWeights(mi);
      }
    }
    const res = [
      loadWeightAddr(202 /* KernelPtr */, weightsIdx),
      repeatIdx(config.depthMultiplier, (q) => [repeatIdx(inpch, (k) => {
        const res2 = [];
        const setOutput = (res3) => {
          res3.push(loadDataAddr(201 /* OutputPtr */, info.outputOff));
          res3.push(addPtr(201 /* OutputPtr */, k, config.depthMultiplier));
          res3.push(addPtr(201 /* OutputPtr */, q));
        };
        setOutput(res2);
        if (config.useBias)
          res2.push(load(0 /* S0 */, 1, 202 /* KernelPtr */, true));
        else
          res2.push(load0(0 /* S0 */));
        res2.push(
          repeat(outw * outh, () => [
            store(201 /* OutputPtr */, 0 /* S0 */, 1, false),
            addPtr(201 /* OutputPtr */, null, outch)
          ])
        );
        const kernSz = kh * kw;
        let skipAcc = 0;
        const skipAfter = (kernOff) => {
          const r = (kernOff % kw == kw - 1 ? inpw - kw + 1 : 1) * inpch;
          skipAcc += r;
          return r;
        };
        let chunk = 0;
        for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
          chunk = kernSz - kernOff;
          if (chunk > flashRegs)
            chunk = flashRegs;
          res2.push(loadWeight(mi, flashRegOff, chunk));
          let skip = 0;
          for (let i = 0; i < kernOff; ++i)
            skip += skipAfter(i);
          res2.push(
            loadDataAddr(200 /* InputPtr */, info.inputOff + skip),
            addPtr(200 /* InputPtr */, k)
          );
          setOutput(res2);
          const wSkip = strw * inpch;
          const hSkip = strh * inpw * inpch;
          res2.push(repeat(outh, () => [
            repeat(outw, () => {
              skipAcc = 0;
              const tmp = flatten(
                load0(1 /* S1 */),
                range(chunk).map((i) => [
                  load(0 /* S0 */, 1, 200 /* InputPtr */, false),
                  addPtr(200 /* InputPtr */, null, skipAfter(kernOff + i)),
                  vmul(0 /* S0 */, 0 /* S0 */, i + flashRegOff),
                  vadd(1 /* S1 */, 1 /* S1 */, 0 /* S0 */)
                ]),
                load(0 /* S0 */, 1, 201 /* OutputPtr */, false),
                vadd(0 /* S0 */, 0 /* S0 */, 1 /* S1 */),
                store(201 /* OutputPtr */, 0 /* S0 */, 1, false),
                addPtr(201 /* OutputPtr */, null, outch)
              );
              tmp.push(addPtr(200 /* InputPtr */, null, wSkip - skipAcc));
              return tmp;
            }),
            addPtr(200 /* InputPtr */, null, hSkip - outw * wSkip)
          ]));
        }
        res2.push(relaxWeights());
        return res2;
      })])
    ];
    addActivation(res, info);
    return res;
  }
  function compileMaxPooling(info) {
    const config = info.layer.getConfig();
    const is2D = config.poolSize.length == 2;
    const isAvg = info.layer.getClassName().startsWith("Average");
    validateConfig(info);
    if (isAvg && config.padding != "valid")
      unsupported("only 'valid' padding supported for AvgPool");
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
    assert3(kh <= inph, "KH2");
    assert3(kw <= inpw, "KW2");
    assert3(numch == outch, "CH");
    const singleInputPtr = kh - 1 > numTmpRegs;
    const lineW = inpw * numch;
    return [
      repeatIdx(numch, (filt) => {
        const res = [
          loadDataAddr(201 /* OutputPtr */, info.outputOff),
          addPtr(201 /* OutputPtr */, filt),
          loadDataAddr(200 /* InputPtr */, info.inputOff),
          addPtr(200 /* InputPtr */, filt)
        ];
        const ptrRegs = singleInputPtr ? [] : range(kh - 1).map((i) => 400 /* Tmp0 */ + i);
        ptrRegs.unshift(200 /* InputPtr */);
        if (!singleInputPtr)
          for (let i = 1; i < kh; ++i) {
            const op = addPtr(ptrRegs[i], null, lineW * i, 200 /* InputPtr */);
            op.isDef = true;
            res.push(op);
          }
        res.push(
          repeat(outh, () => flatten(
            repeat(outw, () => {
              const res2 = [];
              for (let i = 0; i < kh; ++i) {
                let preg = ptrRegs[i];
                if (singleInputPtr) {
                  preg = 400 /* Tmp0 */;
                  const op = addPtr(preg, null, lineW * i, 200 /* InputPtr */);
                  if (i == 0)
                    op.isDef = true;
                  res2.push(op);
                }
                for (let j = 0; j < kw; ++j) {
                  const reg = i == 0 && j == 0 ? 0 /* S0 */ : 1 /* S1 */;
                  res2.push(
                    load(reg, 1, preg, true),
                    addPtr(preg, null, numch - 1)
                  );
                  if (reg != 0 /* S0 */) {
                    if (isAvg)
                      res2.push(vadd(0 /* S0 */, 0 /* S0 */, reg));
                    else
                      res2.push(vmax(0 /* S0 */, 0 /* S0 */, reg));
                  }
                }
                if (!singleInputPtr)
                  res2.push(
                    addPtr(preg, null, (strw - kw) * numch)
                  );
              }
              if (isAvg)
                res2.push(
                  loadLit(1 /* S1 */, 1 / (kw * kh)),
                  vmul(0 /* S0 */, 0 /* S0 */, 1 /* S1 */)
                );
              res2.push(
                store(201 /* OutputPtr */, 0 /* S0 */, 1, true),
                addPtr(201 /* OutputPtr */, null, numch - 1)
              );
              if (singleInputPtr)
                res2.push(
                  addPtr(200 /* InputPtr */, null, strw * numch)
                );
              return res2;
            }),
            ptrRegs.map((r) => addPtr(r, null, strh * lineW - outw * strw * numch))
          ))
        );
        return res;
      })
    ];
  }
  function compileBatchNorm(info) {
    const config = info.layer.getConfig();
    const flashRegs = numFPRegs - 2;
    const flashReg0 = 0 /* S0 */ + 2;
    if (info.inputShape.length != 4)
      unsupported("inputShape: " + info.inputShape.length);
    if (config.dtype && config.dtype != "float32")
      unsupported("dtype: " + config.dtype);
    const [_null, outh, outw, numch] = info.inputShape;
    function readVar(name) {
      const r = info.layer.weights.find((w) => w.originalName.endsWith("/" + name)).read().arraySync();
      assert3(r.length == numch);
      return r;
    }
    const gamma = readVar("gamma");
    const beta = readVar("beta");
    const movingMean = readVar("moving_mean");
    const movingVar = readVar("moving_variance");
    const mi = info.model;
    const weightsIdx = weightOffset(mi);
    for (let i = 0; i < numch; i++) {
      const q = 1 / Math.sqrt(movingVar[i] + config.epsilon);
      const mult = q * gamma[i];
      const offset = -q * gamma[i] * movingMean[i] + beta[i];
      addWeight(mi, mult);
      addWeight(mi, offset);
    }
    assert3(info.inputOff == info.outputOff);
    const res = [
      loadWeightAddr(202 /* KernelPtr */, weightsIdx)
    ];
    const kernSz = numch * 2;
    let chunk = 0;
    for (let kernOff = 0; kernOff < kernSz; kernOff += chunk) {
      assert3((kernOff & 1) == 0);
      chunk = kernSz - kernOff;
      if (chunk > flashRegs)
        chunk = flashRegs;
      res.push(
        loadWeight(mi, flashReg0, chunk),
        loadDataAddr(201 /* OutputPtr */, info.outputOff + (kernOff >> 1)),
        repeat(outh * outw, () => flatten(
          range(chunk >> 1).map((i) => [
            load(0 /* S0 */, 1, 201 /* OutputPtr */, false),
            vmul(0 /* S0 */, 0 /* S0 */, i * 2 + flashReg0),
            vadd(0 /* S0 */, 0 /* S0 */, i * 2 + 1 + flashReg0),
            store(201 /* OutputPtr */, 0 /* S0 */, 1, true)
          ]),
          addPtr(201 /* OutputPtr */, null, numch - (chunk >> 1))
        ))
      );
    }
    return res;
  }
  function compileDense(info) {
    const config = info.layer.getConfig();
    const maxChunk = (numFPRegs >> 1) - 2;
    const memReg0 = 1 /* S1 */;
    const flashReg0 = memReg0 + maxChunk;
    if (info.inputShape.length != 2)
      unsupported("inputShape: " + info.inputShape.length);
    if (config.dtype && config.dtype != "float32")
      unsupported("dtype: " + config.dtype);
    const weights = info.layer.weights[0].read().arraySync();
    const inpsize = info.inputShape[1];
    assert3(weights.length == inpsize, "IH");
    assert3(weights[0].length == config.units, "UN");
    const mi = info.model;
    const weightsIdx = weightOffset(mi);
    const bias = config.useBias ? info.layer.weights[1].read().arraySync() : null;
    for (let f = 0; f < config.units; f++) {
      if (bias)
        addBias(mi, bias[f]);
      for (let i = 0; i < inpsize; ++i)
        addWeight(mi, weights[i][f]);
      alignWeights(mi);
    }
    const res = [
      loadWeightAddr(202 /* KernelPtr */, weightsIdx),
      loadDataAddr(201 /* OutputPtr */, info.outputOff),
      repeat(config.units, () => {
        const res2 = [];
        if (config.useBias)
          res2.push(load(0 /* S0 */, 1, 202 /* KernelPtr */, true));
        else
          res2.push(load0(0 /* S0 */));
        res2.push(loadDataAddr(200 /* InputPtr */, info.inputOff));
        const addChunk = (len) => flatten(
          load(memReg0, len, 200 /* InputPtr */, true),
          loadWeight(mi, flashReg0, len),
          range(len + 1).map((i) => [
            i < len ? vmul(memReg0 + i, memReg0 + i, flashReg0 + i) : null,
            i >= 1 ? vadd(0 /* S0 */, 0 /* S0 */, memReg0 + i - 1) : null
          ])
        );
        const numRep = inpsize / maxChunk | 0;
        if (numRep > 0)
          res2.push(repeat(numRep, () => addChunk(maxChunk)));
        const left = inpsize - numRep * maxChunk;
        if (left > 0)
          pushRange(res2, addChunk(left));
        res2.push(store(201 /* OutputPtr */, 0 /* S0 */, 1, true));
        res2.push(relaxWeights());
        return res2;
      })
    ];
    addActivation(res, info);
    return res;
  }
  function compileActivation(info) {
    const res = [];
    addActivation(res, info);
    return res;
  }
  function compileSoftmax(info) {
    const res = [];
    addSoftmax(res, info);
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
    if (info.testable === void 0)
      info.testable = !!info.compile;
    if (!info.compile) {
      if (info.inPlace === void 0)
        info.inPlace = true;
      info.compile = noop;
    }
    if (!info.computePaddedInputShape)
      info.computePaddedInputShape = (info2) => info2.inputShape.slice();
  }
  function isInPlace(layer) {
    var _a;
    return !!((_a = compilers[layer.getClassName()]) == null ? void 0 : _a.inPlace);
  }
  function isTestable(layer) {
    var _a;
    return !!((_a = compilers[layer.getClassName()]) == null ? void 0 : _a.testable);
  }
  function needMInfPadding(layer) {
    var _a;
    return !!((_a = compilers[layer.getClassName()]) == null ? void 0 : _a.needsMInfPadding);
  }
  function shapeToString(shape) {
    return `[${shape.filter((x) => x != null).join(",")}]`;
  }
  function assignLayerInfos(m, opts) {
    if (!inited2) {
      inited2 = true;
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
      } else {
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
      } else {
        info.rawInputOff = null;
      }
      const elts = shapeElts(info.outputShape);
      if (isInPlace(l)) {
        recordMax(shapeElts(info.inputShape));
        recordMax(shapeElts(info.outputShape));
      } else {
        recordMax(shapeElts(info.inputShape) + shapeElts(info.outputShape));
        currIdx = currIdx == 0 ? 1 : 0;
      }
      info.outputOff = currIdx;
      if (elts > maxSize[currIdx])
        maxSize[currIdx] = elts;
      prev = info;
    }
    modelInfo.outputShape = prev.outputShape;
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
      console.log("possible arena shrink with wiser allocation: " + (arenaSize / totalMax).toFixed(3) + "x");
    }
    return modelInfo;
  }
  function compilePadding(info) {
    const res = [
      comment("padding")
    ];
    if (info.rawInputOff == null)
      return res;
    const is2D = info.rawInputShape.length >= 4;
    const fix1D = (a) => {
      a = a.slice();
      a.shift();
      if (!is2D)
        a.unshift(1);
      return a;
    };
    const [inpy, inpx, numch] = fix1D(info.rawInputShape);
    const [outy, outx, outch] = fix1D(info.inputShape);
    assert3(numch == outch);
    const padx = outx - inpx;
    const x0 = padx >> 1;
    const x1 = padx - x0;
    const pady = outy - inpy;
    const y0 = pady >> 1;
    const y1 = pady - y0;
    const numZero = numFPRegs >> 1;
    const numData = numFPRegs - numZero;
    const dataReg = 0 /* S0 */ + numZero;
    res.push(needMInfPadding(info.layer) ? loadMInf(0 /* S0 */) : load0(0 /* S0 */));
    for (let i = 1; i < numZero; ++i)
      res.push(vadd(0 /* S0 */ + i, 0 /* S0 */, 0 /* S0 */));
    res.push(loadDataAddr(200 /* InputPtr */, info.rawInputOff));
    res.push(loadDataAddr(201 /* OutputPtr */, info.inputOff));
    const topPad = y0 * outx + x0;
    const linePad = x1 + x0;
    const bottomPad = x1 + y1 * outx;
    res.push(...setZero(topPad));
    res.push(repeat(inpy - 1, () => flatten(
      copyOver(inpx),
      setZero(linePad)
    )));
    res.push(...copyOver(inpx));
    res.push(...setZero(bottomPad));
    return res;
    function setZero(n) {
      const res2 = [];
      n *= numch;
      const leftover = n % numZero;
      const reps = (n - leftover) / numZero;
      if (reps)
        res2.push(repeat(reps, () => [
          store(201 /* OutputPtr */, 0 /* S0 */, numZero, true)
        ]));
      if (leftover)
        res2.push(store(201 /* OutputPtr */, 0 /* S0 */, leftover, true));
      return res2;
    }
    function copyOver(n) {
      const res2 = [];
      n *= numch;
      const leftover = n % numData;
      const reps = (n - leftover) / numData;
      if (reps)
        res2.push(repeat(reps, () => [
          load(dataReg, numData, 200 /* InputPtr */, true),
          store(201 /* OutputPtr */, dataReg, numData, true)
        ]));
      if (leftover) {
        res2.push(
          load(dataReg, leftover, 200 /* InputPtr */, true),
          store(201 /* OutputPtr */, dataReg, leftover, true)
        );
      }
      return res2;
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
    return shape.filter((x) => x != null);
  }
  function compileModelCore(m, opts) {
    const modelInfo = assignLayerInfos(m, opts);
    if (opts.optimize === void 0)
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
        info.stats.arenaBytes = shapeElts(info.rawInputShape) + shapeElts(info.inputShape) << 2;
        info.stats.hasPadding = true;
      }
      const cinfo = compilers[l.getClassName()];
      if (cinfo) {
        const size0 = weightOffset(modelInfo);
        const tmp = optimizeWithComment(opts, cinfo.compile(info), info.stats);
        info.stats.weightBytes = weightOffset(modelInfo) - size0 << 2;
        const shapeinfo = `data: ${shapeToString(info.inputShape)}@${info.inputOff} => ${shapeToString(info.outputShape)}@${info.outputOff}`;
        const infostr = `Layer: ${l.getClassName()}; ${shapeinfo}`;
        tmp.opcodes.unshift(comment(infostr));
        if (opts.verbose)
          console.log(infostr + " " + tmp.optinfo);
        ops.push(tmp.opcodes);
      } else {
        console.log(l.getConfig());
        unsupported("layer: " + l.getClassName());
      }
      if (info.stats.unoptimizedCycles)
        info.stats.arenaBytes = Math.max(info.stats.arenaBytes, shapeElts(info.inputShape) + shapeElts(info.outputShape) << 2);
      totalStats.unoptimizedCycles += info.stats.unoptimizedCycles;
      ops.push([label("end_" + statsIdx)]);
    }
    let flat = flatten(ops);
    const lastInfo = getLayerInfo(m.layers[m.layers.length - 1]);
    modelInfo.outputOffset = lastInfo.outputOff;
    const mhz = 64;
    const cycles = numCycles(flat);
    const cycleinfo = `total cycles: ${cycles} (${(cycles / (mhz * 1e3)).toFixed(3)}ms at ${mhz}MHz)`;
    modelInfo.stats = cycleinfo;
    totalStats.optimizedCycles = cycles;
    if (opts.verbose)
      console.log(modelInfo.stats);
    modelInfo.weightBuffer = modelInfo.weightBuffer.slice(0, modelInfo.weightPtr);
    const inputSize = shapeElts(getLayerInfo(m.layers[0]).rawInputShape);
    let js = `
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
        if (inputs.length != ${inputSize})
            throw new Error("invalid input size; expected ${inputSize}, got " + inputs.length)
        mem.set(inputs, dataOff)
        let input, output, kernel
        let ${range(numTmpRegs).map((r) => "tmp" + r).join(", ")}
        let ${range(numFPRegs).map((r) => "s" + r).join(", ")}

${toJSs(modelInfo, flat)}
        
        return mem.slice(${lastInfo.outputOff}, ${lastInfo.outputOff + shapeElts(lastInfo.outputShape)})
    })
})
`;
    const execute = (0, eval)(js)(modelInfo.weightBuffer, mkRuntime);
    js = `${stringifyComment(modelInfo.stats)}
const modelFromWeights = ${js};
`;
    const w = Array.from(new Uint32Array(modelInfo.weightBuffer.buffer));
    js += `const weights = new Uint8Array(new Uint32Array(${JSON.stringify(w)}).buffer);
`;
    js += `const modelFromRuntime = mkR => modelFromWeights(weights, mkR);
`;
    js += `return { weights, modelFromRuntime, modelFromWeights, inputSize: ${inputSize} };
`;
    let thumb = "";
    if (opts.includeTest && opts.testOutput && opts.testOutputFromJS) {
      const prev = opts.testOutput;
      opts.testOutput = execute(opts.testInput);
      thumb = toThumb(modelInfo, flat);
      opts.testOutput = prev;
    } else {
      thumb = toThumb(modelInfo, flat);
    }
    const res = {
      execute,
      js,
      thumb,
      machineCode: null,
      options: opts,
      memInfo: null,
      timeInfo: modelInfo.stats,
      stats: {
        total: totalStats,
        layers: layerStats
      }
    };
    return res;
  }
  async function serializeModel(m) {
    let mod;
    await m.save({
      save: (m2) => {
        mod = m2;
        const res = {
          modelArtifactsInfo: {
            dateSaved: new Date(),
            modelTopologyType: "JSON"
          }
        };
        return Promise.resolve(res);
      }
    });
    return mod;
  }
  async function* partialModels(m, opts) {
    var _a;
    const mod = await serializeModel(m);
    delete mod.weightData;
    delete mod.weightSpecs;
    const cfg = (_a = mod.modelTopology) == null ? void 0 : _a.config;
    const layersJson = (cfg == null ? void 0 : cfg.layers) || [];
    for (let i = 0; i < m.layers.length; ++i) {
      const layerJson = layersJson[i];
      const layer = m.layers[i];
      const info = getLayerInfo(layer);
      if ((layerJson == null ? void 0 : layerJson.class_name) != layer.getClassName())
        throw new Error("invalid serialization");
      if (!isTestable(layer))
        continue;
      const lcfg = layerJson.config;
      lcfg.batch_input_shape = info.rawInputShape;
      cfg.layers = [layerJson];
      const copy = await tf.loadLayersModel({ load: () => Promise.resolve(mod) });
      console.log(`testing ${layer.getClassName()}: ${shapeToString(info.rawInputShape)} => ${shapeToString(info.outputShape)}...`);
      yield copy;
      layerJson.config.batch_input_shape = info.rawInputShape;
      if (lcfg.activation && lcfg.activation != "linear") {
        lcfg.activation = null;
        const withoutAct = await tf.loadLayersModel({ load: () => Promise.resolve(mod) });
        console.log(`also with no activation...`);
        yield withoutAct;
      }
    }
  }
  async function* prefixModels(m, opts) {
    var _a;
    const mod = await serializeModel(m);
    const cfg = (_a = mod.modelTopology) == null ? void 0 : _a.config;
    const layersJson = (cfg == null ? void 0 : cfg.layers) || [];
    for (let i = 0; i < m.layers.length; ++i) {
      const layerJson = layersJson[i];
      const layer = m.layers[i];
      const info = getLayerInfo(layer);
      if ((layerJson == null ? void 0 : layerJson.class_name) != layer.getClassName())
        throw new Error("invalid serialization");
      if (!isTestable(layer))
        continue;
      cfg.layers = layersJson.slice(0, i + 1);
      const copy = await tf.loadLayersModel({ load: () => Promise.resolve(mod) }, { strict: false });
      console.log(`testing prefix ${layer.getClassName()} => ${shapeToString(info.outputShape)}...`);
      yield copy;
    }
  }

  // src/driver.ts
  var epsF32 = 9e-5;
  var epsF16 = 0.01;
  function mkProcessorFile() {
    const b = new File(new ThumbProcessor());
    b.ei.testAssembler();
    b.disablePeepHole = true;
    b.lookupExternalLabel = (_name) => null;
    b.normalizeExternalLabel = (s) => s;
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
    const binary = new Uint8Array((procFile.buf.length << 1) + 15 & ~15);
    for (let i = 0; i < procFile.buf.length; ++i) {
      binary[i << 1] = procFile.buf[i] & 255;
      binary[(i << 1) + 1] = procFile.buf[i] >> 8 & 255;
    }
    return { binary, procFile };
  }
  function randomTensor(shape, mult = 1) {
    shape = shape.map((s) => s == null ? 1 : s);
    const num = shapeElts(shape);
    return tf2.tidy(() => tf2.tensor(range(num).map((_) => mult * randomSFloat())).reshape(shape));
  }
  function randomPosTensor(shape, mult = 1) {
    shape = shape.map((s) => s == null ? 1 : s);
    const num = shapeElts(shape);
    return tf2.tidy(() => tf2.tensor(range(num).map((_) => mult * randomUFloat())).reshape(shape));
  }
  function setRandomWeights(l) {
    let idx = 0;
    for (const w of l.weights) {
      const mult = 1;
      if (w.originalName.endsWith("/moving_variance"))
        w.write(randomPosTensor(w.shape, mult));
      else
        w.write(randomTensor(w.shape, mult));
      idx++;
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
      let save = function() {
        opts.testInput = randomInput.flatten().arraySync();
        opts.testOutput = res;
      };
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
      if (count++ > (opts.includeTest ? 1e3 : 100) || maxMul > 0.1) {
        if (!mul)
          save();
        break;
      }
    }
    return opts;
  }
  function compileModel(m, opts) {
    const cres = compileModelCore(m, opts);
    const ares = assemble(cres.thumb);
    cres.machineCode = ares.binary;
    let idx = 0;
    for (const st2 of cres.stats.layers) {
      st2.codeBytes = ares.procFile.lookupLabel("end_" + idx) - ares.procFile.lookupLabel("begin_" + idx);
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
    for await (const mod of partialModels(m, optsPart)) {
      for (const l of mod.layers)
        setRandomWeights(l);
      compileAndTest(mod, optsPart);
    }
    console.log("Validating prefix models...");
    for await (const mod of prefixModels(m, optsPart)) {
      compileAndTest(mod, optsPart);
    }
    console.log("Compiling full model...");
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
        console.log(`at ${i} ${res[i]}[exp] - ${res2[i]} = ${res[i] - res2[i]}`);
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
    } catch (e) {
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
    return (bin[off] | bin[off + 1] << 8 | bin[off + 2] << 16 | bin[off + 3] << 24) >>> 0;
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
    if (magic0 != 809963362)
      return null;
    const modelSize = testInpOff || totalSize;
    const codeSize = weightsOff - hdSize;
    const codePerc = codeSize * 100 / modelSize;
    const testSize = totalSize - modelSize;
    function sz(n) {
      return (n / 1024).toFixed(2) + "k";
    }
    const info = `model: ${sz(modelSize)}; code: ${sz(codeSize)} (${codePerc.toFixed(1)}%); arena: ${sz(arenaSize)}; test ${sz(testSize)}`;
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
    const cfg = (_b = (_a = modelJSON.modelTopology) == null ? void 0 : _a.model_config) == null ? void 0 : _b.config;
    const outLayers = [];
    let seq_id = 0;
    function addLayer(layer) {
      const layerConfig = layer == null ? void 0 : layer.config;
      if (layerConfig) {
        layerConfig.bias_regularizer = null;
        layerConfig.activity_regularizer = null;
        layerConfig.bias_constraint = null;
      }
      if (layer.class_name == "Sequential") {
        seq_id++;
        for (const l of layer.config.layers) {
          if (l.class_name == "InputLayer")
            continue;
          if (l.config.name == "dropout")
            l.config.name += "_seq_" + seq_id;
          addLayer(l);
        }
      } else {
        outLayers.push(layer);
      }
    }
    if (cfg == null ? void 0 : cfg.layers) {
      cfg.layers.forEach(addLayer);
      cfg.layers = outLayers;
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

  // pxt/extension.ts
  var import_tfjs = __require("@tensorflow/tfjs");
  function inIFrame() {
    try {
      return typeof window !== "undefined" && window.self !== window.top;
    } catch (e) {
      return typeof window !== "undefined";
    }
  }
  var CHANGE = "change";
  var READ = "read";
  var MESSAGE_PACKET = "messagepacket";
  var HIDDEN = "hidden";
  var SHOWN = "shown";
  var CONNECT = "connect";
  var fakeSample = `
export function _sample() {
    while (true) {
        basic.showString("_sample() missing")
    }
    return [@samples@]
}`;
  var accelSample = `
export function _sample() {
    return [
        input.acceleration(Dimension.X) / 1024,
        input.acceleration(Dimension.Y) / 1024,
        input.acceleration(Dimension.Z) / 1024
    ]
}`;
  var buttonSample = `
let _button = Button.A
export function _sample() {
    return [input.buttonIsPressed(_button) ? 1 : 0]
}

//% block="set ml button %button" blockId="ml_set_button"
export function setButton(button: Button) {
    _button = button
}
`;
  var MakeCodeEditorExtensionClient = class {
    constructor() {
      this.pendingCommands = {};
      this.extensionId = inIFrame() ? window.location.hash.substr(1) : void 0;
      this._connected = false;
      this._visible = false;
      this.nextRequestId = 1;
      this.handleMessage = this.handleMessage.bind(this);
      window.addEventListener("message", this.handleMessage, false);
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
        return Promise.resolve(void 0);
      return new Promise((resolve, reject) => {
        const msg = this.mkRequest(resolve, reject, action, body);
        window.parent.postMessage(msg, "*");
      });
    }
    handleMessage(ev) {
      const msg = ev.data;
      if ((msg == null ? void 0 : msg.type) !== "pxtpkgext")
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
            this.emit("datastream", true);
            break;
          case "extconsole":
            this.emit("console", msg.body);
            break;
          case "extmessagepacket":
            this.emit(MESSAGE_PACKET, msg.body);
            break;
          default:
            console.debug("Unhandled event", msg);
        }
      } else {
        const { action, resolve, reject } = this.pendingCommands[msg.id] || {};
        delete this.pendingCommands[msg.id];
        if (msg.success && resolve)
          resolve(msg.resp);
        else if (!msg.success && reject)
          reject(msg.resp);
        switch (action) {
          case "extinit":
            this._connected = true;
            this.emit("CONNECT");
            this.emit(CHANGE);
            break;
          case "extusercode":
            this.emit("readuser", msg.resp);
            this.emit(CHANGE);
            break;
          case "extreadcode":
            this.emit(READ, msg.resp);
            this.emit(CHANGE);
            break;
          case "extwritecode":
            this.emit("written", void 0);
            break;
        }
      }
    }
    async init() {
      this.log(`initializing`);
      await this.sendRequest("extinit");
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
      } else {
        const resp = await this.sendRequest("extreadcode");
        return resp;
      }
    }
    async readUser() {
      await this.sendRequest("extusercode");
    }
    async write(code, json, jres, dependencies) {
      if (!this.extensionId) {
        this.emit("written", void 0);
      } else {
        await this.sendRequest("extwritecode", {
          code: code || void 0,
          json: json || void 0,
          jres: jres || void 0,
          dependencies
        });
      }
    }
    async queryPermission() {
      await this.sendRequest("extquerypermission");
    }
    async requestPermission(console2) {
      await this.sendRequest("extrequestpermission", {
        console: console2
      });
    }
    async dataStreamConsole(console2) {
      await this.sendRequest("extdatastream", {
        console: console2
      });
    }
    async dataStreamMessages(messages) {
      await this.sendRequest("extdatastream", {
        messages
      });
    }
  };
  async function start() {
    (0, import_tfjs.setBackend)("cpu");
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
    dropbox.addEventListener(
      "drop",
      (e) => {
        setStatus("reading model");
        stopEv(e);
        const file = e.dataTransfer.files.item(0);
        const reader = new FileReader();
        reader.onload = async (e2) => {
          try {
            const mod = JSON.parse(
              e2.target.result
            );
            await compileModel2(mod, file.name);
          } catch (e3) {
            console.error(e3.stack);
            setError(e3.message);
          }
        };
        reader.readAsText(file);
      },
      false
    );
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
    async function compileModel2(mod, fileName) {
      const name = mod.name || fileName;
      const ma = loadFlatJSONModel(mod);
      const m = await (0, import_tfjs.loadLayersModel)({ load: () => Promise.resolve(ma) });
      const inpTen = m.getInputAt(0);
      const numClasses = shapeElements(
        m.getOutputAt(0).shape
      );
      const labels = (mod.labels || []).slice();
      while (labels.length > numClasses)
        labels.pop();
      while (labels.length < numClasses)
        labels.push("class " + labels.length);
      const inputShape = inpTen.shape;
      const samplingPeriod = mod.inputInterval || 100;
      setStatus("compiling...");
      const res = await compileModelAndFullValidate(m, {
        verbose: false,
        includeTest: true,
        float16weights: options.f16,
        optimize: true
      });
      setStatus("compiled!");
      const shape2 = inputShape.filter((v) => v != null);
      const samplesInWindow = shape2.shift();
      const elementsInSample = shapeElements(shape2);
      let code = `// model: ${name}; input: ${JSON.stringify(
        inputShape
      )}; sampling at: ${samplingPeriod}ms
// ${res.memInfo}
// ${res.timeInfo}
`;
      code += "const enum MLEvent {\n";
      let idx = 0;
      for (let lbl of labels) {
        lbl = lbl.replace(/_/g, " ");
        code += `    //% block="${lbl}"
`;
        code += `    ${toCamelCase(lbl)} = ${idx},
`;
        idx++;
      }
      code += `}

`;
      code += `namespace ml {
`;
      code += `
            let _classifier: Classifier
            export function classifier() {
                if (_classifier) return _classifier
                _classifier = new Classifier(input => _model.invoke(input), _sample)
                _classifier.detectionThreshold = 0.7
                _classifier.samplingInterval = ${Math.round(
        samplingPeriod
      )} // ms
                _classifier.samplesOverlap = ${Math.max(
        samplesInWindow >> 2,
        1
      )}
                _classifier.samplesInWindow = ${samplesInWindow}
                _classifier.elementsInSample = ${elementsInSample}
                _classifier.noiseClassNo = -1 // disable
                _classifier.noiseSuppressionTime = 500 // ms
                _classifier.start()
                return _classifier
            }

            /**
             * Run some code when a particular ML event is detected.
             */
            //% blockId=ml_on_event block="on ml event %condition"
            //% blockGap=12
            export function onEvent(mlevent: MLEvent, handler: () => void) {
                classifier().onEvent(mlevent, handler)
            }
            `;
      let sample = fakeSample;
      if (elementsInSample == 1)
        sample = buttonSample;
      else if (elementsInSample == 3)
        sample = accelSample;
      const exampleSample = [];
      for (let i = 0; i < elementsInSample; ++i)
        exampleSample.push(i);
      code += "\n" + sample.replace("@sample@", JSON.stringify(exampleSample)) + "\n";
      code += `export const _model = new ml4f.Model(
hex\``;
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
      const d2 = document.createElement("div");
      d2.textContent = text;
      return d2;
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
      box.addEventListener("change", () => {
        if (box.checked)
          options[field] = !!box.checked;
      });
      maindiv.appendChild(lbl);
    }
  }
})();
//# sourceMappingURL=pxtml4f.js.map
