///<reference path="pxtpackage.d.ts" />

import { asmDeps, asmFns } from './library'
import * as U from './util'

export interface Options {
    verbose?: boolean
    testInput?: number[]
    testOutput?: number[]
    info?: string
    includeTest?: boolean
}

export interface ModelInfo {
    weights: number[];
    arenaSize: number;
    minArenaSize: number;
    inputShape: number[];
    outputShape: number[];
    outputOffset: number;
    opts: Options;
    stats: string;
}

export enum OpCode {
    comment,
    repeat,
    loadWeightAddr,
    loadDataAddr,
    addPtr,
    loadFConst,
    load,
    store,
    vmul,
    vmax,
    vadd,
    relu,
    fcall,
}

export enum Reg {
    S0 = 0,
    S1 = 1,
    S15 = 15,
    S31 = 32,
    InputPtr = 200,
    OutputPtr,
    KernelPtr,
    DataDescPtr,
    Index0 = 300,
    Tmp0 = 400,
    Zero = 500,
}

export interface Op {
    opcode: OpCode
    dst?: Reg
    src?: Reg
    srcAlt?: Reg
    num?: number
    isDef?: boolean
    increment?: boolean
    body?: Op[]
    fname?: string
}


function assert(cond: boolean, msg = "assertion failed") {
    if (!cond) {
        debugger
        throw new Error("ir: " + msg)
    }
}

const unrollLimit = 10

export function stringifyComment(msg: string) {
    if (!msg) return ""
    return "// " + msg.replace(/\n/g, "\n// ")
}

export function indent(s: string) {
    return "  " + s.replace(/\n$/, "").replace(/\n/g, "\n  ") + "\n"
}

export function numCycles(ops: Op[]): number {
    let cycles = 0
    let prevDst: Reg = null
    const addConst = (k: number) => k < (1 << 12) ? 1 : 2
    for (const op of ops) {
        switch (op.opcode) {
            case OpCode.comment:
                break
            case OpCode.repeat:
                cycles += (numCycles(op.body) + 4 + (op.isDef ? 1 : 0)) * op.num + 1
                break
            case OpCode.loadWeightAddr:
                cycles += 2 + addConst(op.num * 4)
                break
            case OpCode.loadDataAddr:
                cycles += addConst(op.num * 4 + 8)
                break
            case OpCode.addPtr:
                if (op.src == null)
                    cycles += addConst(op.num * 4)
                else {
                    if (op.num != 1) {
                        if (op.src > Reg.Zero) {
                            if (op.src == Reg.Zero + 1) { }
                            else if (op.src == Reg.Zero + 2) {
                                cycles++
                            }
                            else {
                                cycles += 2
                            }
                        } else {
                            cycles++
                        }
                    }
                    cycles += 2
                }
                if (op.num == 1)
                    cycles += 1
                else
                    cycles += 3
                break
            case OpCode.loadFConst:
                if (op.num == 0)
                    cycles += 2
                else if (op.num == 1)
                    cycles += 1
                else
                    cycles += 4 // ??
                break
            case OpCode.load:
                cycles += 1 + op.num
                break
            case OpCode.store:
                cycles += 1 + op.num
                break
            case OpCode.relu:
                cycles += 6
                break
            case OpCode.vmax:
                cycles += 4
                if (op.src != op.dst)
                    cycles++
                break
            case OpCode.vmul:
            case OpCode.vadd:
                if (op.src === prevDst || op.srcAlt === prevDst)
                    cycles += 2
                else
                    cycles += 1
                prevDst = op.dst
                break
            case OpCode.fcall:
                if (op.fname == "softmax")
                    cycles += 200 + op.num * 150 // estimate
                else
                    cycles += 500 + op.num * 500 // estimate
                break
            default:
                throw new Error("bad op " + op.opcode)
        }
    }
    return cycles
}

export function toThumb(modelInfo: ModelInfo, ops: Op[]) {
    const weightAddrDO = 0
    const zeroDO = 4
    const descWords = 2
    const usedFns: SMap<boolean> = {}

    const hasTest = !!modelInfo.opts.testInput && !!modelInfo.opts.includeTest
    let ind = ""
    const byteOffset = (n: number) => 4 * (n + descWords)
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
    ]
    for (let i = 0; i < 4; ++i)
        header.push(`0 // padding`)

    addShape(modelInfo.inputShape, "input")
    addShape(modelInfo.outputShape, "output")

    let initCmt = ""
    while (ops[0]?.opcode == OpCode.comment) {
        const op = ops.shift()
        initCmt += stringifyComment(op.fname) + "\n"
    }

    let regAlloc: SMap<number> = {}
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
`
    for (const h of header)
        write(`.word ${h}`)

    let lblid = 0

    regAlloc[Reg.InputPtr] = 1
    regAlloc[Reg.OutputPtr] = 2
    regAlloc[Reg.KernelPtr] = 3
    regAlloc[Reg.DataDescPtr] = 7

    write(`_start_model:`)
    write(`push {r4,r5,r6,r7,r8,r9,r10,r11,r12,lr}`)
    write(`mov ${reg(Reg.DataDescPtr)}, r1`)
    write(`ldr r1, [r0, #4*4] // weight offset`)
    write(`adds r1, r0 // weight addr`)
    write(`str r1, [${reg(Reg.DataDescPtr)}, #${weightAddrDO}]`)
    write(`movs r1, #0`)
    write(`str r1, [${reg(Reg.DataDescPtr)}, #${zeroDO}]`)
    compiles(ops)

    write(`pop {r4,r5,r6,r7,r8,r9,r10,r11,r12,pc}`)

    for (const k of Object.keys(usedFns)) {
        for (const d of asmDeps[k] || [])
            usedFns[d] = true
    }
    for (const k of Object.keys(usedFns)) {
        write(asmFns[k])
    }

    write(".balign 4")
    writeArray("_weights", modelInfo.weights)

    if (hasTest) {
        writeArray("_testInput", modelInfo.opts.testInput)
        writeArray("_testOutput", modelInfo.opts.testOutput)
    }

    write("_end:")

    return resText

    function writeArray(lbl: string, vals: number[]) {
        write(`${lbl}:`)
        for (const w of vals)
            write(`.float ${w}`)
    }

    function addShape(shape: number[], lbl: string) {
        for (const shp of shape)
            if (shp != null)
                header.push(`${shp} // ${lbl} shape`)
        header.push(`0 // end of ${lbl} shape`)
    }

    function alloc(r: Reg, f?: () => void) {
        assert(!regAlloc[r])
        const copy: SMap<number> = {}
        const used: SMap<boolean> = {}
        for (const k of Object.keys(regAlloc)) {
            copy[k] = regAlloc[k]
            used[copy[k]] = true
        }
        let all = -1
        for (let i = 4; i <= 12; ++i) {
            if (!used[i]) {
                all = i
                break
            }
        }
        if (all < 0)
            oops("can't alloc " + r)

        regAlloc[r] = all

        if (f) {
            const pind = ind
            try {
                ind += "    "
                f()
            } finally {
                ind = pind
                regAlloc = copy
            }
        }
    }

    function write(asm: string) {
        if (isFake(asm))
            oops("wrong reg: " + asm)
        resText += ind + asm + "\n"
    }

    function oops(msg: string) {
        throw new Error("internal thumb error: " + msg)
    }

    function reg(r: Reg) {
        if (r == null)
            return "<fake>"
        if (r <= Reg.S31)
            return "s" + (r - Reg.S0)
        if (r >= Reg.Zero)
            return "#" + (r - Reg.Zero)
        const id = regAlloc[r]
        if (id == undefined)
            return "<fake:" + regName(r) + ">"
        return "r" + id
    }

    function isFake(r: string) {
        return r.indexOf("<fake") >= 0
    }

    function isLowReg(reg: string) {
        return /^r[0-7]$/.test(reg)
    }

    function loadConst(dst: string, num: number) {
        // TODO?
        if (num <= 0xff && isLowReg(dst))
            write(`movs ${dst}, #${num}`)
        else
            write(`movw ${dst}, #${num}`)
    }

    function addConst(dst: string, src: string, num: number) {
        if (Math.abs(num) < (1 << 12)) {
            if (num < 0)
                write(`subw ${dst}, ${src}, #${-num}`)
            else
                write(`addw ${dst}, ${src}, #${num}`)
        } else {
            assert(src != dst)
            loadConst(dst, num)
            write(`adds ${dst}, ${src}, ${dst}`)
        }
    }

    function compiles(ops: Op[]) {
        for (const op of ops) compile(op)
    }

    function range(op: Op) {
        return "{" + U.range(op.num).map(k => reg(op.dst + k)).join(",") + "}"
    }

    function compile(op: Op) {
        let dst = reg(op.dst)
        const src = reg(op.src)
        const srcAlt = reg(op.srcAlt)
        const incr = op.increment ? "!" : ""

        switch (op.opcode) {
            case OpCode.comment:
                write(stringifyComment(op.fname))
                break
            case OpCode.repeat:
                assert(op.num >= 1)
                alloc(op.dst, () => {
                    dst = reg(op.dst)
                    const lbl = `.l.${lblid++}`
                    loadConst(dst, op.isDef ? 0 : op.num)
                    write(`${lbl}:  // rep ${op.num}`)
                    compiles(op.body)
                    if (op.isDef) {
                        write(`adds ${dst}, #1`)
                        write(`cmp ${dst}, #${op.num}`)
                        write(`blt ${lbl}`)
                    } else {
                        if (isLowReg(dst))
                            write(`subs ${dst}, #1`)
                        else
                            write(`subs ${dst}, ${dst}, #1`)
                        write(`bne ${lbl}`)
                    }
                })
                break
            case OpCode.loadWeightAddr:
                write(`ldr r0, [${reg(Reg.DataDescPtr)}, #${weightAddrDO}]`)
                addConst(dst, "r0", op.num * 4)
                break
            case OpCode.loadDataAddr:
                addConst(dst, reg(Reg.DataDescPtr), byteOffset(op.num))
                break
            case OpCode.addPtr:
                if (isFake(dst) && op.isDef) {
                    alloc(op.dst)
                    dst = reg(op.dst)
                }
                if (op.src == null) {
                    addConst(dst, srcAlt, op.num * 4)
                } else {
                    if (op.num != 1) {
                        loadConst("r0", op.num * 4)
                        if (src[0] == '#') {
                            const n = +src.slice(1)
                            if (n == 0)
                                loadConst("r0", 0)
                            else if (n == 1) {
                                // do nothing
                            } else if (n == 2) {
                                write(`adds r0,r0`)
                            } else {
                                assert(dst != srcAlt)
                                loadConst(dst, n)
                                write(`muls r0, ${dst}`)
                            }
                        } else {
                            write(`muls r0, ${src}`)
                        }
                    } else {
                        if (src[0] == '#') {
                            const n = +src.slice(1)
                            loadConst("r0", n << 2)
                        } else {
                            write(`lsls r0, ${src}, #2`)
                        }
                    }
                    write(`adds ${dst}, ${srcAlt}, r0`)
                }
                break
            case OpCode.loadFConst:
                if (op.num == 0.0)
                    write(`vldr ${dst}, [${reg(Reg.DataDescPtr)}, #${zeroDO}]`)
                else
                    write(`vmov ${dst}, #${op.num}e+0`)
                break
            case OpCode.load:
                write(`vldm ${src}${incr}, ${range(op)}`)
                break
            case OpCode.store:
                write(`vstm ${src}${incr}, ${range(op)}`)
                break
            case OpCode.relu:
                write(`ldr r0, [${dst}, #0]`)
                // negative check on FP and int is the same
                write(`cmp r0, #0`)
                write(`it lt`)
                // int 0 is same as 0.0f
                // this could be movslt but GAS always assembles this as movw, so for bit-exactness we stick to movw
                write(`movwlt r0, #0`)
                write(`stm ${dst}!, {r0}`)
                break
            case OpCode.vmul:
                write(`vmul.f32 ${dst}, ${src}, ${srcAlt}`)
                break
            case OpCode.vadd:
                write(`vadd.f32 ${dst}, ${src}, ${srcAlt}`)
                break
            case OpCode.vmax:
                assert(dst != srcAlt)
                if (src != dst)
                    write(`vmov ${dst}, ${src}`)
                write(`vcmp.f32 ${dst}, ${srcAlt}`)
                write(`vmrs APSR_nzcv, FPSCR`)
                write(`it mi`)
                write(`vmovmi.f32 ${dst}, ${srcAlt}`)
                break
            case OpCode.fcall:
                write(`mov r0, ${dst}`)
                loadConst("r1", op.num)
                write(`bl ${op.fname}`)
                usedFns[op.fname] = true
                break
            default:
                oops("bad op " + op.opcode)
        }
    }
}

function toJS(modelInfo: ModelInfo, op: Op): string {
    const dst = op.dst == null ? null : reg(op.dst)
    const src = op.src == null ? null : reg(op.src)
    const srcAlt = op.srcAlt == null ? null : reg(op.srcAlt)

    switch (op.opcode) {
        case OpCode.comment:
            return stringifyComment(op.fname) + "\n"
        case OpCode.repeat:
            return `for (let ${dst} = 0; ${dst} < ${op.num}; ${dst}++) {\n${indent(toJSs(modelInfo, op.body))}}\n`
        case OpCode.loadWeightAddr:
            return `${dst} = weightOff + ${op.num}\n`
        case OpCode.loadDataAddr:
            return `${dst} = dataOff + ${op.num}\n`
        case OpCode.addPtr:
            if (op.src == null)
                return `${dst} = ${srcAlt} + ${op.num}\n`
            return `${dst} = ${srcAlt} + ${src}${op.num == 1 ? "" : " * " + op.num}\n`
        case OpCode.loadFConst:
            return `${dst} = ${op.num}\n`
        case OpCode.load: {
            let r = ""
            let dp = op.dst + 0
            if (op.increment) {
                for (let i = 0; i < op.num; ++i)
                    r += `${reg(dp++)} = mem[${src}++]\n`
            } else {
                for (let i = 0; i < op.num; ++i)
                    r += `${reg(dp++)} = mem[${src} + ${i}]\n`
            }
            return r
        }
        case OpCode.store: {
            let r = ""
            let dp = op.dst + 0
            if (op.increment) {
                for (let i = 0; i < op.num; ++i)
                    r += `mem[${src}++] = ${reg(dp++)}\n`
            } else {
                for (let i = 0; i < op.num; ++i)
                    r += `mem[${src} + ${i}] = ${reg(dp++)}\n`
            }
            return r
        }
        case OpCode.relu:
            return `if (mem[${dst}] < 0) mem[${dst}] = 0; ${dst}++\n`
        case OpCode.vmul:
            return `${dst} = f32(${src} * ${srcAlt})\n`
        case OpCode.vadd:
            return `${dst} = f32(${src} + ${srcAlt})\n`
        case OpCode.vmax:
            return `${dst} = Math.max(${src}, ${srcAlt})\n`
        case OpCode.fcall:
            return `${op.fname}(${dst}, ${op.num})`
        default:
            throw new Error("bad op " + op.opcode)
    }


    function reg(r: Reg) {
        const res = regName(r)
        if (res[0] === "?")
            assert(false, "bad reg " + r)
        return res
    }

}

function regName(r: Reg) {
    if (r <= Reg.S31)
        return "s" + (r - Reg.S0)
    if (r >= Reg.Zero)
        return "" + (r - Reg.Zero)
    if (r >= Reg.Tmp0)
        return "tmp" + (r - Reg.Tmp0)
    if (r >= Reg.Index0)
        return "idx" + (r - Reg.Index0)
    switch (r) {
        case Reg.InputPtr:
            return "input"
        case Reg.KernelPtr:
            return "kernel"
        case Reg.OutputPtr:
            return "output"
        default:
            return "???" + r
    }
}

export function toJSs(modelInfo: ModelInfo, op: Op[]) {
    return op.map(o => toJS(modelInfo, o)).join("")
}

let repIdx = 0
export function repeatIdx(n: number, f: (idx: Reg) => Op[]): Op {
    const idx = Reg.Index0 + repIdx++
    return {
        opcode: OpCode.repeat,
        dst: idx,
        num: n,
        body: f(idx),
        isDef: true
    }
}

export function repeat(n: number, f: () => Op[]): Op {
    const r = repeatIdx(n, f)
    r.isDef = false
    return r
}

export function comment(str: string): Op {
    return {
        opcode: OpCode.comment,
        fname: str
    }
}

export function loadWeightAddr(dst: Reg, idx: number): Op {
    assert(idx >= 0)
    return {
        opcode: OpCode.loadWeightAddr,
        dst,
        num: idx
    }
}

export function loadDataAddr(dst: Reg, idx: number): Op {
    assert(idx >= 0)
    return {
        opcode: OpCode.loadDataAddr,
        dst,
        num: idx
    }
}

export function addPtr(dst: Reg, idx: Reg | null, mult = 1, base?: Reg): Op {
    if (!base) base = dst
    return {
        opcode: OpCode.addPtr,
        dst,
        src: idx,
        srcAlt: base,
        num: mult
    }
}

export function load0(dst: number): Op {
    return {
        opcode: OpCode.loadFConst,
        dst,
        num: 0.0
    }
}

export function load(dst: Reg, num: number, src: Reg, adv: boolean): Op {
    return {
        opcode: OpCode.load,
        dst,
        src,
        num: num,
        increment: adv
    }
}

export function store(dst: Reg, src: Reg, num: number, adv: boolean): Op {
    return {
        opcode: OpCode.store,
        src: dst,
        dst: src,
        num: num,
        increment: adv
    }
}

export function relu(dst: Reg): Op {
    return {
        opcode: OpCode.relu,
        dst,
        increment: true
    }
}

export function vmul(dst: Reg, a: Reg, b: Reg) {
    return {
        opcode: OpCode.vmul,
        dst,
        src: a,
        srcAlt: b,
    }
}

export function vmax(dst: Reg, a: Reg, b: Reg) {
    if (b == dst)
        [a, b] = [b, a]
    return {
        opcode: OpCode.vmax,
        dst,
        src: a,
        srcAlt: b,
    }
}

export function vadd(dst: Reg, a: Reg, b: Reg) {
    return {
        opcode: OpCode.vadd,
        dst,
        src: a,
        srcAlt: b,
    }
}

export function fcall(name: string, dst: Reg, len: number): Op {
    return {
        opcode: OpCode.fcall,
        fname: name,
        dst,
        num: len,
    }
}

export function flatten(...args: (Op | Op[] | Op[][])[]) {
    const res: Op[] = []
    const add = (a: Op) => {
        if (a) res.push(a)
    }
    for (const a of args) {
        if (Array.isArray(a)) {
            for (const b of a) {
                if (Array.isArray(b)) {
                    for (const c of b)
                        add(c)
                } else {
                    add(b)
                }
            }
        } else {
            add(a)
        }
    }
    return res
}

export function optimize(ops: Op[], replMap: SMap<Reg> = {}): Op[] {
    const repl = (r: Reg) => {
        if (!r) return r
        if (replMap[r] != undefined)
            return replMap[r]
        return r
    }

    const res: Op[] = []
    for (let op of ops) {
        op = {
            opcode: op.opcode,
            dst: repl(op.dst),
            src: repl(op.src),
            srcAlt: repl(op.srcAlt),
            isDef: op.isDef,
            increment: op.increment,
            num: op.num,
            body: op.body,
            fname: op.fname
        }
        switch (op.opcode) {
            case OpCode.repeat:
                if (op.num == 0) { }
                else if (op.num == 1) {
                    replMap[op.dst] = Reg.Zero
                    U.pushRange(res, optimize(op.body, replMap))
                } else {
                    op.body = optimize(op.body, replMap)

                    const stripLoop = op.num * op.body.length < unrollLimit * 2
                    const canUnroll = !op.isDef && 2 * op.body.length < unrollLimit

                    if (stripLoop) {
                        for (let i = 0; i < op.num; ++i) {
                            replMap[op.dst] = Reg.Zero + i
                            U.pushRange(res, optimize(op.body, replMap))
                        }
                    } else if (canUnroll) {
                        const unrollCnt = (unrollLimit / op.body.length) | 0
                        const tmp = op.body.slice()
                        for (let i = 1; i < unrollCnt; ++i)
                            U.pushRange(op.body, tmp)
                        const newnum = (op.num / unrollCnt) | 0
                        res.push(op)
                        const left = op.num - newnum * unrollCnt
                        op.num = newnum
                        for (let i = 0; i < left; ++i)
                            U.pushRange(res, tmp)
                    } else {
                        res.push(op)
                    }
                }
                break
            case OpCode.addPtr:
                if (op.dst == op.srcAlt && (op.num == 0 || op.src == Reg.Zero)) { }
                else res.push(op)
                break
            default:
                res.push(op)
        }
    }
    return res
}

export function reset() {
    repIdx = 0
}
