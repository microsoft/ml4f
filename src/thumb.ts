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

import * as assembler from "./assembler";
import { assert, lookup } from "./util";

const thumbRegs: pxt.Map<number> = {
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
}

const armConditions: SMap<number> = {
    "eq": 0,
    "ne": 1,
    "cs": 2,
    "hs": 2, // cs
    "cc": 3,
    "lo": 3, // cc
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
}

let fpRegs: pxt.Map<number>

export class ThumbProcessor extends assembler.AbstractProcessor {
    runtimeIsARM = false

    constructor() {
        super();

        if (!fpRegs) {
            fpRegs = {}
            for (let i = 0; i < 32; ++i)
                fpRegs["s" + i] = i
        }

        const allConds = (f: (cond: string, id: number) => void, inclAl = false) => {
            for (const k of Object.keys(armConditions))
                if (armConditions[k] != 14 || inclAl)
                    f(k, armConditions[k])
        }

        // Registers
        // $r0 - bits 2:1:0
        // $r1 - bits 5:4:3
        // $r2 - bits 7:2:1:0
        // $r3 - bits 6:5:4:3
        // $r4 - bits 8:7:6
        // $r5 - bits 10:9:8

        this.addEnc("$r0", "R0-7", v => this.inrange(7, v, v))
        this.addEnc("$r1", "R0-7", v => this.inrange(7, v, v << 3))
        this.addEnc("$r2", "R0-15", v => this.inrange(15, v, (v & 7) | ((v & 8) << 4)))
        this.addEnc("$r3", "R0-15", v => this.inrange(15, v, v << 3))
        this.addEnc("$r4", "R0-7", v => this.inrange(7, v, v << 6))
        this.addEnc("$r5", "R0-7", v => this.inrange(7, v, v << 8))
        // this for setting both $r0 and $r1 (two argument adds and subs)
        this.addEnc("$r01", "R0-7", v => this.inrange(7, v, (v | v << 3)))

        // Immdiates:
        // $i0 - bits 7-0
        // $i1 - bits 7-0 * 4
        // $i2 - bits 6-0 * 4
        // $i3 - bits 8-6
        // $i4 - bits 10-6
        // $i5 - bits 10-6 * 4
        // $i6 - bits 10-6, 0 is 32
        // $i7 - bits 10-6 * 2

        this.addEnc("$i0", "#0-255", v => this.inrange(255, v, v))
        this.addEnc("$i1", "#0-1020", v => this.inrange(255, v / 4, v >> 2))
        this.addEnc("$i2", "#0-510", v => this.inrange(127, v / 4, v >> 2))
        this.addEnc("$i3", "#0-7", v => this.inrange(7, v, v << 6))
        this.addEnc("$i4", "#0-31", v => this.inrange(31, v, v << 6))
        this.addEnc("$i5", "#0-124", v => this.inrange(31, v / 4, (v >> 2) << 6))
        this.addEnc("$i6", "#1-32", v => v == 0 ? null : v == 32 ? 0 : this.inrange(31, v, v << 6))
        this.addEnc("$i7", "#0-62", v => this.inrange(31, v / 2, (v >> 1) << 6))
        this.addEnc("$i32", "#0-2^32", v => 1)

        this.addEnc("$rl0", "{R0-7,...}", v => this.inrange(255, v, v))
        this.addEnc("$rl1", "{LR,R0-7,...}", v => (v & 0x4000) ? this.inrange(255, (v & ~0x4000), 0x100 | (v & 0xff)) : this.inrange(255, v, v))
        this.addEnc("$rl2", "{PC,R0-7,...}", v => (v & 0x8000) ? this.inrange(255, (v & ~0x8000), 0x100 | (v & 0xff)) : this.inrange(255, v, v))


        this.addEnc("$la", "LABEL", v => this.inrange(255, v / 4, v >> 2)).isWordAligned = true;
        this.addEnc("$lb", "LABEL", v => this.inrangeSigned(127, v / 2, v >> 1))
        this.addEnc("$lb11", "LABEL", v => this.inrangeSigned(1023, v / 2, v >> 1))

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

        allConds((cond, id) =>
            this.addInst(`b${cond} $lb`, 0xd000 | (id << 8), 0xff00))

        this.addInst("b     $lb11", 0xe000, 0xf800);
        this.addInst("bal   $lb11", 0xe000, 0xf800);

        // handled specially - 32 bit instruction
        this.addInst("bl    $lb", 0xf000, 0xf800, true);
        // this is normally emitted as 'b' but will be emitted as 'bl' if needed
        this.addInst("bb    $lb", 0xe000, 0xf800, true);

        // this will emit as PC-relative LDR or ADDS
        this.addInst("ldlit   $r5, $i32", 0x4800, 0xf800);

        // 32 bit encodings
        this.addEnc("$RL0", "{R0-15,...}", v => this.inrange(0xffff, v, v))
        this.addEnc("$R0", "R0-15", v => this.inrange(15, v, v << 8)) // 8-11
        this.addEnc("$R1", "R0-15", v => this.inrange(15, v, v << 16)) // 16-19
        this.addEnc("$R2", "R0-15", v => this.inrange(15, v, v << 12)) // 12-15
        this.addEnc("$I0", "#0-4095", v => this.inrange(4095, v, (v & 0xff) | ((v & 0x700) << 4) | ((v & 0x800) << 15)))
        this.addEnc("$I1", "#0-4095", v => this.inrange(4095, v, v))
        this.addEnc("$I2", "#0-65535", v => this.inrange(0xffff, v,
            (v & 0xff) | ((v & 0x700) << 4) | ((v & 0x800) << 15) | ((v & 0xf000) << 4)))

        this.addEnc("$LB", "LABEL", v => {
            const q = ((v >> 1) & 0x7ff)
                | (((v >> 12) & 0x3f) << 16)
                | (((v >> 18) & 0x1) << 13)
                | (((v >> 19) & 0x1) << 11)
                | (((v >> 20) & 0x1) << 26)
            if (this.inrangeSigned((1 << 20) - 1, v / 2, q) == null)
                return null
            return q
        })

        this.addEnc("$S0", "S0-31", v => this.inrange(31, v, ((v >> 1) << 0) | ((v & 1) << 5))) // 0-3 + 5
        this.addEnc("$S1", "S0-31", v => this.inrange(31, v, ((v >> 1) << 12) | ((v & 1) << 22))) // 12-15 + 22
        this.addEnc("$S2", "S0-31", v => this.inrange(31, v, ((v >> 1) << 16) | ((v & 1) << 7))) // 16-19 + 7

        this.addEnc("$SL0", "{S0-S31}",
            v => {
                v |= 0
                const v0 = v
                if (!v) return null
                let reg0 = 0
                while (reg0 < 32 && 0 == (v & (1 << reg0)))
                    reg0++
                v >>>= reg0
                if (!v) return null
                let num = 0
                while (v & 1) {
                    v >>= 1
                    num++
                }
                if (v) return null // non-consecutive
                v = reg0
                // console.log(v0.toString(16), v, num)
                return ((v >> 1) << 12) | ((v & 1) << 22) | num
            })

        this.addInst32("push  $RL0", 0xe92d0000, 0xffff0000)
        this.addInst32("pop   $RL0", 0xe8bd0000, 0xffff0000)
        this.addInst32("addw  $R0, $R1, $I0", 0xf2000000, 0xfbf08000)
        this.addInst32("subw  $R0, $R1, $I0", 0xf2a00000, 0xfbf08000)
        this.addInst32("ldr   $R2, [$R1, $I1]", 0xf8d00000, 0xfff00000);
        this.addInst32("str   $R2, [$R1, $I1]", 0xf8c00000, 0xfff00000);
        this.addInst32("movw  $R0, $I2", 0xf2400000, 0xfbf08000);

        allConds((cond, id) =>
            this.addInst32(`b${cond} $LB`, 0xf0008000 | (id << 22), 0xfbc0d000), true)

        allConds((cond, id) =>
            this.addInst(`it ${cond}`, 0xbf08 | (id << 4), 0xffff), true)

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
        this.addInst32("vmrs         APSR_nzcv, fpscr", 0xeef1fa10, 0xffffffff);
        this.addInst32("vmrs         APSR_nzcv, FPSCR", 0xeef1fa10, 0xffffffff);
        this.addInst32("vmov.f32     $S1, $S0", 0xeeb00a40, 0xffbf0fd0);

        /*
        vmsr
        vpush
        vpop
        vrint
        vsel
        */

    }

    public stripCondition(name: string): string {
        if (name.length >= 5) {
            const dot = name.indexOf(".")
            let suff = ""
            if (dot > 0) {
                suff = name.slice(dot)
                name = name.slice(0, dot)
            }
            if (armConditions[name.slice(-2)])
                return name.slice(0, -2) + suff
        }
        return null
    }

    public toFnPtr(v: number, baseOff: number, lbl: string) {
        if (this.runtimeIsARM && /::/.test(lbl))
            return (v + baseOff) & ~1;
        return (v + baseOff) | 1;
    }

    public wordSize() {
        return 4
    }

    public is32bit(i: assembler.Instruction) {
        return i.name == "bl" || i.name == "bb" || i.is32bit;
    }

    public postProcessAbsAddress(f: assembler.File, v: number) {
        // Thumb addresses have last bit set, but we are ourselves always
        // in Thumb state, so to go to ARM state, we signal that with that last bit
        v ^= 1
        v -= f.baseOffset
        return v
    }

    public emit32(v0: number, v: number, actual: string): assembler.EmitResult {
        let isBLX = v % 2 ? true : false
        if (isBLX) {
            v = (v + 1) & ~3
        }
        let off = v >> 1
        assert(off != null)
        // Range is +-4M (i.e., 2M instructions)
        if ((off | 0) != off ||
            !(-2 * 1024 * 1024 < off && off < 2 * 1024 * 1024))
            return assembler.emitErr("jump out of range", actual);

        // note that off is already in instructions, not bytes
        let imm11 = off & 0x7ff
        let imm10 = (off >> 11) & 0x3ff

        return {
            opcode: (off & 0xf0000000) ? (0xf400 | imm10) : (0xf000 | imm10),
            opcode2: isBLX ? (0xe800 | imm11) : (0xf800 | imm11),
            stack: 0,
            numArgs: [v],
            labelName: actual
        }
    }

    public expandLdlit(f: assembler.File): void {
        let nextGoodSpot: assembler.Line
        let needsJumpOver = false
        let outlines: assembler.Line[] = []
        let values: pxt.Map<string> = {}
        let seq = 1

        for (let i = 0; i < f.lines.length; ++i) {
            let line = f.lines[i]
            outlines.push(line)
            if (line.type == "instruction" && line.instruction && line.instruction.name == "ldlit") {
                if (!nextGoodSpot) {
                    let limit = line.location + 900 // leave some space - real limit is 1020
                    let j = i + 1
                    for (; j < f.lines.length; ++j) {
                        if (f.lines[j].location > limit)
                            break
                        let op = f.lines[j].getOp()
                        if (op == "b" || op == "bb" || (op == "pop" && f.lines[j].words[2] == "pc"))
                            nextGoodSpot = f.lines[j]
                    }
                    if (nextGoodSpot) {
                        needsJumpOver = false
                    } else {
                        needsJumpOver = true
                        while (--j > i) {
                            if (f.lines[j].type == "instruction") {
                                nextGoodSpot = f.lines[j]
                                break
                            }
                        }
                    }
                }
                let reg = line.words[1]
                // make sure the key in values[] below doesn't look like integer
                // we rely on Object.keys() returning stuff in insertion order, and integers mess with it
                // see https://www.ecma-international.org/ecma-262/6.0/#sec-ordinary-object-internal-methods-and-internal-slots-ownpropertykeys
                // or possibly https://www.stefanjudis.com/today-i-learned/property-order-is-predictable-in-javascript-objects-since-es2015/
                let v = "#" + line.words[3]
                let lbl = lookup(values, v)
                if (!lbl) {
                    lbl = "_ldlit_" + ++seq
                    values[v] = lbl
                }
                line.update(`ldr ${reg}, ${lbl}`)
            }
            if (line === nextGoodSpot) {
                nextGoodSpot = null
                let txtLines: string[] = []
                let jmplbl = "_jmpwords_" + ++seq
                if (needsJumpOver)
                    txtLines.push("bb " + jmplbl)
                txtLines.push(".balign 4")
                for (let v of Object.keys(values)) {
                    let lbl = values[v]
                    txtLines.push(lbl + ": .word " + v.slice(1))
                }
                if (needsJumpOver)
                    txtLines.push(jmplbl + ":")
                for (let t of txtLines) {
                    f.buildLine(t, outlines)
                    let ll = outlines[outlines.length - 1]
                    ll.scope = line.scope
                    ll.lineNo = line.lineNo
                }
                values = {}
            }
        }
        f.lines = outlines
    }

    public getAddressFromLabel(f: assembler.File, i: assembler.Instruction, s: string, wordAligned = false): number {
        let l = f.lookupLabel(s);
        if (l == null) return null;
        let pc = f.location() + 4
        if (wordAligned) pc = pc & 0xfffffffc
        return l - pc;
    }

    public isPop(opcode: number): boolean {
        return opcode == 0xbc00;
    }

    public isPush(opcode: number): boolean {
        return opcode == 0xb400;
    }

    public isAddSP(opcode: number): boolean {
        return opcode == 0xb000;
    }

    public isSubSP(opcode: number): boolean {
        return opcode == 0xb080;
    }

    public peephole(ln: assembler.Line, lnNext: assembler.Line, lnNext2: assembler.Line) {
        let lb11 = this.encoders["$lb11"]
        let lb = this.encoders["$lb"]

        // +/-8 bytes is because the code size can slightly change due to .balign directives
        // inserted by literal generation code; see https://github.com/Microsoft/pxt-adafruit/issues/514
        // Most likely 4 would be enough, but we play it safe
        function fits(enc: assembler.Encoder, ln: assembler.Line) {
            return (
                enc.encode(ln.numArgs[0] + 8) != null &&
                enc.encode(ln.numArgs[0] - 8) != null &&
                enc.encode(ln.numArgs[0]) != null
            )
        }

        let lnop = ln.getOp()
        let isSkipBranch = false
        if (lnop == "bne" || lnop == "beq") {
            if (lnNext.getOp() == "b" && ln.numArgs[0] == 0)
                isSkipBranch = true;
            if (lnNext.getOp() == "bb" && ln.numArgs[0] == 2)
                isSkipBranch = true;
        }

        if (lnop == "bb" && fits(lb11, ln)) {
            // RULE: bb .somewhere -> b .somewhere (if fits)
            ln.update("b " + ln.words[1])
        } else if (lnop == "b" && ln.numArgs[0] == -2) {
            // RULE: b .somewhere; .somewhere: -> .somewhere:
            ln.update("")
        } else if (lnop == "bne" && isSkipBranch && fits(lb, lnNext)) {
            // RULE: bne .next; b .somewhere; .next: -> beq .somewhere
            ln.update("beq " + lnNext.words[1])
            lnNext.update("")
        } else if (lnop == "beq" && isSkipBranch && fits(lb, lnNext)) {
            // RULE: beq .next; b .somewhere; .next: -> bne .somewhere
            ln.update("bne " + lnNext.words[1])
            lnNext.update("")
        } else if (lnop == "push" && ln.numArgs[0] == 0x4000 && lnNext.getOp() == "push" && !(lnNext.numArgs[0] & 0x4000)) {
            // RULE: push {lr}; push {X, ...} -> push {lr, X, ...}
            ln.update(lnNext.text.replace("{", "{lr, "))
            lnNext.update("")
        } else if (lnop == "pop" && lnNext.getOp() == "pop" && lnNext.numArgs[0] == 0x8000) {
            // RULE: pop {X, ...}; pop {pc} -> push {X, ..., pc}
            ln.update(ln.text.replace("}", ", pc}"))
            lnNext.update("")
        } else if (lnop == "push" && lnNext.getOp() == "pop" && ln.numArgs[0] == lnNext.numArgs[0]) {
            // RULE: push {X}; pop {X} -> nothing
            assert(ln.numArgs[0] > 0)
            ln.update("")
            lnNext.update("")
        } else if (lnop == "push" && lnNext.getOp() == "pop" &&
            ln.words.length == 4 &&
            lnNext.words.length == 4) {
            // RULE: push {rX}; pop {rY} -> mov rY, rX
            assert(ln.words[1] == "{")
            ln.update("mov " + lnNext.words[2] + ", " + ln.words[2])
            lnNext.update("")
        } else if (lnNext2 && ln.getOpExt() == "movs $r5, $i0" && lnNext.getOpExt() == "mov $r0, $r1" &&
            ln.numArgs[0] == lnNext.numArgs[1] &&
            clobbersReg(lnNext2, ln.numArgs[0])) {
            // RULE: movs rX, #V; mov rY, rX; clobber rX -> movs rY, #V
            ln.update("movs r" + lnNext.numArgs[0] + ", #" + ln.numArgs[1])
            lnNext.update("")
        } else if (lnop == "pop" && singleReg(ln) >= 0 && lnNext.getOp() == "push" &&
            singleReg(ln) == singleReg(lnNext)) {
            // RULE: pop {rX}; push {rX} -> ldr rX, [sp, #0]
            ln.update("ldr r" + singleReg(ln) + ", [sp, #0]")
            lnNext.update("")
        } else if (lnop == "push" && lnNext.getOpExt() == "ldr $r5, [sp, $i1]" &&
            singleReg(ln) == lnNext.numArgs[0] && lnNext.numArgs[1] == 0) {
            // RULE: push {rX}; ldr rX, [sp, #0] -> push {rX}
            lnNext.update("")
        } else if (lnNext2 && lnop == "push" && singleReg(ln) >= 0 && preservesReg(lnNext, singleReg(ln)) &&
            lnNext2.getOp() == "pop" && singleReg(ln) == singleReg(lnNext2)) {
            // RULE: push {rX}; movs rY, #V; pop {rX} -> movs rY, #V (when X != Y)
            ln.update("")
            lnNext2.update("")
        }
    }

    public registerNo(actual: string, enc: assembler.Encoder) {
        if (!actual) return null;
        actual = actual.toLowerCase()
        let map = thumbRegs
        if (enc.name[1] == "S") {
            map = fpRegs
        }
        const r = map[actual]
        if (r === undefined)
            return null
        return r
    }

    public testAssembler() {
        assembler.expectError(this, "lsl r0, r0, #8");
        //assembler.expectError(this, "push {pc,lr}");
        assembler.expectError(this, "push {r17}");
        assembler.expectError(this, "mov r0, r1 foo");
        assembler.expectError(this, "movs r14, #100");
        assembler.expectError(this, "push {r0");
        assembler.expectError(this, "push lr,r0}");
        //assembler.expectError(this, "pop {lr,r0}");
        assembler.expectError(this, "b #+11");
        assembler.expectError(this, "b #+10240000");
        assembler.expectError(this, "bne undefined_label");
        assembler.expectError(this, ".foobar");

        assembler.expect(this,
            "0200      lsls    r0, r0, #8\n" +
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
            "fffe\n")

        assembler.expect(this,
            "4291      cmp     r1, r2\n" +
            "d100      bne     l6\n" +
            "e000      b       l8\n" +
            "1840  l6: adds    r0, r0, r1\n" +
            "4718  l8: bx      r3\n")

        assembler.expect(this,
            "          @stackmark base\n" +
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
            "          @stackempty base\n")

        assembler.expect(this,
            "b090      sub sp, #4*16\n" +
            "b010      add sp, #4*16\n"
        )

        assembler.expect(this,
            "6261      .string \"abc\"\n" +
            "0063      \n"
        )

        assembler.expect(this,
            "6261      .string \"abcde\"\n" +
            "6463      \n" +
            "0065      \n"
        )

        assembler.expect(this,
            "3042      adds r0, 0x42\n" +
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
            "9003      str r0, [sp, #4*3]\n")
    }
}


// if true then instruction doesn't write r<n> and doesn't read/write memory
function preservesReg(ln: assembler.Line, n: number) {
    if (ln.getOpExt() == "movs $r5, $i0" && ln.numArgs[0] != n)
        return true;
    return false;
}

function clobbersReg(ln: assembler.Line, n: number) {
    // TODO add some more
    if (ln.getOp() == "pop" && ln.numArgs[0] & (1 << n))
        return true;
    return false;
}

function singleReg(ln: assembler.Line) {
    assert(ln.getOp() == "push" || ln.getOp() == "pop")
    let k = 0;
    let ret = -1;
    let v = ln.numArgs[0]
    while (v > 0) {
        if (v & 1) {
            if (ret == -1) ret = k;
            else ret = -2;
        }
        v >>= 1;
        k++;
    }
    if (ret >= 0) return ret;
    else return -1;
}
