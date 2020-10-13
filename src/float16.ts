/// based on: Fast Half Float Conversions, Jeroen van der Zijp, link: http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf

const basetable = new Uint16Array(512);
const shifttable = new Uint8Array(512);

const mantissatable = new Uint32Array(2048);
const offsettable = new Uint16Array(64);
const exponenttable = new Uint32Array(64);

let inited = false

function init() {
    inited = true
    for (let i = 0; i < 256; ++i) {
        const e = i - 127;
        if (e < -24) { // Very small numbers map to zero
            basetable[i | 0x000] = 0x0000;
            basetable[i | 0x100] = 0x8000;
            shifttable[i | 0x000] = 24;
            shifttable[i | 0x100] = 24;
        } else if (e < -14) { // Small numbers map to denorms
            basetable[i | 0x000] = (0x0400 >> (-e - 14));
            basetable[i | 0x100] = (0x0400 >> (-e - 14)) | 0x8000;
            shifttable[i | 0x000] = -e - 1;
            shifttable[i | 0x100] = -e - 1;
        } else if (e <= 15) { // Normal numbers just lose precision
            basetable[i | 0x000] = ((e + 15) << 10);
            basetable[i | 0x100] = ((e + 15) << 10) | 0x8000;
            shifttable[i | 0x000] = 13;
            shifttable[i | 0x100] = 13;
        } else if (e < 128) { // Large numbers map to Infinity
            basetable[i | 0x000] = 0x7C00;
            basetable[i | 0x100] = 0xFC00;
            shifttable[i | 0x000] = 24;
            shifttable[i | 0x100] = 24;
        } else { // Infinity and NaN's stay Infinity and NaN's
            basetable[i | 0x000] = 0x7C00;
            basetable[i | 0x100] = 0xFC00;
            shifttable[i | 0x000] = 13;
            shifttable[i | 0x100] = 13;
        }
    }

    for (let i = 1; i < 2048; ++i) {
        if (i < 1024)
            mantissatable[i] = convertmantissa(i)
        else
            mantissatable[i] = 0x38000000 + ((i - 1024) << 13)
    }

    exponenttable[32] = 0x80000000
    exponenttable[31] = 0x47800000
    exponenttable[63] = 0xC7800000
    for (let i = 1; i <= 30; ++i)
        exponenttable[i] = i << 23
    for (let i = 33; i <= 62; ++i)
        exponenttable[i] = 0x80000000 + (i - 32) << 23

    for (let i = 1; i < offsettable.length; ++i)
        offsettable[i] = 1024
    offsettable[32] = 0

    function convertmantissa(i: number) {
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

export function float32ToUInt32(v: number) {
    const buf = new Float32Array(1)
    buf[0] = v
    return new Uint32Array(buf.buffer)[0]
}

export function float16toUInt16(v: number) {
    const f = float32ToUInt32(v)
    if (!inited) init()
    return basetable[(f >> 23) & 0x1ff] | ((f & 0x007fffff) >> shifttable[(f >> 23) & 0x1ff])
}

export function float16AsUintToFloat(h: number) {
    const tmp = mantissatable[offsettable[h >> 10] + (h & 0x3ff)] + exponenttable[h >> 10]
    const buf = new Uint32Array(1)
    buf[0] = tmp
    return new Float32Array(buf.buffer)[0]
}

export function testFloatConv() {
    for (let i = 0; i < 30000; ++i) {
        test(i)
        test(-i)
        test(1 / i)
        test(-1 / i)
        test(1 / (i * 100))
        test(-1 / (i * 100))
    }

    function test(v: number) {
        const u = float16toUInt16(v)
        const v2 = float16AsUintToFloat(u)
        const d = Math.min(10000 * Math.abs(v - v2), Math.abs(v - v2) / v)
        if (d > 0.002) {
            throw new Error(`fail: ${v} -> ${u} -> ${v2} (dd=${v - v2} d=${d})`)
        }
    }
}
