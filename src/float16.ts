/// based on: Fast Half Float Conversions, Jeroen van der Zijp, link: http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf

const basetable = new Uint16Array(512);
const shifttable = new Uint8Array(512);
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
