export function assert(cond: boolean, msg = "Assertion failed") {
    if (!cond) {
        debugger
        throw new Error(msg)
    }
}

export function userError(msg: string): Error {
    let e = new Error(msg);
    (<any>e).isUserError = true;
    throw e
}


export function lookup<T>(m: pxt.Map<T>, key: string): T {
    if (m.hasOwnProperty(key))
        return m[key]
    return null
}


export function oops(msg = "OOPS"): Error {
    debugger
    throw new Error(msg)
}


export function endsWith(str: string, suffix: string) {
    if (str.length < suffix.length) return false
    if (suffix.length == 0) return true
    return str.slice(-suffix.length) == suffix
}

export function startsWith(str: string, prefix: string) {
    if (str.length < prefix.length) return false
    if (prefix.length == 0) return true
    return str.slice(0, prefix.length) == prefix
}


export function iterMap<T>(m: pxt.Map<T>, f: (k: string, v: T) => void) {
    Object.keys(m).forEach(k => f(k, m[k]))
}

export function mapMap<T, S>(m: pxt.Map<T>, f: (k: string, v: T) => S) {
    let r: pxt.Map<S> = {}
    Object.keys(m).forEach(k => r[k] = f(k, m[k]))
    return r
}


export function pushRange<T>(trg: T[], src: ArrayLike<T>): void {
    for (let i = 0; i < src.length; ++i)
        trg.push(src[i])
}

// TS gets lost in type inference when this is passed an array
export function concatArrayLike<T>(arrays: ArrayLike<ArrayLike<T>>): T[] {
    return concat(arrays as any)
}

export function concat<T>(arrays: T[][]): T[] {
    let r: T[] = []
    for (let i = 0; i < arrays.length; ++i) {
        pushRange(r, arrays[i])
    }
    return r
}

export function range(len: number) {
    let r: number[] = []
    for (let i = 0; i < len; ++i) r.push(i)
    return r
}

let seed = 13 * 0x1000193

export function seedRandom(v: number) {
    seed = (v * 0x1000193) >>> 0
}

export function randomUint32() {
    let x = seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    seed = x;
    return x;
}

export function randomInclusive(min: number, max: number) {
    return min + randomUint32() % (max - min + 1)
}

export function randomPermute<T>(arr: T[]) {
    for (let i = 0; i < arr.length; ++i) {
        let j = randomUint32() % arr.length
        let tmp = arr[i]
        arr[i] = arr[j]
        arr[j] = tmp
    }
}

export function randomPick<T>(arr: T[]): T {
    if (arr.length == 0) return null;
    return arr[randomUint32() % arr.length];
}

export function randomUFloat() {
    return randomUint32() / 0x1_0000_0000;
}

export function randomSFloat() {
    return 2 * randomUFloat() - 1;
}
