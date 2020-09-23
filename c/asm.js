const fs = require("fs")

let mbuf = "const modelBuf = hex\`"
const objbin = fs.readFileSync("foo.bin")
mbuf += objbin.toString("hex").replace(/.{0,128}/g, s => s + "\n")
mbuf += "\`\n"
fs.writeFileSync("foo.ts", mbuf)
const obj2bin = Buffer.from(fs.readFileSync("foo.s", "utf8").replace(/[^]*BUF: /, ""), "hex")
fs.writeFileSync("foo2.bin", obj2bin)

const elfobj = fs.readFileSync("foo.o")
const idx = elfobj.indexOf(objbin.slice(0, 8))
if (idx <= 0) throw "blah"
elfobj.set(obj2bin.slice(0,objbin.length), idx)
fs.writeFileSync("foo2.o", elfobj)
