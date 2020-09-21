#!/bin/sh
set -x
FL="-masm-syntax-unified -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -O3 -mfloat-abi=hard"
arm-none-eabi-gcc -W -Wall $FL -c foo.s
arm-none-eabi-objcopy -O binary foo.o foo.bin
(echo "export const modelBuf = hex\`"
node -e 'console.log(require("fs").readFileSync("foo.bin").toString("hex").replace(/.{0,128}/g, s => s + "\n"))'
echo "\`") > foo.ts
