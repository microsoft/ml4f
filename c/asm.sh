#!/bin/sh
set -x
FL="-masm-syntax-unified -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -O3 -mfloat-abi=hard"
cp ../tmp/built/model.asm foo.s
arm-none-eabi-gcc -W -Wall $FL -c foo.s
arm-none-eabi-objcopy -O binary foo.o foo.bin
node asm
hexdump -C foo.bin > foo.hex
hexdump -C foo2.bin > foo2.hex
diff -u foo.hex foo2.hex > foo.diff
arm-none-eabi-objdump -d foo.o > foo.dump
arm-none-eabi-objdump -d foo2.o > foo2.dump
diff -u foo.dump foo2.dump > foo-dump.diff
