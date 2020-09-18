#!/bin/sh
set -x
FL="-masm-syntax-unified -mthumb -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -O3 -mfloat-abi=hard"
arm-none-eabi-gcc -W -Wall $FL -c foo.s
