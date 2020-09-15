#!/bin/sh
set -x
FL="-mthumb -mcpu=cortex-m4 -std=c99 -mfpu=fpv4-sp-d16 -O3 -mfloat-abi=hard"
arm-none-eabi-gcc $FL -S tt.c 
arm-none-eabi-gcc $FL -O3 -c tt.c 
arm-none-eabi-gcc $FL tt.o -o tt.elf -lm
