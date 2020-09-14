#!/bin/sh
arm-none-eabi-gcc -mthumb -mcpu=cortex-m4 -std=c99 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O3 -c -S tt.c 
