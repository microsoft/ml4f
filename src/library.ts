export const asmDeps: SMap<string[]> = {
    'softmax': ['expf_asm']
}

export const asmFns: SMap<string> = {
    "expf_asm": `
// based on https://stackoverflow.com/questions/29381117
expf_asm:
	vldr.32	s15, .L10
	vcmpe.f32	s0, s15
	vmrs	APSR_nzcv, FPSCR
	bmi	.L5
	vldr.32	s15, .L10+4
	vcmpe.f32	s0, s15
	vmrs	APSR_nzcv, FPSCR
	bgt	.L9
	vldr.32	s15, .L10+8
	vldr.32	s9, .L10+12
	vldr.32	s6, .L10+16
	vldr.32	s7, .L10+20
	vldr.32	s10, .L10+24
	vldr.32	s8, .L10+28
	vldr.32	s11, .L10+32
	vldr.32	s12, .L10+36
	vldr.32	s13, .L10+40
	vmul.f32	s15, s0, s15
	vmov.f32	s14, #1.0e+0
	vadd.f32	s15, s15, s9
	vsub.f32	s15, s15, s9
	vfma.f32	s0, s15, s6
	vcvt.s32.f32	s9, s15
	vfma.f32	s0, s15, s7
	vmov.f32	s15, s10
	vfma.f32	s15, s8, s0
	vmov	r3, s9	@ int
	vfma.f32	s11, s15, s0
	vfma.f32	s12, s11, s0
	vfma.f32	s13, s12, s0
	vmov.f32	s15, s13
	vmov.f32	s13, s14
	vfma.f32	s13, s15, s0
	vfma.f32	s14, s13, s0
	vmov	r2, s14	@ int
	add	r3, r2, r3, lsl #24
	vmov	s0, r3	@ int
	bx	lr
.L9:
	vldr.32	s15, .L10+44
	vmov.f32	s14, #1.0e+0
	vdiv.f32	s0, s14, s15
	bx	lr
.L5:
	vldr.32	s0, .L10+44
	bx	lr
.L11:
	.align	2
.L10:
	.word	3268542464
	.word	1121058816
	.word	1069066811
	.word	1262485504
	.word	3207688704
	.word	3049242254
	.word	1007234926
	.word	984915968
	.word	1026207149
	.word	1042983464
	.word	1056964603
	.word	0
`,

    "softmax": `
softmax:
	cmp	r1, #1
	push	{r3, r4, r5, lr}
	vldr.32	s5, [r0]
	bls	.L13
	adds	r3, r0, #4
	add	r2, r0, r1, lsl #2
.L16:
	vldmia.32	r3!, {s15}
	vcmp.f32	s15, s5
	vmrs	APSR_nzcv, FPSCR
	it	gt
	vmovgt.f32	s5, s15
	cmp	r2, r3
	bne	.L16
.L17:
	movs	r4, #0
	vmov	s4, r4
	mov	r5, r0
.L19:
	vldr.32	s0, [r5]
	vsub.f32	s0, s0, s5
	bl	expf_asm
	adds	r4, r4, #1
	cmp	r1, r4
	vadd.f32	s4, s4, s0
	vstmia.32	r5!, {s0}
	bhi	.L19
	movs	r3, #0
.L20:
	vldr.32	s14, [r0]
	vdiv.f32	s15, s14, s4
	adds	r3, r3, #1
	cmp	r1, r3
	vstmia.32	r0!, {s15}
	bhi	.L20
	pop	{r3, r4, r5, pc}
.L13:
	cmp	r1, #0
	bne	.L17
	pop	{r3, r4, r5, pc}
`

}