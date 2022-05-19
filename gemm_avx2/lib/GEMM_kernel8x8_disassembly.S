Disassembly of section .text:

0000000000000000 <_Z15gemm_kernel_8x8iPfiS_iS_i>:
   0:	41 57                	push   %r15
   2:	41 56                	push   %r14
   4:	53                   	push   %rbx
   5:	48 63 44 24 20       	movslq 0x20(%rsp),%rax
   a:	c4 c1 7c 10 09       	vmovups (%r9),%ymm1
   f:	c4 c1 7c 10 04 81    	vmovups (%r9,%rax,4),%ymm0
  15:	8d 14 00             	lea    (%rax,%rax,1),%edx
  18:	48 63 d2             	movslq %edx,%rdx
  1b:	c4 c1 7c 10 14 91    	vmovups (%r9,%rdx,4),%ymm2
  21:	8d 1c 40             	lea    (%rax,%rax,2),%ebx
  24:	4c 63 c3             	movslq %ebx,%r8
  27:	c4 81 7c 10 1c 81    	vmovups (%r9,%r8,4),%ymm3
  2d:	8d 1c 85 00 00 00 00 	lea    0x0(,%rax,4),%ebx
  34:	4c 63 d3             	movslq %ebx,%r10
  37:	c4 81 7c 10 24 91    	vmovups (%r9,%r10,4),%ymm4
  3d:	8d 1c 80             	lea    (%rax,%rax,4),%ebx
  40:	4c 63 db             	movslq %ebx,%r11
  43:	c4 81 7c 10 2c 99    	vmovups (%r9,%r11,4),%ymm5
  49:	8d 1c 52             	lea    (%rdx,%rdx,2),%ebx
  4c:	4c 63 f3             	movslq %ebx,%r14
  4f:	c4 81 7c 10 34 b1    	vmovups (%r9,%r14,4),%ymm6
  55:	8d 1c c5 00 00 00 00 	lea    0x0(,%rax,8),%ebx
  5c:	29 c3                	sub    %eax,%ebx
  5e:	4c 63 fb             	movslq %ebx,%r15
  61:	c4 81 7c 10 3c b9    	vmovups (%r9,%r15,4),%ymm7
  67:	85 ff                	test   %edi,%edi
  69:	7e 71                	jle    dc <_Z15gemm_kernel_8x8iPfiS_iS_i+0xdc>
  6b:	31 db                	xor    %ebx,%ebx
  6d:	0f 1f 00             	nopl   (%rax)
  70:	c5 7c 10 04 19       	vmovups (%rcx,%rbx,1),%ymm8
  75:	c4 62 7d 18 0c 1e    	vbroadcastss (%rsi,%rbx,1),%ymm9
  7b:	c4 c2 3d b8 c9       	vfmadd231ps %ymm9,%ymm8,%ymm1
  80:	c4 62 7d 18 4c 1e 04 	vbroadcastss 0x4(%rsi,%rbx,1),%ymm9
  87:	c4 c2 3d b8 c1       	vfmadd231ps %ymm9,%ymm8,%ymm0
  8c:	c4 62 7d 18 4c 1e 08 	vbroadcastss 0x8(%rsi,%rbx,1),%ymm9
  93:	c4 c2 3d b8 d1       	vfmadd231ps %ymm9,%ymm8,%ymm2
  98:	c4 62 7d 18 4c 1e 0c 	vbroadcastss 0xc(%rsi,%rbx,1),%ymm9
  9f:	c4 c2 3d b8 d9       	vfmadd231ps %ymm9,%ymm8,%ymm3
  a4:	c4 62 7d 18 4c 1e 10 	vbroadcastss 0x10(%rsi,%rbx,1),%ymm9
  ab:	c4 c2 3d b8 e1       	vfmadd231ps %ymm9,%ymm8,%ymm4
  b0:	c4 62 7d 18 4c 1e 14 	vbroadcastss 0x14(%rsi,%rbx,1),%ymm9
  b7:	c4 c2 3d b8 e9       	vfmadd231ps %ymm9,%ymm8,%ymm5
  bc:	c4 62 7d 18 4c 1e 18 	vbroadcastss 0x18(%rsi,%rbx,1),%ymm9
  c3:	c4 c2 3d b8 f1       	vfmadd231ps %ymm9,%ymm8,%ymm6
  c8:	c4 62 7d 18 4c 1e 1c 	vbroadcastss 0x1c(%rsi,%rbx,1),%ymm9
  cf:	c4 c2 3d b8 f9       	vfmadd231ps %ymm9,%ymm8,%ymm7
  d4:	48 83 c3 20          	add    $0x20,%rbx
  d8:	ff cf                	dec    %edi
  da:	75 94                	jne    70 <_Z15gemm_kernel_8x8iPfiS_iS_i+0x70>
  dc:	c4 c1 7c 11 09       	vmovups %ymm1,(%r9)
  e1:	c4 c1 7c 11 04 81    	vmovups %ymm0,(%r9,%rax,4)
  e7:	c4 c1 7c 11 14 91    	vmovups %ymm2,(%r9,%rdx,4)
  ed:	c4 81 7c 11 1c 81    	vmovups %ymm3,(%r9,%r8,4)
  f3:	c4 81 7c 11 24 91    	vmovups %ymm4,(%r9,%r10,4)
  f9:	c4 81 7c 11 2c 99    	vmovups %ymm5,(%r9,%r11,4)
  ff:	c4 81 7c 11 34 b1    	vmovups %ymm6,(%r9,%r14,4)
 105:	c4 81 7c 11 3c b9    	vmovups %ymm7,(%r9,%r15,4)
 10b:	5b                   	pop    %rbx
 10c:	41 5e                	pop    %r14
 10e:	41 5f                	pop    %r15
 110:	c5 f8 77             	vzeroupper 
 113:	c3                   	retq   

Disassembly of section .text.startup:

0000000000000000 <_GLOBAL__sub_I_GEMM_new.cpp>:
   0:	53                   	push   %rbx
   1:	48 8d 1d 00 00 00 00 	lea    0x0(%rip),%rbx        # 8 <_GLOBAL__sub_I_GEMM_new.cpp+0x8>
   8:	48 89 df             	mov    %rbx,%rdi
   b:	e8 00 00 00 00       	callq  10 <_GLOBAL__sub_I_GEMM_new.cpp+0x10>
  10:	48 8b 3d 00 00 00 00 	mov    0x0(%rip),%rdi        # 17 <_GLOBAL__sub_I_GEMM_new.cpp+0x17>
  17:	48 8d 15 00 00 00 00 	lea    0x0(%rip),%rdx        # 1e <_GLOBAL__sub_I_GEMM_new.cpp+0x1e>
  1e:	48 89 de             	mov    %rbx,%rsi
  21:	5b                   	pop    %rbx
  22:	e9 00 00 00 00       	jmpq   27 <_GLOBAL__sub_I_GEMM_new.cpp+0x27>
