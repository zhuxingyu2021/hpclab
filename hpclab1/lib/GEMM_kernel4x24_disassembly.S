Disassembly of section .text:

0000000000000000 <_Z16gemm_kernel_4x24iPfiS_iS_i>:
   0:	55                   	push   %rbp
   1:	41 57                	push   %r15
   3:	41 56                	push   %r14
   5:	41 54                	push   %r12
   7:	53                   	push   %rbx
   8:	48 63 44 24 30       	movslq 0x30(%rsp),%rax
   d:	c4 c1 7c 10 01       	vmovups (%r9),%ymm0
  12:	c4 c1 7c 10 49 04    	vmovups 0x4(%r9),%ymm1
  18:	c4 c1 7c 10 51 08    	vmovups 0x8(%r9),%ymm2
  1e:	c4 c1 7c 10 1c 81    	vmovups (%r9,%rax,4),%ymm3
  24:	c4 c1 7c 10 6c 81 04 	vmovups 0x4(%r9,%rax,4),%ymm5
  2b:	c4 c1 7c 10 64 81 08 	vmovups 0x8(%r9,%rax,4),%ymm4
  32:	4c 8d 50 01          	lea    0x1(%rax),%r10
  36:	4c 8d 40 02          	lea    0x2(%rax),%r8
  3a:	8d 14 00             	lea    (%rax,%rax,1),%edx
  3d:	48 63 d2             	movslq %edx,%rdx
  40:	c4 c1 7c 10 3c 91    	vmovups (%r9,%rdx,4),%ymm7
  46:	c4 c1 7c 10 74 91 08 	vmovups 0x8(%r9,%rdx,4),%ymm6
  4d:	8d 6c 00 01          	lea    0x1(%rax,%rax,1),%ebp
  51:	4c 63 dd             	movslq %ebp,%r11
  54:	c4 01 7c 10 04 99    	vmovups (%r9,%r11,4),%ymm8
  5a:	4c 8d 72 02          	lea    0x2(%rdx),%r14
  5e:	8d 1c 40             	lea    (%rax,%rax,2),%ebx
  61:	48 63 db             	movslq %ebx,%rbx
  64:	c4 41 7c 10 1c 99    	vmovups (%r9,%rbx,4),%ymm11
  6a:	c4 41 7c 10 54 99 04 	vmovups 0x4(%r9,%rbx,4),%ymm10
  71:	c4 41 7c 10 4c 99 08 	vmovups 0x8(%r9,%rbx,4),%ymm9
  78:	4c 8d 7b 01          	lea    0x1(%rbx),%r15
  7c:	4c 8d 63 02          	lea    0x2(%rbx),%r12
  80:	85 ff                	test   %edi,%edi
  82:	0f 8e 5d 01 00 00    	jle    1e5 <_Z16gemm_kernel_4x24iPfiS_iS_i+0x1e5>
  88:	83 ff 01             	cmp    $0x1,%edi
  8b:	0f 84 ed 00 00 00    	je     17e <_Z16gemm_kernel_4x24iPfiS_iS_i+0x17e>
  91:	89 fd                	mov    %edi,%ebp
  93:	83 e5 fe             	and    $0xfffffffe,%ebp
  96:	f7 dd                	neg    %ebp
  98:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  9f:	00 
  a0:	c4 62 7d 18 26       	vbroadcastss (%rsi),%ymm12
  a5:	c5 7c 10 29          	vmovups (%rcx),%ymm13
  a9:	c5 7c 10 71 20       	vmovups 0x20(%rcx),%ymm14
  ae:	c5 7c 10 79 40       	vmovups 0x40(%rcx),%ymm15
  b3:	c4 c2 1d b8 c5       	vfmadd231ps %ymm13,%ymm12,%ymm0
  b8:	c4 c2 1d b8 ce       	vfmadd231ps %ymm14,%ymm12,%ymm1
  bd:	c4 c2 05 b8 d4       	vfmadd231ps %ymm12,%ymm15,%ymm2
  c2:	c4 62 7d 18 66 04    	vbroadcastss 0x4(%rsi),%ymm12
  c8:	c4 c2 1d b8 dd       	vfmadd231ps %ymm13,%ymm12,%ymm3
  cd:	c4 c2 1d b8 ee       	vfmadd231ps %ymm14,%ymm12,%ymm5
  d2:	c4 c2 05 b8 e4       	vfmadd231ps %ymm12,%ymm15,%ymm4
  d7:	c4 62 7d 18 66 08    	vbroadcastss 0x8(%rsi),%ymm12
  dd:	c4 c2 1d b8 fd       	vfmadd231ps %ymm13,%ymm12,%ymm7
  e2:	c4 42 1d b8 c6       	vfmadd231ps %ymm14,%ymm12,%ymm8
  e7:	c4 c2 05 b8 f4       	vfmadd231ps %ymm12,%ymm15,%ymm6
  ec:	c4 62 7d 18 66 0c    	vbroadcastss 0xc(%rsi),%ymm12
  f2:	c4 42 1d b8 dd       	vfmadd231ps %ymm13,%ymm12,%ymm11
  f7:	c5 7c 10 69 60       	vmovups 0x60(%rcx),%ymm13
  fc:	c4 42 1d b8 d6       	vfmadd231ps %ymm14,%ymm12,%ymm10
 101:	c4 42 1d b8 cf       	vfmadd231ps %ymm15,%ymm12,%ymm9
 106:	c5 7c 10 a1 80 00 00 	vmovups 0x80(%rcx),%ymm12
 10d:	00 
 10e:	c5 7c 10 b1 a0 00 00 	vmovups 0xa0(%rcx),%ymm14
 115:	00 
 116:	c4 62 7d 18 7e 10    	vbroadcastss 0x10(%rsi),%ymm15
 11c:	c4 c2 05 b8 c5       	vfmadd231ps %ymm13,%ymm15,%ymm0
 121:	c4 c2 05 b8 cc       	vfmadd231ps %ymm12,%ymm15,%ymm1
 126:	c4 c2 0d b8 d7       	vfmadd231ps %ymm15,%ymm14,%ymm2
 12b:	c4 62 7d 18 7e 14    	vbroadcastss 0x14(%rsi),%ymm15
 131:	c4 c2 05 b8 dd       	vfmadd231ps %ymm13,%ymm15,%ymm3
 136:	c4 c2 05 b8 ec       	vfmadd231ps %ymm12,%ymm15,%ymm5
 13b:	c4 c2 0d b8 e7       	vfmadd231ps %ymm15,%ymm14,%ymm4
 140:	c4 62 7d 18 7e 18    	vbroadcastss 0x18(%rsi),%ymm15
 146:	c4 c2 05 b8 fd       	vfmadd231ps %ymm13,%ymm15,%ymm7
 14b:	c4 42 05 b8 c4       	vfmadd231ps %ymm12,%ymm15,%ymm8
 150:	c4 c2 0d b8 f7       	vfmadd231ps %ymm15,%ymm14,%ymm6
 155:	c4 62 7d 18 7e 1c    	vbroadcastss 0x1c(%rsi),%ymm15
 15b:	c4 42 05 b8 dd       	vfmadd231ps %ymm13,%ymm15,%ymm11
 160:	c4 42 05 b8 d4       	vfmadd231ps %ymm12,%ymm15,%ymm10
 165:	c4 42 05 b8 ce       	vfmadd231ps %ymm14,%ymm15,%ymm9
 16a:	48 81 c1 c0 00 00 00 	add    $0xc0,%rcx
 171:	48 83 c6 20          	add    $0x20,%rsi
 175:	83 c5 02             	add    $0x2,%ebp
 178:	0f 85 22 ff ff ff    	jne    a0 <_Z16gemm_kernel_4x24iPfiS_iS_i+0xa0>
 17e:	40 f6 c7 01          	test   $0x1,%dil
 182:	74 61                	je     1e5 <_Z16gemm_kernel_4x24iPfiS_iS_i+0x1e5>
 184:	c5 7c 10 21          	vmovups (%rcx),%ymm12
 188:	c5 7c 10 69 20       	vmovups 0x20(%rcx),%ymm13
 18d:	c5 7c 10 71 40       	vmovups 0x40(%rcx),%ymm14
 192:	c4 62 7d 18 7e 0c    	vbroadcastss 0xc(%rsi),%ymm15
 198:	c4 42 05 b8 ce       	vfmadd231ps %ymm14,%ymm15,%ymm9
 19d:	c4 42 05 b8 d5       	vfmadd231ps %ymm13,%ymm15,%ymm10
 1a2:	c4 42 1d b8 df       	vfmadd231ps %ymm15,%ymm12,%ymm11
 1a7:	c4 62 7d 18 7e 08    	vbroadcastss 0x8(%rsi),%ymm15
 1ad:	c4 c2 05 b8 f6       	vfmadd231ps %ymm14,%ymm15,%ymm6
 1b2:	c4 42 05 b8 c5       	vfmadd231ps %ymm13,%ymm15,%ymm8
 1b7:	c4 c2 1d b8 ff       	vfmadd231ps %ymm15,%ymm12,%ymm7
 1bc:	c4 62 7d 18 7e 04    	vbroadcastss 0x4(%rsi),%ymm15
 1c2:	c4 c2 05 b8 e6       	vfmadd231ps %ymm14,%ymm15,%ymm4
 1c7:	c4 c2 05 b8 ed       	vfmadd231ps %ymm13,%ymm15,%ymm5
 1cc:	c4 c2 1d b8 df       	vfmadd231ps %ymm15,%ymm12,%ymm3
 1d1:	c4 62 7d 18 3e       	vbroadcastss (%rsi),%ymm15
 1d6:	c4 c2 05 b8 d6       	vfmadd231ps %ymm14,%ymm15,%ymm2
 1db:	c4 c2 05 b8 cd       	vfmadd231ps %ymm13,%ymm15,%ymm1
 1e0:	c4 c2 05 b8 c4       	vfmadd231ps %ymm12,%ymm15,%ymm0
 1e5:	c4 c1 7c 11 01       	vmovups %ymm0,(%r9)
 1ea:	c4 c1 7c 11 49 04    	vmovups %ymm1,0x4(%r9)
 1f0:	c4 c1 7c 11 51 08    	vmovups %ymm2,0x8(%r9)
 1f6:	c4 c1 7c 11 1c 81    	vmovups %ymm3,(%r9,%rax,4)
 1fc:	c4 81 7c 11 2c 91    	vmovups %ymm5,(%r9,%r10,4)
 202:	c4 81 7c 11 24 81    	vmovups %ymm4,(%r9,%r8,4)
 208:	c4 c1 7c 11 3c 91    	vmovups %ymm7,(%r9,%rdx,4)
 20e:	c4 01 7c 11 04 99    	vmovups %ymm8,(%r9,%r11,4)
 214:	c4 81 7c 11 34 b1    	vmovups %ymm6,(%r9,%r14,4)
 21a:	c4 41 7c 11 1c 99    	vmovups %ymm11,(%r9,%rbx,4)
 220:	c4 01 7c 11 14 b9    	vmovups %ymm10,(%r9,%r15,4)
 226:	c4 01 7c 11 0c a1    	vmovups %ymm9,(%r9,%r12,4)
 22c:	5b                   	pop    %rbx
 22d:	41 5c                	pop    %r12
 22f:	41 5e                	pop    %r14
 231:	41 5f                	pop    %r15
 233:	5d                   	pop    %rbp
 234:	c5 f8 77             	vzeroupper 
 237:	c3                   	retq   

