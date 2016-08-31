-include .build_debug/killpca.d

.build_debug/killpca.l: \
	.build_debug/killpca.d \

debug/killpca: .build_debug/killpca.l \
	.build_debug/killpca.o \

