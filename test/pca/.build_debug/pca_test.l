-include .build_debug/pca_test.d

.build_debug/pca_test.l: \
	.build_debug/pca_test.d \

debug/pca_test: .build_debug/pca_test.l \
	.build_debug/pca_test.o \

