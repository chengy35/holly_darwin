# set the binaries that have to be built
TARGETS := killpca

# set the build configuration set
#BUILD := release
BUILD := debug

# set bin and build dirs
BUILDDIR := .build_$(BUILD)
BINDIR := $(BUILD)

# libraries
LDLIBS = $(addprefix -l, $(LIBS) $(LIBS_$(notdir $*)))
LIBS := \
	opencv_core opencv_imgproc opencv_video opencv_nonfree opencv_features2d opencv_flann opencv_calib3d opencv_objdetect opencv_legacy opencv_highgui \
	avformat avutil avcodec swscale z

# set some flags and compiler/linker specific commands
CXXFLAGS = -w -pipe -D __STDC_CONSTANT_MACROS -D STD=std -Wall $(CXXFLAGS_$(BUILD)) -I. -I/usr/local/include
CXXFLAGS_release := -O3 -DNDEBUG -ggdb
LDFLAGS = -L/usr/local/lib -pipe -Wall $(LDFLAGS_$(BUILD))
LDFLAGS_debug := -w -ggdb
LDFLAGS_release := -O3 -ggdb

include make/generic.mk
