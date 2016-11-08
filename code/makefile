VERSION=2.4.13_3
LIB_PATHS=-L/usr/local/Cellar/opencv/${VERSION}/lib
LIBS=-lopencv_highgui -lopencv_core -lopencv_imgproc
all: sharpen

sharpen: sharpen.cpp pixel.cpp
	g++ sharpen.cpp pixel.cpp ${LIB_PATHS} ${LIBS} -o sharpen

.PHONY:clean

clean:
	rm sharpen
