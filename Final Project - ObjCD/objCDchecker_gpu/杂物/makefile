objects = hom3.o

hom3 : hom3.o
	nvcc -o hom3 $(objects) -Xcompiler -fopenmp
hom3.o : hom3.cu
	nvcc -c hom3.cu -I /usr/local/cuda-10.0/samples/common/inc -arch=sm_75 -lgomp

.PHONY: clean run
clean: 
	-rm hom3 $(objects)
run : 
	-./hom3
