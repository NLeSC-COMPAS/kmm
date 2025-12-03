CLANG_FORMAT=clang-format --verbose

pretty:
	${CLANG_FORMAT} -i include/kmm/*.hpp include/kmm/*/*.hpp
	$(CLANG_FORMAT) -i src/*/*.cpp src/*/*.cu src/*/*.cuh
	${CLANG_FORMAT} -i test/*/*.cpp
	${CLANG_FORMAT} -i examples/*.cu
	${CLANG_FORMAT} -i benchmarks/*.cu

all: pretty

.PHONY : pretty
