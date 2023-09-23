//**************************************************************************************
//  Copyright (C) 2017 - 2022, Min Tang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"

#pragma once
#include "forceline.h"

class tri3f {
public:
	unsigned int _ids[3];

	FORCEINLINE tri3f() {
		_ids[0] = _ids[1] = _ids[2] = -1;
	}

	FORCEINLINE tri3f(unsigned int id0, unsigned int id1, unsigned int id2) {
		set(id0, id1, id2);
	}

	FORCEINLINE void set(unsigned int id0, unsigned int id1, unsigned int id2) {
		_ids[0] = id0;
		_ids[1] = id1;
		_ids[2] = id2;
	}

	FORCEINLINE unsigned int id(int i) { return _ids[i]; }
	FORCEINLINE unsigned int id0() {return _ids[0];}
	FORCEINLINE unsigned int id1() {return _ids[1];}
	FORCEINLINE unsigned int id2() {return _ids[2];}

	__device__ unsigned int id0_dev() { return _ids[0]; }
	__device__ unsigned int id1_dev() { return _ids[1]; }
	__device__ unsigned int id2_dev() { return _ids[2]; }
};


