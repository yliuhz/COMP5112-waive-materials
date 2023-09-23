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

#include "cmesh.cuh"
#include <set>
#include <iostream>
#include "tri.cuh"

using namespace std;
//#include "mat3f.h"
//#include "box.h"

#include <omp.h>

#include <immintrin.h>
#include <stdint.h>

#define HANDLE_ERROR checkCudaErrors

#define START_GPU {\
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("GPU Time used:  %3.1f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));}


# define	TIMING_BEGIN \
	{double tmp_timing_start = omp_get_wtime();

# define	TIMING_END(message) \
	{double tmp_timing_finish = omp_get_wtime();\
	double  tmp_timing_duration = tmp_timing_finish - tmp_timing_start;\
	printf("%s: %2.5f seconds\n", (message), tmp_timing_duration);}}


//#define POVRAY_EXPORT
#define OBJ_DIR "e:\\temp\\output-objs\\"

//#define VEC_CLOTH

#pragma warning(disable: 4996)

mesh* cloths[16];
mesh* lions[16];

set<int> cloth_set;
set<int> lion_set;
set<int> dummy_set;

BOX g_box;
static int sidx = 0;

extern void clearFronts();
double ww, hh, dd;

#ifdef HI_RES
int Nx = 501;
int Nz = 501;
double xmin = 0.f, xmax = 500.f;
double zmin = 0.f, zmax = 500.f;
#else
int Nx = 101;
int Nz = 101;
double xmin = 0.f, xmax = 200.f;
double zmin = 0.f, zmax = 200.f;
#endif

#include "cmesh.cuh"
//#include "mesh_defs.h"
#include <vector>
using namespace std;

// for fopen
#pragma warning(disable: 4996)

bool readtrfile(const char* path, vec3f& shift)
{
	FILE* fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (strstr(buf, "<translate")) { //translate
			char* idx = strstr(buf, "x=\"");
			if (idx) {
				sscanf(idx + strlen("x=\""), "%lf", &shift.x);
			}

			idx = strstr(buf, "y=\"");
			if (idx) {
				sscanf(idx + strlen("y=\""), "%lf", &shift.y);
			}

			idx = strstr(buf, "z=\"");
			if (idx) {
				sscanf(idx + strlen("z=\""), "%lf", &shift.z);
			}
		}
	}

	fclose(fp);
	return true;
}

bool readobjfile_Vtx(const char* path, unsigned int numVtx, vec3f* vtxs, double scale, vec3f shift, bool swap_xyz)
{
	FILE* fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	int idx = 0;
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
			double x, y, z;
			sscanf(buf + 2, "%lf%lf%lf", &x, &y, &z);

			vec3f p;
			if (swap_xyz)
				p = vec3f(z, x, y) * scale + shift;
			else
				p = vec3f(x, y, z) * scale + shift;

			vtxs[idx++] = p;
		}
	}

	if (idx != numVtx)
		printf("vtx num do not match!\n");

	fclose(fp);
	return true;
}

bool readobjfile(const char* path,
	unsigned int& numVtx, unsigned int& numTri,
	tri3f*& tris, vec3f*& vtxs, double scale, vec3f shift, bool swap_xyz)
{
	vector<tri3f> triset;
	vector<vec3f> vtxset;

	FILE* fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
			double x, y, z;
			sscanf(buf + 2, "%lf%lf%lf", &x, &y, &z);

			if (swap_xyz)
				vtxset.push_back(vec3f(z, x, y) * scale + shift);
			else
				vtxset.push_back(vec3f(x, y, z) * scale + shift);
		}
		else
			if (buf[0] == 'f' && buf[1] == ' ') {
				int id0, id1, id2, id3 = 0;
				bool quad = false;

				sscanf(buf + 2, "%d", &id0);
				char* nxt = strchr(buf + 2, ' ');
				sscanf(nxt + 1, "%d", &id1);
				nxt = strchr(nxt + 1, ' ');
				sscanf(nxt + 1, "%d", &id2);

				nxt = strchr(nxt + 1, ' ');
				if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
					if (sscanf(nxt + 1, "%d", &id3))
						quad = true;
				}

				id0--, id1--, id2--, id3--;
				triset.push_back(tri3f(id0, id1, id2));

				if (quad)
					triset.push_back(tri3f(id0, id2, id3));
			}
	}
	fclose(fp);

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;

	numVtx = vtxset.size();
	vtxs = new vec3f[numVtx];
	for (unsigned int i = 0; i < numVtx; i++)
		vtxs[i] = vtxset[i];

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i = 0; i < numTri; i++)
		tris[i] = triset[i];

	return true;
}


bool read_transf(const char* path, matrix3f& trf, vec3f& shifted, bool swap_xyz)
{
	// reading this typical input ...
	//<rotate angle="109.093832" x="-0.115552" y="0.038672" z="-0.992548"/>
	//<scale value="1.000000"/>
	//<translate x="-0.181155" y="0.333876" z="0.569190"/>

	FILE* fp = fopen(path, "rt");
	if (fp == NULL)
		return false;

	char buffer[1024];
	float angle, x, y, z;
	trf = matrix3f::identity();

	while (fgets(buffer, 1024, fp)) {
		char* ptr = NULL;
		if (ptr = strstr(buffer, "rotate")) {
			ptr += strlen("rotate") + 1;

			sscanf(strstr(ptr, "angle=") + strlen("angle=") + 1, "%g", &angle);

			if (swap_xyz) {
				sscanf(strstr(ptr, "x=") + strlen("x=") + 1, "%g", &z);
				sscanf(strstr(ptr, "y=") + strlen("y=") + 1, "%g", &x);
				sscanf(strstr(ptr, "z=") + strlen("z=") + 1, "%g", &y);
			}
			else {
				sscanf(strstr(ptr, "x=") + strlen("x=") + 1, "%g", &x);
				sscanf(strstr(ptr, "y=") + strlen("y=") + 1, "%g", &y);
				sscanf(strstr(ptr, "z=") + strlen("z=") + 1, "%g", &z);
			}

			trf *= matrix3f::rotation(-vec3f(x, y, z), (angle / 180.f) * M_PI);
		}
		else if (ptr = strstr(buffer, "scale")) {
			ptr += strlen("scale") + 1;
			sscanf(strstr(ptr, "value=") + strlen("value=") + 1, "%g", &angle);

			trf *= matrix3f::scaling(angle, angle, angle);
		}
		else if (ptr = strstr(buffer, "translate")) {
			ptr += strlen("translate") + 1;

			if (swap_xyz) {
				sscanf(strstr(ptr, "x=") + strlen("x=") + 1, "%g", &z);
				sscanf(strstr(ptr, "y=") + strlen("y=") + 1, "%g", &x);
				sscanf(strstr(ptr, "z=") + strlen("z=") + 1, "%g", &y);
			}
			else {
				sscanf(strstr(ptr, "x=") + strlen("x=") + 1, "%g", &x);
				sscanf(strstr(ptr, "y=") + strlen("y=") + 1, "%g", &y);
				sscanf(strstr(ptr, "z=") + strlen("z=") + 1, "%g", &z);
			}

			shifted = vec3f(x, y, z);
		}
	}

	fclose(fp);
	return true;
}


bool readobjdir(const char* path,
	unsigned int& numVtx, unsigned int& numTri,
	tri3f*& tris, vec3f*& vtxs, double scale2, vec3f shift2, bool swap_xyz)
{
	char objfile[1024];
	char transfile[1024];

	vector<tri3f> triset;
	vector<vec3f> vtxset;
	int idxoffset = 0;

	for (int i = 0; i < 16; i++) {
		sprintf(objfile, "%sobs_%02d.obj", path, i);
		sprintf(transfile, "%s0000obs%02d.txt", path, i);

		matrix3f trf;
		vec3f shifted;

		if (false == read_transf(transfile, trf, shifted, swap_xyz))
		{
			printf("trans file %s read failed...\n", transfile);
			return false;
		}

		FILE* fp = fopen(objfile, "rt");
		if (fp == NULL) return false;

		char buf[1024];
		while (fgets(buf, 1024, fp)) {
			if (buf[0] == 'v' && buf[1] == ' ') {
				float x, y, z;
				if (swap_xyz)
					sscanf(buf + 2, "%g%g%g", &y, &z, &x);
				else
					sscanf(buf + 2, "%g%g%g", &x, &y, &z);

				vec3f pt = vec3f(x, y, z) * trf + shifted;

				vtxset.push_back(pt * scale2 + shift2);
			}
			else
				if (buf[0] == 'f' && buf[1] == ' ') {
					int id0, id1, id2, id3;
					bool quad = false;

					sscanf(buf + 2, "%d", &id0);
					char* nxt = strchr(buf + 2, ' ');
					sscanf(nxt + 1, "%d", &id1);
					nxt = strchr(nxt + 1, ' ');
					sscanf(nxt + 1, "%d", &id2);

					nxt = strchr(nxt + 1, ' ');
					if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
						if (sscanf(nxt + 1, "%d", &id3))
							quad = true;
					}

					id0--, id1--, id2--, id3--;
					triset.push_back(tri3f(id0 + idxoffset, id1 + idxoffset, id2 + idxoffset));

					if (quad)
						triset.push_back(tri3f(id0 + idxoffset, id2 + idxoffset, id3 + idxoffset));
				}
		}
		fclose(fp);

		if (triset.size() == 0 || vtxset.size() == 0)
			return false;

		idxoffset = vtxset.size();
	}

	numVtx = vtxset.size();
	vtxs = new vec3f[numVtx];
	for (unsigned int i = 0; i < numVtx; i++)
		vtxs[i] = vtxset[i];

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i = 0; i < numTri; i++)
		tris[i] = triset[i];

	return true;
}

void initObjs(char* path, int stFrame)
{
	unsigned int numVtx = 0, numTri = 0;
	vec3f* vtxs = NULL;
	tri3f* tris = NULL;

	double scale = 1.;
	vec3f shift(0.0F, 0.0F, 0.0F);

	char buff[512];

	sprintf(buff, "%s\\%04d_ob.obj", path, stFrame);
	if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false)) {
		printf("loading %s ...\n", buff);

		unsigned int numEdge = 0;
		edge4f* edges = NULL;
		//buildEdges(numTri, tris, numEdge, edges);

		lions[0] = new mesh(numVtx, numTri, numEdge, tris, edges, vtxs, NULL);
		lions[0]->updateNrms();
		printf("Read obj file don (#tri=%d, #vtx=%d, #edge=%d)\n", numTri, numVtx, numEdge);
	}
	else
		for (int idx = 0; idx < 16; idx++) {
			sprintf(buff, "%s\\obs_%02d.obj", path, idx);
			printf("loading %s ...\n", buff);

			if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false)) {
				unsigned int numEdge = 0;
				edge4f* edges = NULL;
				//buildEdges(numTri, tris, numEdge, edges);

				lions[idx] = new mesh(numVtx, numTri, numEdge, tris, edges, vtxs, NULL);
				lions[idx]->updateNrms();
				printf("Read obj file don (#tri=%d, #vtx=%d, #edge=%d)\n", numTri, numVtx, numEdge);
			}
		}
}


void initAnimation()
{
}

inline unsigned int IDX(int i, int j, int Ni, int Nj)
{
	if (i < 0 || i >= Ni) return -1;
	if (j < 0 || j >= Nj) return -1;
	return i * Nj + j;
}

void initCloths(char* path, int stFrame)
{
	unsigned int numVtx = 0, numTri = 0;
	vec3f* vtxs = NULL;
	tri3f* tris = NULL;

	double scale = 1.f;
	vec3f shift(0.0, 0.1, 0);
	// vec3f shift;

	sidx = stFrame;

	for (int idx = 0; idx < 16; idx++) {
		char buff[512];
		sprintf(buff, "%s\\%04d_%02d.obj", path, sidx, idx);
		printf("loading %s ...\n", buff);

		if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false)) {
			unsigned int numEdge = 0;
			edge4f* edges = NULL;
			//buildEdges(numTri, tris, numEdge, edges);

			cloths[idx] = new mesh(numVtx, numTri, numEdge, tris, edges, vtxs, NULL);
			g_box += cloths[idx]->bound();
			cloths[idx]->getRestLength();
			cloths[idx]->updateNrms();
			printf("Read cloth from obj file (#tri=%d, #vtx=%d, #edge=%d)\n", numTri, numVtx, numEdge);
			cout << g_box.getMin() << endl;
			cout << g_box.getMax() << endl;
			ww = g_box.width();
			hh = g_box.height();
			dd = g_box.depth();
			cout << "w =" << g_box.width() << ", h =" << g_box.height() << ", d =" << g_box.depth() << endl;
		}
	}

	sidx++;
}

void initModel(char* path, int st)
{
#ifndef VEC_CLOTH
	initCloths(path, st);
#endif

	initObjs(path, st);

#ifdef POVRAY_EXPORT
	// output POVRAY file
	{
		char pvfile[512];
		sprintf(pvfile, "c:\\temp\\pov\\cloth0000.pov");
		cloth->updateNrms(); // we need normals
		cloth->povray(pvfile, true);
	}
#endif
}

void quitModel()
{
	for (int i = 0; i < 16; i++)
		if (cloths[i])
			delete cloths[i];
	for (int i = 0; i < 16; i++)
		if (lions[i])
			delete lions[i];
}

extern void beginDraw(BOX&);
extern void endDraw();
/*
int fixed_nodes[] = {52, 145, 215, 162, 214, 47, 48, 38, 221, 190, 1, 42,  25, 34, 65, 63};
int fixed_num = 0;
*/
//int fixed_num = 0;
/*
int fixed_nodes[] = {
0, 23638, 23639, 23640, 23641, 23642, 23643, 23644,
23645, 23646, 23647, 23648, 23649, 23650, 23651,
23652, 23653, 23654, 23655, 23656, 23657, 23658,
23659, 23660, 23661, 23662, 23663, 23664, 23665,
23666, 23667, 23668,
68, 69, 70, 71, 72, 73, 74, 75,
76, 77, 78, 79, 80, 81, 82, 83,
84, 85, 86, 87, 88, 89, 90, 91,
92, 93, 94, 95, 96, 97, 98, 99};
int fixed_num = sizeof(fixed_nodes)/sizeof(int);
*/
/*
int fixed_nodes[] = {
11252, 11253, 11254, 11255, 11256, 11257, 11258, 11259, 11260,
11261, 11262, 11263, 11264, 11265, 11539, 11327,
4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560,
4561, 4562, 4563, 4564, 4654};
*/
/*
5388, 5390, 34, 34, 11910, 11910, 11911, 3955, 16165, 16164, 16139,
16140, 16140, 3945, 16144, 16142, 5146, 5388, 5388, 5387, 509, 509, 5386, 5525,
9818, 1881, 9791, 9788, 1870, 9789,
*/

//bishop
int fixed_nodes[] = {
	9681, 14539, 14538, 14537, 14536, 14535, 14534, 14533,
	14532, 14531, 14530, 14529, 14528, 14527, 14526, 14525,
	29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
	0, 4897, 4896, 4895, 4894, 4893, 4892, 4891, 4890, 4889,
	4888, 4887, 4886, 4885, 4884, 4883, 4882,
	9710, 9711, 9712, 9713, 9714, 9715, 9716,
	9717, 9718, 9719, 9720, 9721, 9722, 9723, 9724
};
//int fixed_num = sizeof(fixed_nodes)/sizeof(int);
int fixed_num = 0;

int find_nodes[200];
int find_num = 0;
//int find_nodes[] = {2509, 11702, 11688, 5289, 5303, 434, 5128, 463, 5458, 5455, 11843, 11845, 11802, 11672, 11541, 2509};
//int find_num = fixed_num;

/*
int find_nodes[] = {11537, 11537, 2564, 11538, 11538, 11538, 2563, 2563, 2563, 2563, 11685,
11685, 11685, 2562, 11850, 11850, 5464, 5462, 5462, 487, 5280, 5280, 5280, 488,
488, 488, 5122, 5122, 5122, 5122, 5123, 5123, 5123, 5307, 5307, 5310, 5310, 5413,
5411, 491, 491, 5401, 5401, 492, 492, 5315, 5315, 29, 29, 11712, 11712, 2567,
2567, 2567, 11791, 11792, 11792, 11803, 11803, 11708, 11708, 11707, 11707, 11707};
int find_num = sizeof(find_nodes)/sizeof(int);
*/

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
	if (!g_box.empty())
		beginDraw(g_box);

	if (edge)
		for (int i = 0; i < 16; i++)
			//for (int i=0; i<1; i++)
			if (cloths[i])
				cloths[i]->display(tri, false, re, level, false, cloth_set, i);

#ifndef FIXED
	for (int i = 0; i < 16; i++)
		if (pnt && lions[i])
			lions[i]->display(tri, false, false, level, true, i == 0 ? lion_set : dummy_set, i);
#endif

	if (!g_box.empty())
		endDraw();
}

void dumpModel()
{
}

void loadModel()
{
}



inline double fmax(double a, double b, double c)
{
	double t = a;
	if (b > t) t = b;
	if (c > t) t = c;
	return t;
}

inline double fmin(double a, double b, double c)
{
	double t = a;
	if (b < t) t = b;
	if (c < t) t = c;
	return t;
}

inline int project3(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;
	return 1;
}

inline int project6(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3,
	vec3f& q1, vec3f& q2, vec3f& q3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);
	double Q1 = ax.dot(q1);
	double Q2 = ax.dot(q2);
	double Q3 = ax.dot(q3);

	double mx1 = fmax(P1, P2, P3);
	double mn1 = fmin(P1, P2, P3);
	double mx2 = fmax(Q1, Q2, Q3);
	double mn2 = fmin(Q1, Q2, Q3);

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;
	return 1;
}

#include "ccd.h"

bool
vf_test(
	vec3f& p0, vec3f& p00, vec3f& q0, vec3f& q00, vec3f& q1, vec3f& q10, vec3f& q2, vec3f& q20)
{
	vec3f qi, baryc;
	double ret =
		Intersect_VF(q00, q10, q20, q0, q1, q1, p00, p0, qi, baryc);

	return ret > -0.5;
}

bool
ccd_contact(
	vec3f& p0, vec3f& p00, vec3f& p1, vec3f& p10, vec3f& p2, vec3f& p20,
	vec3f& q0, vec3f& q00, vec3f& q1, vec3f& q10, vec3f& q2, vec3f& q20)
{
	if (vf_test(p0, p00, q0, q00, q1, q10, q2, q20))
		return true;
	if (vf_test(p1, p10, q0, q00, q1, q10, q2, q20))
		return true;
	if (vf_test(p2, p20, q0, q00, q1, q10, q2, q20))
		return true;
	if (vf_test(q0, q00, p0, p00, p1, p10, p2, p20))
		return true;
	if (vf_test(q1, q10, p0, p00, p1, p10, p2, p20))
		return true;
	if (vf_test(q2, q20, p0, p00, p1, p10, p2, p20))
		return true;

	return false;
}

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles

bool
tri_contact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
	vec3f p1;
	vec3f p2 = P2 - P1;
	vec3f p3 = P3 - P1;
	vec3f q1 = Q1 - P1;
	vec3f q2 = Q2 - P1;
	vec3f q3 = Q3 - P1;

	vec3f e1 = p2 - p1;
	vec3f e2 = p3 - p2;
	vec3f e3 = p1 - p3;

	vec3f f1 = q2 - q1;
	vec3f f2 = q3 - q2;
	vec3f f3 = q1 - q3;

	vec3f n1 = e1.cross(e2);
	vec3f m1 = f1.cross(f2);

	vec3f g1 = e1.cross(n1);
	vec3f g2 = e2.cross(n1);
	vec3f g3 = e3.cross(n1);

	vec3f  h1 = f1.cross(m1);
	vec3f h2 = f2.cross(m1);
	vec3f h3 = f3.cross(m1);

	vec3f ef11 = e1.cross(f1);
	vec3f ef12 = e1.cross(f2);
	vec3f ef13 = e1.cross(f3);
	vec3f ef21 = e2.cross(f1);
	vec3f ef22 = e2.cross(f2);
	vec3f ef23 = e2.cross(f3);
	vec3f ef31 = e3.cross(f1);
	vec3f ef32 = e3.cross(f2);
	vec3f ef33 = e3.cross(f3);

	// now begin the series of tests
	if (!project3(n1, q1, q2, q3)) {
		  return false;
	}
	if (!project3(m1, -q1, p2 - q1, p3 - q1)) {
		return false;
	}

	if (!project6(ef11, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(ef12, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(ef13, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(ef21, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(ef22, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(ef23, p1, p2, p3, q1, q2, q3)) {  return false; }
	if (!project6(ef31, p1, p2, p3, q1, q2, q3)) {
		return false;
	}
	if (!project6(ef32, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(ef33, p1, p2, p3, q1, q2, q3)) {
		return false;
	}
	if (!project6(g1, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(g2, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(g3, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(h1, p1, p2, p3, q1, q2, q3)) {
	 return false;
	}
	if (!project6(h2, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6(h3, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}

	return true;
}

void checkCD(bool ccd)
{
	cloth_set.clear();
	lion_set.clear();

	TIMING_BEGIN
		if (ccd)
			printf("start ccd checking ...\n");
		else
			printf("start cd checking ...\n");


	for (int idx = 0; idx < 16; idx++) {
		mesh* lion = lions[idx];
		if (lion == NULL) continue;

		for (int k = 0; k < 16; k++) {
			mesh* cloth = cloths[k];
			if (cloth == NULL) continue;

			// Find the largest loop
			//if (idx == 0 && k == 0) {
			//	printf("lion.NbFaces: %d\ncloth.NbFace: %d\n", lion->getNbFaces(), \
			//		cloth->getNbFaces());
			//	break;
			//}

#pragma omp parallel for
			for (int i = 0; i < lion->getNbFaces(); i++)
				for (int j = 0; j < cloth->getNbFaces(); j++) {
					tri3f& a = lion->_tris[i];
					tri3f& b = cloth->_tris[j];

					if (ccd) {
						vec3f p0 = lion->_vtxs[a.id0()];
						vec3f p1 = lion->_vtxs[a.id1()];
						vec3f p2 = lion->_vtxs[a.id2()];
						vec3f q0 = cloth->_vtxs[b.id0()];
						vec3f q1 = cloth->_vtxs[b.id1()];
						vec3f q2 = cloth->_vtxs[b.id2()];
						vec3f p00 = lion->_ovtxs[a.id0()];
						vec3f p10 = lion->_ovtxs[a.id1()];
						vec3f p20 = lion->_ovtxs[a.id2()];
						vec3f q00 = cloth->_ovtxs[b.id0()];
						vec3f q10 = cloth->_ovtxs[b.id1()];
						vec3f q20 = cloth->_ovtxs[b.id2()];

						if (ccd_contact(p0, p00, p1, p10, p2, p20, q0, q00, q1, q10, q2, q20)) {
							printf("triangle ccd contact found at (%d, %d)\n", i, j);
							lion_set.insert(i);
							cloth_set.insert(j);
						}
					}
					else {
						vec3f p0 = lion->_vtxs[a.id0()];
						vec3f p1 = lion->_vtxs[a.id1()];
						vec3f p2 = lion->_vtxs[a.id2()];
						vec3f q0 = cloth->_vtxs[b.id0()];
						vec3f q1 = cloth->_vtxs[b.id1()];
						vec3f q2 = cloth->_vtxs[b.id2()];

						/*printf("%d %d %d\n", b.id0(), b.id1(), b.id2());*/

						if (tri_contact(p0, p1, p2, q0, q1, q2)) {
							printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
								i, j,
								a.id0(), a.id1(), a.id2(),
								b.id0(), b.id1(), b.id2());

							lion_set.insert(i);
							cloth_set.insert(j);
						}
					}
				}
			printf("lion Size: %d\n", lion_set.size());
			printf("cloth Size: %d\n", cloth_set.size());
		}
	}

	TIMING_END("end checking")
}

__device__ vec3f cross(vec3f &a, vec3f &b)
{
	vec3f c(0.0f);
	c.x = a.y * b.z - a.z * b.y;
	c.y = a.z * b.x - a.x * b.z;
	c.z = a.x * b.y - a.y * b.x;
	return c;
}

__device__ float dot(vec3f &a, vec3f &b)
{
	float c = 0.0f;
	/*for (int i = 0; i < 3; i++) {
		c += a.v[i] * b.v[i];
	}*/
	c += a.x * b.x;
	c += a.y * b.y;
	c += a.z * b.z;
	return c;
}

__device__ float fmax_dev(float a, float b, float c)
{
	float t = a;
	if (b > t) t = b;
	if (c > t) t = c;
	return t;
}

__device__ float fmin_dev(float a, float b, float c)
{
	float t = a;
	if (b < t) t = b;
	if (c < t) t = c;
	return t;
}

__device__ int project3_dev(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3)
{
	float P1 = dot(ax, p1);
	float P2 = dot(ax, p2);
	float P3 = dot(ax, p3);

	float mx1 = fmax_dev(P1, P2, P3);
	float mn1 = fmin_dev(P1, P2, P3);

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;
	return 1;
}

__device__ int project6_dev(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3,
	vec3f& q1, vec3f& q2, vec3f& q3)
{
	float P1 = dot(ax, p1);
	float P2 = dot(ax, p2);
	float P3 = dot(ax, p3);
	float Q1 = dot(ax, q1);
	float Q2 = dot(ax, q2);
	float Q3 = dot(ax, q3);

	float mx1 = fmax_dev(P1, P2, P3);
	float mn1 = fmin_dev(P1, P2, P3);
	float mx2 = fmax_dev(Q1, Q2, Q3);
	float mn2 = fmin_dev(Q1, Q2, Q3);

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;
	return 1;
}

__device__ vec3f vminus(vec3f& v1, vec3f& v2)
{
	vec3f v(0.0f);
	v.x = v1.x - v2.x;
	v.y = v1.y - v2.y;
	v.z = v1.z - v2.z;

	return v;
}

__device__ vec3f p2n(vec3f& v1)
{
	vec3f v(0.0f);
	v.x = -v1.x;
	v.y = -v1.y;
	v.z = -v1.z;

	return v;
}

__device__ bool tri_contact_dev(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3)
{
	vec3f p1(0.0f);
	vec3f p2 = /*P2 - P1;*/vminus(P2, P1);
	vec3f p3 = /*P3 - P1;*/vminus(P3, P1);
	vec3f q1 = /*Q1 - P1;*/vminus(Q1, P1);
	vec3f q2 = /*Q2 - P1;*/vminus(Q2, P1);
	vec3f q3 = /*Q3 - P1;*/vminus(Q3, P1);

	vec3f e1 = /*p2 - p1;*/vminus(p2, p1);
	vec3f e2 = /*p3 - p2;*/vminus(p3, p2);
	vec3f e3 = /*p1 - p3;*/vminus(p1, p3);

	vec3f f1 = /*q2 - q1;*/vminus(q2, q1);
	vec3f f2 = /*q3 - q2;*/vminus(q3, q2);
	vec3f f3 = /*q1 - q3;*/vminus(q1, q3);

	vec3f n1 = cross(e1, e2);
	vec3f m1 = cross(f1, f2);

	vec3f g1 = cross(e1, n1);
	vec3f g2 = cross(e2, n1);
	vec3f g3 = cross(e3, n1);

	vec3f h1 = cross(f1, m1);
	vec3f h2 = cross(f2, m1);
	vec3f h3 = cross(f3, m1);

	vec3f ef11 = cross(e1, f1);
	vec3f ef12 = cross(e1, f2);
	vec3f ef13 = cross(e1, f3);
	vec3f ef21 = cross(e2, f1);
	vec3f ef22 = cross(e2, f2);
	vec3f ef23 = cross(e2, f3);
	vec3f ef31 = cross(e3, f1);
	vec3f ef32 = cross(e3, f2);
	vec3f ef33 = cross(e3, f3);

	// now begin the series of tests
	if (!project3_dev(n1, q1, q2, q3)) {
		  return false;
	}
	if (!project3_dev(m1, /*-q1*/p2n(q1), /*p2 - q1*/vminus(p2, q1), /*p3 - q1*/vminus(p3, q1)
	)) {
		 return false;
		}

	if (!project6_dev(ef11, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(ef12, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(ef13, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(ef21, p1, p2, p3, q1, q2, q3)) {
		return false;
	}
	if (!project6_dev(ef22, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(ef23, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(ef31, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(ef32, p1, p2, p3, q1, q2, q3)) {
		return false;
	}
	if (!project6_dev(ef33, p1, p2, p3, q1, q2, q3)) {
		return false;
	}
	if (!project6_dev(g1, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(g2, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(g3, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(h1, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(h2, p1, p2, p3, q1, q2, q3)) {
		 return false;
	}
	if (!project6_dev(h3, p1, p2, p3, q1, q2, q3)) {
		return false;
	}

	return true;
}

__device__ uint64_t xy_to_morton(uint32_t x, uint32_t y)
{
	x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
	x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
	x = (x | (x << 2)) & 0x3333333333333333;
	x = (x | (x << 1)) & 0x5555555555555555;

	y = (y | (y << 16)) & 0x0000FFFF0000FFFF;
	y = (y | (y << 8)) & 0x00FF00FF00FF00FF;
	y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F;
	y = (y | (y << 2)) & 0x3333333333333333;
	y = (y | (y << 1)) & 0x5555555555555555;

	uint64_t d = x | (y << 1);
	return d;
}

void morton_to_xy(uint64_t m, uint32_t* x, uint32_t* y)
{
	*x = _pext_u64(m, 0x5555555555555555);
	*y = _pext_u64(m, 0xaaaaaaaaaaaaaaaa);
}

tri3f* lion_tris, * cloth_tris;
vec3f* lion_vtxs, * cloth_vtxs;
mesh* lion, * cloth;

__global__ void checkCD_dev(tri3f *lion_tris, tri3f *cloth_tris, vec3f *lion_vtxs, vec3f *cloth_vtxs, 
	int* re, int nsize, int ni, int nj)
{
	unsigned int rei = 0;

	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;

	//int ni = lion->getNbFaces_dev();
	//int nj = cloth->getNbFaces_dev();
	int stridei = blockDim.x * gridDim.x;
	int stridej = blockDim.y * gridDim.y;

	int i = i0, j = j0;

	/*printf("i0 = %d, j0 = %d, ni = %d, nj = %d, si = %d, sj = %d\n", i0, j0, ni, nj, stridei, 
		stridej);*/

	while (i < ni) {
		j = j0;
		while (j < nj) {
			/*printf("i = %d\n", i);*/
			tri3f& a = lion_tris[i];
			tri3f& b = cloth_tris[j];
			
			vec3f p0(0.0f), p1(0.0f), p2(0.0f), q0(0.0f), q1(0.0f), q2(0.0f);

			p0 = lion_vtxs[a.id0_dev()];
			p1 = lion_vtxs[a.id1_dev()];
			p2 = lion_vtxs[a.id2_dev()];
			q0 = cloth_vtxs[b.id0_dev()];
			q1 = cloth_vtxs[b.id1_dev()];
			q2 = cloth_vtxs[b.id2_dev()];

			//(2455, 6589)
			if (i == 2455 && j == 6589) {
				printf("Yahoo!\n");
				printf("Result: %d\n", tri_contact_dev(p0, p1, p2, q0, q1, q2));
				printf("======\n");
			}
			/*printf("i = %d, j = %d, stridei = %d, ni = %d\n", i, j, stridei, ni);*/

			// Start checking
			if (tri_contact_dev(p0, p1, p2, q0, q1, q2)) {
				/*printf("triangle contact found at (%d, %d)\n",
						i, j);*/
				uint64_t m = xy_to_morton(i, j);
				
				// Where to write answers?
				unsigned int index = i0 * 1000 + rei;
				re[index] = m;
				rei++;
			}

			j += stridej;
		}
		i += stridei;
	}
}

//Alloc memory on device
inline void setm(void)
{
	unsigned int lion_trin, lion_vtxn, cloth_trin, cloth_vtxn;
	lion_trin = lion->_num_tri;
	lion_vtxn = lion->_num_vtx;
	cloth_trin = cloth->_num_tri;
	cloth_vtxn = cloth->_num_vtx;

	printf("Lion: trin = %d, vtxn = %d\n", lion_trin, lion_vtxn);
	printf("Cloth: trin = %d, vtxn = %d\n", cloth_trin, cloth_vtxn);

	cudaError_t err;
	cudaMalloc((void**)&lion_tris, sizeof(tri3f) * lion_trin);
	cudaMalloc((void**)&lion_vtxs, sizeof(vec3f) * lion_vtxn);
	cudaMalloc((void**)&cloth_tris, sizeof(tri3f) * cloth_trin);
	cudaMalloc((void**)&cloth_vtxs, sizeof(vec3f) * cloth_vtxn);

	cudaMemcpy(lion_tris, lion->_tris, sizeof(tri3f) * lion_trin, cudaMemcpyHostToDevice);
	cudaMemcpy(lion_vtxs, lion->_vtxs, sizeof(vec3f) * lion_vtxn, cudaMemcpyHostToDevice);
	cudaMemcpy(cloth_tris, cloth->_tris, sizeof(tri3f) * cloth_trin, cudaMemcpyHostToDevice);
	cudaMemcpy(cloth_vtxs, cloth->_vtxs, sizeof(vec3f) * cloth_vtxn, cudaMemcpyHostToDevice);
}

inline void freem(void)
{
	cudaFree(lion_tris);
	cudaFree(lion_vtxs);
	cudaFree(cloth_tris);
	cudaFree(cloth_vtxs);
}

void checkCD_gpu(bool ccd)
{
	START_GPU
	
	cloth_set.clear();
	lion_set.clear();

		if (ccd)
			printf("start ccd checking G...\n");
		else
			printf("start cd checking G...\n");

	int cnt = 0;

	for (int idx = 0; idx < 16; idx++) {
		lion = lions[idx];
		if (lion == NULL) {
			continue;
		}

		for (int k = 0; k < 16; k++) {
			cloth = cloths[k];
			if (cloth == NULL) {
				continue;
			}
			
			int ni = lion->getNbFaces();
			int nj = cloth->getNbFaces();

			dim3 grid(64, 1);
			dim3 block(64, 1);

			int nsize = 64 * 64 * 1000;

			setm();

			int* hre = (int *) malloc(nsize * sizeof(int));
			int* dre;

			cudaError_t err;

			err = cudaMalloc((void**)&dre, nsize * sizeof(int));
			if (err != cudaSuccess) {
				printf("cudaMalloc: %s\n", cudaGetErrorString(err));
			}
			/*err = cudaMemset(dre, 0, nsize * sizeof(int));
			if (err != cudaSuccess) {
				printf("cudaMemset: %s\n", cudaGetErrorString(err));
			}*/

			for (int i = 0; i < nsize; i++) {
				hre[i] = -1;
			}

			cudaMemcpy(dre, hre, nsize * sizeof(int), cudaMemcpyHostToDevice);

			checkCD_dev << <grid, block >> > (lion_tris, cloth_tris, lion_vtxs, cloth_vtxs, 
				dre, nsize, ni, nj);

			err = cudaMemcpy(hre, dre, nsize * sizeof(int), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				printf("cudaMemcpy4: %s\n", cudaGetErrorString(err));
			}

			for (int i = 0; i < nsize; i++) {
				if (hre[i] >= 0) {
					unsigned int x, y;
					morton_to_xy(hre[i], &x, &y);

					printf("triangle contact found at (%d, %d), hre = %d\n",
						x, y, hre[i]);

					lion_set.insert(x);
					cloth_set.insert(y);
				}
			}

			cudaFree(dre);
			free(hre);
			freem();

			printf("lion Size: %d\n", lion_set.size());
			printf("cloth Size: %d\n", cloth_set.size());
		}
	}

	/*printf("cnt= %d\n", cnt);*/
	printf("End checking.\n");

	END_GPU
}

bool checkSelfIJ(int i, int j, mesh* cloth, bool ccd)
{
	tri3f& a = cloth->_tris[i];
	tri3f& b = cloth->_tris[j];

	for (int k = 0; k < 3; k++)
		for (int l = 0; l < 3; l++)
			if (a.id(k) == b.id(l))
				return false;

	if (!ccd) {
		vec3f p0 = cloth->_vtxs[a.id0()];
		vec3f p1 = cloth->_vtxs[a.id1()];
		vec3f p2 = cloth->_vtxs[a.id2()];
		vec3f q0 = cloth->_vtxs[b.id0()];
		vec3f q1 = cloth->_vtxs[b.id1()];
		vec3f q2 = cloth->_vtxs[b.id2()];

		if (tri_contact(p0, p1, p2, q0, q1, q2)) {
			printf("self contact found at (%d, %d)\n", i, j);
			return true;
		}
		else
			return false;
	}
	else {
		vec3f p0 = cloth->_vtxs[a.id0()];
		vec3f p1 = cloth->_vtxs[a.id1()];
		vec3f p2 = cloth->_vtxs[a.id2()];
		vec3f q0 = cloth->_vtxs[b.id0()];
		vec3f q1 = cloth->_vtxs[b.id1()];
		vec3f q2 = cloth->_vtxs[b.id2()];
		vec3f p00 = cloth->_ovtxs[a.id0()];
		vec3f p10 = cloth->_ovtxs[a.id1()];
		vec3f p20 = cloth->_ovtxs[a.id2()];
		vec3f q00 = cloth->_ovtxs[b.id0()];
		vec3f q10 = cloth->_ovtxs[b.id1()];
		vec3f q20 = cloth->_ovtxs[b.id2()];

		if (ccd_contact(p0, p00, p1, p10, p2, p20, q0, q00, q1, q10, q2, q20)) {
			printf("self ccd contact found at (%d, %d)\n", i, j);
			return true;
		}
		else
			return false;
	}
}

void checkSelfCD(bool ccd)
{
	cloth_set.clear();
	lion_set.clear();

	TIMING_BEGIN
		if (ccd)
			printf("start checking self-CCD...\n");
		else
			printf("start checking self-CD...\n");

	for (int k = 0; k < 16; k++) {
		mesh* cloth = cloths[k];
		if (cloth == NULL) continue;

#pragma omp parallel for
		for (int i = 0; i < cloth->getNbFaces(); i++)
			for (int j = 0; j < cloth->getNbFaces(); j++) {
				if (i >= j) continue;

				if (checkSelfIJ(i, j, cloth, ccd)) {
					cloth_set.insert(i);
					cloth_set.insert(j);
				}
			}
	}
	TIMING_END("end checking")
}

static int objIdx = 0;
static bool first = true;

bool dynamicModel(char* path, bool output, bool rev)
{
	char buff[512];
	BOX gbox;

	if (rev)
		sidx -= 2;

	for (int k = 0; k < 16; k++) {
		sprintf(buff, "%s\\%04d_%02d.obj", path, sidx, k);
		printf("loading %s ...\n", buff);

		FILE* fp = fopen(buff, "rb");
		if (fp == NULL || cloths[k] == NULL)
			continue;
		fclose(fp);

		double scale = 1.f;
		vec3f shift;
		matrix3f trf;

		memcpy(cloths[k]->getOVtxs(), cloths[k]->getVtxs(), sizeof(vec3f) * cloths[k]->getNbVertices());
		readobjfile_Vtx(buff, cloths[k]->getNbVertices(), cloths[k]->getVtxs(), scale, shift, false);
		cloths[k]->updateNrms();
	}

	sprintf(buff, "%s\\%04d_ob.obj", path, sidx);
	if (lions[0])
		memcpy(lions[0]->getOVtxs(), lions[0]->getVtxs(), sizeof(vec3f) * lions[0]->getNbVertices());

	if (lions[0] && readobjfile_Vtx(buff, lions[0]->getNbVertices(), lions[0]->getVtxs(), 1., vec3f(), false))
	{
		printf("loading %s ...\n", buff);
		lions[0]->updateNrms();
	}
	else
	{
		vec3f shift;
		matrix3f trf;

		for (int idx = 0; idx < 16; idx++) {
			if (lions[idx] == NULL) continue;

			sprintf(buff, "%s\\%04dobs%02d.txt", path, sidx, idx);
			printf("loading %s ...\n", buff);
			//readtrfile(buff, shift);
			if (read_transf(buff, trf, shift, false))
				lions[idx]->update(trf, shift);

			if (output) {
				sprintf(buff, "%s\\%04d-obj-%02d.obj", path, sidx, idx);
				lions[idx]->exportObj(buff, false, idx);
			}
		}
	}

	sidx += 1;
	return true;
}
