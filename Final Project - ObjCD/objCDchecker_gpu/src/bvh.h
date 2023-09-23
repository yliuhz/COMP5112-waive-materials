#pragma once

#include "aabb.h"
#include "cmesh.cuh"
#include "tri.cuh"
#include <vector>

using namespace std;

class aap
{
public:
	char _xyz;
	double _p;

	aap(const BOX& total)
	{
		vec3f center = total.center();
		char xyz = 2;
		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		}
		else
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

		_xyz = xyz;
		_p = center[xyz];
	}

	bool inside(const vec3f& mid) const
	{
		return mid[_xyz] > _p;
	}
};

class bvhnode
{
public:
	tri3f _item;
	vector<vec3f> vtxs; // All xyz coords
	bvhnode* _parent;
	bvhnode* _left;
	bvhnode* _right;
	BOX _box;

	void Construct(bvhnode* p, tri3f& pt)
	{
		_parent = p;
		_left = _right = nullptr;
		_item = pt;
		// 
	}

	bvhnode()
	{
		_parent = _left = _right = nullptr;
	}

	bvhnode(bvhnode* p, tri3f& pt)
	{
		Construct(p, pt);
	}

	bvhnode(bvhnode* p, vector<tri3f>& pts)
	{
		if (pts.size() == 1) {
			Construct(p, pts[0]);
			return;
		}

		_parent = p;
		int num = pts.size();

		for (int i = 0; i < num; i++) {
			vec3f& p1 = vtxs[pts[i].id0()];
			vec3f& p2 = vtxs[pts[i].id1()];
			vec3f& p3 = vtxs[pts[i].id2()];
			_box += p1;
			_box += p2;
			_box += p3;
		}

		if (num == 2) {
			_left = new bvhnode(this, pts[0]);
			_right = new bvhnode(this, pts[1]);
			return;
		}

		aap pln(_box);
		vector<tri3f> left, right;

		for (int i = 0; i < num; i++) {
			tri3f& pt = pts[i];
			vec3f& p1 = vtxs[pts[i].id0()];
			vec3f& p2 = vtxs[pts[i].id1()];
			vec3f& p3 = vtxs[pts[i].id2()];
			if (pln.inside(p1) || pln.inside(p2) || pln.inside(p3)) {
				left.push_back(pt);
			}
			else {
				right.push_back(pt);
			}
		}

		if (left.size() == 0)
		{
			left = std::vector<tri3f>(
				std::make_move_iterator(right.begin() + right.size() / 2),
				std::make_move_iterator(right.end()));
			right.erase(right.begin() + right.size() / 2, right.end());
		}
		else if (right.size() == 0) {
			right = std::vector<tri3f>(
				std::make_move_iterator(left.begin() + left.size() / 2),
				std::make_move_iterator(left.end()));
			left.erase(left.begin() + left.size() / 2, left.end());
		}

		_left = new bvhnode(this, left);
		_right = new bvhnode(this, right);
	}
};


class bvh
{
public:
	bvhnode* root = nullptr;

	bvh() = default;

	bvhnode* construct(mesh *m, bvhnode* root)
	{

	}
};