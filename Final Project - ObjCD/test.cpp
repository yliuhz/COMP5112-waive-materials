#include <immintrin.h>
#include <stdint.h>
#include <iostream>

using namespace std;

// on GCC, compile with option -mbmi2, requires Haswell or better.

uint64_t xy_to_morton(uint32_t x, uint32_t y)
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

int main(void)
{
    unsigned int x = 10, y = 14, xx = 0, yy = 0;
    uint64_t morton = xy_to_morton(x, y);
    cout<<morton<<endl;
    morton_to_xy(morton, &xx, &yy);
    cout<<xx<<yy<<endl;
}