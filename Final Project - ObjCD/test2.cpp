#include <iostream>

using namespace std;

void f(int **ptr)
{
    *ptr = (int*) malloc(sizeof(int) * 5);
    (*ptr)[3] = 2;
}

int main(void)
{
    int *ptr;
    f(&ptr);
    for(int i = 0; i < 5; i++){
        cout<<ptr[i]<<endl;
    }
}