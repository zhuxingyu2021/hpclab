#include <time.h>
#ifdef _WIN64
#pragma comment(lib, "winmm.lib ")
#include <Windows.h>
#else
#include <sys/time.h>
#endif

double get_wall_time(){
#ifdef _WIN64
    DWORD msec = timeGetTime();
    return msec * 0.001;
#else
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}