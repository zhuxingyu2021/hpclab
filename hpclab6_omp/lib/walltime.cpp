#include <time.h>
#ifdef WIN64
#include <Windows.h>
#else
#include <sys/time.h>
#endif

double get_wall_time(){
#ifdef WIN64
    return (double)GetTickCount() * .001;
#else
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}