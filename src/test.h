#pragma once

//#define TEST
//#define SINGLE_KERNEL //рисовать только 1 поток и выводить все printf
//#define DIFFERENCE //чтобы увидеть разницу между тем, что должно быть, и тем, что есть
//#define SMOOTH
//#define SIMPLE_DISPLACE
//#define NO_DISPLACE
//#define PROCEDURE_DISPLACE

//#define NORMAL_AS_COLOR

#define COORDS(M) (M).x, (M).y, (M).z

#define SUBDIV_PARAMETER 128
//#define MAX_HEIGHT 0.03f

#define AMBIENT 0.6f
#define DIFF 0.5f
//#define ATTENUATION 25.f //для monsterfrog
#define ATTENUATION 1.0f //для куба

