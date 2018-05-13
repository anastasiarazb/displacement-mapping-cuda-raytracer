#include "glm_outp.h"

__host__ __device__  void printfloat3(const float3 &v)
{
    printf("(%f; %f; %f)", v.x, v.y, v.z);
}

__host__ __device__  void printfloat3e(const float3 &v)
{
    printf("(%e; %e; %e)", v.x, v.y, v.z);
}


__host__ __device__  void printfloat4(const float4 &v)
{
    printf("(%f; %f; %f; %f)", v.x, v.y, v.z, v.w);
//    printf("(%e; %e; %e; %e)", v.x, v.y, v.z, v.w);
}

__host__ __device__ void printuchar3(const uchar3 &v)
{
    printf("(%u; %u; %u)", v.x, v.y, v.z);
}

__host__ __device__  void printv3f(const glm::vec3& v)
{
    printf("(%f; %f; %f)", v.x, v.y, v.z);
}

__host__ __device__  void printv4f(const glm::vec4& v)
{
    printf("(%f; %f; %f; %f)", v.x, v.y, v.z, v.w);
}

__host__ __device__ void print4x4f(const glm::mat4 &M)
{
    printf("MATRIX:\n");
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j) {
            printf("%f ", M[i][j]);
        }
        printf("\n");
    }
}
