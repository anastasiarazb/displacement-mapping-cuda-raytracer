#include "glm_outp.h"
#include <glm/gtc/type_ptr.hpp>

std::ostream& operator<< (std::ostream& os, const glm::vec4 &v)
{
    return os << "(" << v.x << "; " << v.y << "; " << v.z << "; " << v.w << ")";
}

std::ostream& operator<< (std::ostream& os, const glm::vec3 &v)
{
    return os << "(" << v.x << "; " << v.y << "; " << v.z << ")";
}

std::ostream& operator<< (std::ostream& os, const glm::mat4 &M)
{
    os << "MATRIX:\n";
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j) {
            os << M[i][j] << " ";
        }
        os << "\n";
    }
    return os;
}


float3 tofloat3(const glm::vec3 &v)
{
    return make_float3(v.x, v.y, v.z);
}

void matrCopy(float dst[], const glm::mat4 &src)
{
    size_t size = 16 * sizeof(float);
    memcpy(dst, glm::value_ptr(src), size); //В случае массива sizeof возвращает весь его размер, а не указателя
}

//---------------------------------------------


