#include "model.h"
#include "parsers/obj.h"

void Model::initModel()
{
    bb.initBoundingBox();
    triangles = nullptr;
    texture   = nullptr;
    num_of_triangles = 0;
    num_of_texels    = 0;
}

Model new_model()
{
    Model M;
    M.initModel();
    return M;
}

void Model::destroy()
{
    if (triangles != nullptr) {
        free(triangles);
        triangles = nullptr;
        printf("[Model::destroy]: free triangles\n");
    }
    if (texture) {
        free(texture);
        texture = nullptr;
        printf("[Model::destroy]: free textures\n");
    }
    initModel();
    printf("[Model::destroy]:done\n");
}

//BoundingBox Model::computeBoundingBox()
//{
//    bb.initBoundingBox();
//    for (int i = 0; i < num_of_triangles; ++i) {
//        bb.update(triangles[i].p0);
//        bb.update(triangles[i].p1);
//        bb.update(triangles[i].p2);
//    }
//}
