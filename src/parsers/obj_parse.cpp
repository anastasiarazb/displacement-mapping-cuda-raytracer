#include "obj.h"
#define DEBUG
#ifdef DEBUG
#define out printf
#else
#define out(...)
#endif

#include "figures/figures.h"
#include "bounding_box/boundingbox.h"
#include "cuda_numerics/float3.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <inttypes.h>

bool readObjFile(const char *path,
                 Model  &model,
                 Light *&lights, unsigned int &num_of_lights);

//--------------------Triangles----------------------------------

void computeBB(const aiScene *scene, BoundingBox &bb)
{
    bb.initBoundingBox();
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh *mesh = scene->mMeshes[m];
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            aiVector3D p = mesh->mVertices[v];
            bb.update(p.x, p.y, p.z);
        }
    }
}

inline int countFacesNum(const aiScene* scene)
{
    int counter = 0;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh *mesh = scene->mMeshes[m];
        counter += mesh->mNumFaces;
    }
    return counter;
}

float3 make_float3(const aiVector3D &a)
{
    float3 res;
    res.x = a.x;
    res.y = a.y;
    res.z = a.z;
    return res;
}

float2 make_float2(const aiVector3D &a)
{
    float2 res;
    res.x = a.x;
    res.y = a.y;
    return res;
}

#define EPS 1e-6f

bool initTriangleByFace(Triangle *tr, const aiFace &face, const aiMesh* mesh)
{
    if (face.mNumIndices != 3) {
        out ("[initTriangleByFace]: face->mNumIndices = %u != 3\n", face.mNumIndices);
        return false;
    }

    const unsigned int *ids = face.mIndices;
    const aiVector3D   *v   = mesh->mVertices;
    const aiVector3D   *n   = mesh->mNormals;
    const aiVector3D   *uvs = mesh->mTextureCoords[0];

    tr->p0 = make_float3(v[ids[0]]);
    tr->p1 = make_float3(v[ids[1]]);
    tr->p2 = make_float3(v[ids[2]]);

#ifdef EPS
    float3 X = tr->p0/3.f + tr->p1/3.f + tr->p2/3.f; //Медиана
    tr->p0 = tr->p0 + EPS*norma(tr->p0 - X);
    tr->p1 = tr->p1 + EPS*norma(tr->p1 - X);
    tr->p2 = tr->p2 + EPS*norma(tr->p2 - X);
#endif

    tr->n0 = make_float3(n[ids[0]]);
    tr->n1 = make_float3(n[ids[1]]);
    tr->n2 = make_float3(n[ids[2]]);

    if (mesh->HasTextureCoords(0)) {
        tr->uv0 = make_float2(uvs[ids[0]]);
        tr->uv1 = make_float2(uvs[ids[1]]);
        tr->uv2 = make_float2(uvs[ids[2]]);
    }

    return true;
}

bool loadTrianglesFromScene(const aiScene* scene, Triangle *&triangles, unsigned int &num_of_triangles)
{
    if (!scene->HasMeshes()) {
        out("[loadTrianglesFromScene] : !scene->HasMeshes() \n");
        return false;
    }

    num_of_triangles = countFacesNum(scene);
    triangles        = (Triangle *)malloc(num_of_triangles * sizeof(Triangle));
//    out("num of triangles = %u\n", num_of_triangles);

    unsigned int t = 0;

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh *mesh = scene->mMeshes[m];

        if (!mesh->HasTextureCoords(0)) {
            out("[loadTrianglesFromScene] : !mesh->HasTextureCoords(0)\n");
        }

        if (!mesh->HasNormals()) {
            out("[loadTrianglesFromScene] : !mesh->HasHasNormals()\n");
            return false;
        }

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            Triangle     *tr   = triangles+t;
            const aiFace &face = mesh->mFaces[f];
            if (!initTriangleByFace(tr, face, mesh)) {
                return false;
            }
            ++t;
        }
    }

    if (num_of_triangles != t) {
        out ("[loadTrianglesFromScene]: num_of_triangles != t (%u != %u)\n", num_of_triangles, t);
    }

    return true;
}

//------------------------------------Texture-------------------------------------------------

bool loadTextureFromScene(const aiScene* scene, float4 *&texture, unsigned int &size_of_texture)
{
    if (!scene->HasTextures() || !scene->mTextures[0]) {
        out("[loadTextureFromScene]: no texture is found\n");
        return false;
    }

    const aiTexture *ai_texture = scene->mTextures[0];

    if (ai_texture->mHeight == 0) {
        out("[loadTextureFromScene]: texture format is compressed. Expected format .bmp, %s is found.\n",
            ai_texture->achFormatHint);
        return false;
    }
    for (unsigned int i = 0; i < ai_texture->mHeight * ai_texture->mWidth; ++i) {

    }
    return true;
}

//------------------------------------Lights-------------------------------------------------
float4 make_float3(const aiColor3D &a)
{
    float4 res;
    res.x = a.r;
    res.y = a.g;
    res.z = a.b;
    res.w = 1.0f;
    return res;
}

bool initLightByAi(Light *new_light, const aiLight *src)
{
    new_light->center = make_float3(src->mPosition);
    new_light->color  = make_float3(src->mColorAmbient);
    new_light->intensity = 1.0f/src->mAttenuationConstant;
    new_light->Radius = src->mSize.x;
    return true;
}

bool loadLightsFromScene(const aiScene* scene, Light *&lights, unsigned int &num_of_lights, BoundingBox bb)
{
    if (!scene->HasLights()) {
        printf("[loadLightsFromScene]: no light is found\n");
        initLights(lights, num_of_lights, bb);
        return false;
    }
    num_of_lights = scene->mNumLights;
    if (!lights) {
        free(lights);
    }
    lights = (Light *)malloc(num_of_lights * sizeof(Light));
    for (unsigned int i = 0; i < num_of_lights; ++i) {
        initLightByAi(lights + i, scene->mLights[i]);
    }
    return true;
}

//------------------------------------ALL-------------------------------------------------

bool readObjFile(const char *path,
                 Model  &model,
                 Light *&lights, unsigned int &num_of_lights)
{
    out("[readObjFile]: start\n");
    // Create an instance of the Importer class
    Assimp::Importer importer;
    // And have it read the given file with some example postprocessing
    // Usually - if speed is not the most important aspect for you - you'll
    // propably to request more postprocessing than we do in this example.
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals);
          //aiProcess_CalcTangentSpace       |  aiProcess_FlipUVs // uv.y -> 1.f-y
          //aiProcess_Triangulate            |
          //aiProcess_JoinIdenticalVertices  |
          //aiProcess_SortByPType);

    // If the import failed, report it
    if(!scene)
    {
      printf("[readObjFile] : %s\n", importer.GetErrorString());
      return false;
    }
    // Now we can access the file's contents.
    model.destroy();
    computeBB(scene, model.bb);
    loadTrianglesFromScene(scene, model.triangles, model.num_of_triangles);
//    loadTextureFromScene  (scene, model.texture,   model.num_of_texels);
    loadLightsFromScene   (scene, lights, num_of_lights, model.bb);
//    computeBB(scene, bb);
    out("[readObjFile]: success\n");
    // We're done. Everything will be cleaned up by the importer destructor
    return true;
}

// -----------------------------------------------

bool loadTrianglesFromSceneD(const aiScene* scene, Triangle *&triangles, unsigned int &num_of_triangles,
                            const Texture<float> &displaces, int subdiv_param, float shift, float scale)
{
    if (!scene->HasMeshes()) {
        out("[loadTrianglesFromScene] : !scene->HasMeshes() \n");
        return false;
    }

    num_of_triangles = countFacesNum(scene) * subdiv_param * subdiv_param;
    triangles        = (Triangle *)malloc(num_of_triangles * sizeof(Triangle));
//    out("num of triangles = %u\n", num_of_triangles);

    unsigned int t = 0;

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh *mesh = scene->mMeshes[m];

        if (!mesh->HasTextureCoords(0)) {
            out("[loadTrianglesFromScene] : !mesh->HasTextureCoords(0)\n");
        }

        if (!mesh->HasNormals()) {
            out("[loadTrianglesFromScene] : !mesh->HasHasNormals()\n");
            return false;
        }

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            Triangle triangle;
            const aiFace &face = mesh->mFaces[f];
            if (!initTriangleByFace(&triangle, face, mesh)) {
                return false;
            }

            Triangle abc; //микротреугольник
            Baricentric uva, uvb, uvc; //beta-gamma для каждой вершины abc в исходном треугольнике
            float delta = 1.0/float(subdiv_param);
        //    float3 cNormal; //нормаль к вершине с
            int i, j, k; //тройка индексов вершин текущей ячейки от 0 до subdiv_param
            for (k = 0; k < subdiv_param; ++k) {
                for (j = 0; j < subdiv_param; ++j) {
                    for (i = subdiv_param - j - k - 2; i < subdiv_param - j - k; ++i) {
                        if (i < 0) continue;
                        if (i + j + k == subdiv_param - 1) { //Нижний микротреугольник
                            uva.set( j   *delta,  k   *delta);
                            uvb.set((j+1)*delta,  k   *delta);
                            uvc.set( j   *delta, (k+1)*delta);
                        } else if (i + j + k == subdiv_param - 2) { //верхний микротреугольник
                            uva.set((j+1)*delta,  k   *delta);
                            uvb.set((j+1)*delta, (k+1)*delta);
                            uvc.set(j*delta, (k+1)*delta);
                        } else {
                            printf ("ERROR: i+j+k = %d + %d + %d != %d - 1 or %d - 1\n", i, j, k, subdiv_param, subdiv_param);
                            exit(1);
                        }
                        abc = triangle.getMicrotriangle(uva, uvb, uvc);
                        float h0 = scale*(shift + displaces.cpu_get(abc.uv0.x, abc.uv0.y));
                        float h1 = scale*(shift + displaces.cpu_get(abc.uv1.x, abc.uv1.y));
                        float h2 = scale*(shift + displaces.cpu_get(abc.uv2.x, abc.uv2.y));
                        abc.displace(h0, h1, h2);
                        abc.setDefaultNormals(); //Пересчитать нормали как перпендикуляр к плоскости треугольника
                        Triangle     *tr   = triangles+t;
                        *tr = abc;
                        ++t;
                    }
                }
            }
        }
    }

    if (num_of_triangles != t) {
        out ("[loadTrianglesFromScene]: num_of_triangles != t (%u != %u)\n", num_of_triangles, t);
    }

    return true;
}

bool readObjFileD(const char *path,
                 Model  &model,
                 Light *&lights, unsigned int &num_of_lights,
                 const Texture<float> &displaces, int subdiv_param, float shift, float scale)
{
    out("[readObjFile]: start\n");
    // Create an instance of the Importer class
    Assimp::Importer importer;
    // And have it read the given file with some example postprocessing
    // Usually - if speed is not the most important aspect for you - you'll
    // propably to request more postprocessing than we do in this example.
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals);
          //aiProcess_CalcTangentSpace       |  aiProcess_FlipUVs // uv.y -> 1.f-y
          //aiProcess_Triangulate            |
          //aiProcess_JoinIdenticalVertices  |
          //aiProcess_SortByPType);

    // If the import failed, report it
    if(!scene)
    {
      out("[readObjFile] : %s\n", importer.GetErrorString());
      return false;
    }
    // Now we can access the file's contents.
    model.destroy();
    computeBB(scene, model.bb);
    loadTrianglesFromSceneD(scene, model.triangles, model.num_of_triangles, displaces, subdiv_param, shift, scale);
//    loadTextureFromScene  (scene, model.texture,   model.num_of_texels);
    loadLightsFromScene   (scene, lights, num_of_lights, model.bb);
//    computeBB(scene, bb);
    out("[readObjFile]: success\n");
    // We're done. Everything will be cleaned up by the importer destructor
    return true;
}
