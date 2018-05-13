#include "camera.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/matrix_inverse.hpp>

Camera::Camera()
{
    Pos     = glm::vec3(0.0f, 0.0f,  0.0f); //правостороння ск - ось z смотрит на нас
    Forward = glm::vec3(0.0f, 0.0f, -1.0f);
    Up      = glm::vec3(0.0f, 1.0f,  0.0f);
    Right   = glm::normalize(glm::cross(Forward, Up));

}

Camera::Camera(const glm::vec3 &Pos, const glm::vec3 &Forward, const glm::vec3 &Up)
{
    this->Pos = Pos;
    this->Forward = glm::normalize(Forward); //Правосторонняя ск - ось z смотрит на нас
    Right = glm::normalize(glm::cross(Forward, Up));
    this->Up = glm::normalize(glm::cross(Right, Forward)); //Восстанавливаем вектор вверх
}

//Перемещение - ск остается параллельна исходной
void Camera::shiftX(double delta, GLfloat x)
{
    Pos += (GLfloat)(delta * x) * glm::normalize(Right);
}

void Camera::shiftY(double delta, GLfloat y)
{
    Pos += (GLfloat)(delta * y) * glm::normalize(Up);
}

void Camera::shiftZ(double delta, GLfloat z)
{
    Pos += (GLfloat)(delta * z) * glm::normalize(Forward);
}

//Вращение

void Camera::rotateMouse(double x, double y, double dt)
{
//    if (x < 20 || Width - x < 20)
//    {
//        rotateY((mouse_x - x)/100);
//    }
//    if (y < 20 || Height - y < 20)
//    {
//        rotateX((mouse_y - y)/100);
//    }
    float angleY = dt * static_cast<float>(x) * mouse_sensitivity;
    float angleZ = dt * static_cast<float>(y) * mouse_sensitivity;
    rotateYaw(-angleY * dt);
    rotatePitch(-angleZ * dt);
//    rotateY(-angleY * dt);
//    rotateX(-angleZ * dt);
}


void Camera::rotatePitch(float degrees)
{
    Forward = glm::normalize(Forward);
    Right   = glm::normalize(Right);
    glm::mat4 rot = glm::rotate(glm::mat4(1.0), degrees, glm::normalize(Right));
    glm::mat3 rot3(rot);
    Up =  glm::normalize(rot3 * Up);
    Forward =  glm::cross(Up, Right);
//    Forward = glm::normalize(Forward);
//    glm::vec3 axis = glm::cross(Forward, Up);
//    glm::mat4 rot = glm::rotate(glm::mat4(1.0), degrees, axis);
//    glm::mat3 rot3(rot);
//    Up =  glm::normalize(rot3 * Up);
//    Forward =  glm::normalize(rot3 * Forward);
//    Right = glm::normalize(glm::cross(Forward, Up));
}

void Camera::rotateYaw(float degrees)
{
    Forward = glm::normalize(Forward);
    Up = glm::normalize(Up);
    glm::mat4 rot = glm::rotate(glm::mat4(1.0), degrees, Up);
    glm::mat3 rot3(rot);
    Forward = glm::normalize(rot3 * Forward);
    Right = glm::normalize(glm::cross(Forward, Up));
}

void Camera::rotateX(GLfloat angle)
{
    Forward = glm::rotate(Forward, glm::radians(angle), Right);
    Up      = glm::cross (Right, Forward);
}

void Camera::rotateY(GLfloat angle)
{
    Right   = glm::rotate(Right, glm::radians(angle), Up);
    Forward = glm::cross (Up, Right);
}
void Camera::rotateZ(GLfloat angle)
{
    Up    = glm::rotate(Up, glm::radians(angle), Forward);
    Right = glm::cross (Forward, Up);
}

std::ostream &operator<< (std::ostream &out, const Camera &cam) {
    return out << "CAMERA:\n"
               "Pos     = (" << cam.Pos.x << "; " << cam.Pos.y << "; "<< cam.Pos.z << ")\n" <<
               "Forward = (" << cam.Forward.x << "; " << cam.Forward.y << "; "<< cam.Forward.z << ")\n" <<
               "Up      = (" << cam.Up.x << "; " << cam.Up.y << "; "<< cam.Up.z << ")\n" <<
               "Right   = (" << cam.Right.x << "; " << cam.Right.y << "; "<< cam.Right.z << ")\n";
}


void Camera::glSetLookAt()
{
    glMatrixMode(GL_MODELVIEW);

    glLoadIdentity();
    glm::mat4 look_at_matrix = glm::lookAt(Pos, Center(), Up);
    glLoadMatrixf(glm::value_ptr(look_at_matrix));
}

glm::mat4 Camera::lookAt() const
{
    glm::mat4 M = glm::mat4(1);

    glm::vec3 F = glm::normalize(Forward);
    glm::vec3 R = glm::normalize(glm::cross(Forward, Up));
    glm::vec3 U = glm::normalize(glm::cross(R, Forward));

    //По сути умножение на матрицу перехода от базиса мировых координат к видовым + сдвиг на вектор Pos
    M[0] = glm::vec4( R, 0);
    M[1] = glm::vec4( U, 0);
    M[2] = glm::vec4(-F, 0);
    M[3] = glm::vec4(Pos, 1);

    return M; //При таком методе шаг нужен 0,2
//    M[0] = glm::vec4(R.x, U.x, -F.x, Pos.x);
//    M[1] = glm::vec4(R.y, U.y, -F.y, Pos.y);
//    M[2] = glm::vec4(R.z, U.z, -F.z, Pos.z);
//    M[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

//    return glm::transpose(M); //При таком методе шаг нужен 0,2

//    return glm::lookAt(Pos, Center(), Up);
}


void Camera::setViewPort(GLint Width, GLint Height)
{
    this->Width = Width;
    this->Height = Height;
    mouse_x = Width/2;
    mouse_y = Height/2;
}
