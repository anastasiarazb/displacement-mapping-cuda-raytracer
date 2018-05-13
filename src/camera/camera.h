#ifndef CAMERA_H
#define CAMERA_H

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
//using namespace glm;

//Почти стандартная камера
//Стандартная имеет вектор Target,
struct Camera{
    glm::vec3 Pos;  //Текущая точка (радиус-вектор)
    glm::vec3 Forward; //Направление вперед (свободный нормированный вектор)
    glm::vec3 Right; //Направление вправо (свободный нормированный вектор)
//    vec3 Target; //Куда камера смотрит (координаты центра кадра, радиус-вектор) = Pos+Forward
    glm::vec3 Up; //Вектор, задающий направление вверх (свободный нормированный вектор)

    float step = 10;
    float delta_angle = 1; //В градусах!
    float mouse_sensitivity = 2;//1; // чувствительность мыши
    //Viewport
    GLint Width, Height;
    double mouse_x, mouse_y; //Мышь
    bool ancor_set = false; //Установлена ли точка, от которой считать сдвиг мыши

    Camera();
    Camera(const glm::vec3 &Pos, const glm::vec3 &Forward, const glm::vec3 &Up);

    glm::vec3 Center() const {return Pos + Forward;} //Точка, в которую смотрит камера (радиус-вектор), нужен в gluLookUp

    void setViewPort(GLint Width, GLint Height);

    //Сдвиг камеры по осям, ск остается параллельна исходной
    void shiftX(double delta, GLfloat x);
    inline void shiftLeft(double delta) {shiftX(delta, -step);}
    inline void shiftRight(double delta){shiftX(delta, step);}

    void shiftY(double delta, GLfloat y);
    inline void shiftUp(double delta)  {shiftY(delta, step);}
    inline void shiftDown(double delta){shiftY(delta,-step);}

    void rotateMouse(double x, double y, double dt);
    void rotatePitch(float degrees);
    void shiftZ(double delta, GLfloat z);
    inline void shiftForward(double delta)   {shiftZ(delta,  step);} //Левосторонняя ск
    inline void shiftBackwards(double delta) {shiftZ(delta, -step);}
    //Подниятие-опускание взгляда
    void rotateX(GLfloat angle); //Поворот вокруг продольной оси
    inline void rotateUp()   {rotateX( delta_angle);}
    inline void rotateDown() {rotateX(-delta_angle);}
    //Повороты влево-вправо
    void rotateY(GLfloat angle);
    inline void rotateLeft()  {rotateY( delta_angle);}
    inline void rotateRight() {rotateY(-delta_angle);}
    //Наклоны головы вправо-влево (ось z направлена из камеры)
    void rotateZ(GLfloat angle);
    inline void tiltRight() {rotateZ( delta_angle);}
    inline void tiltLeft()  {rotateZ(-delta_angle);}
    void glSetLookAt();
    void rotateYaw(float degrees);
    glm::mat4 lookAt() const;
};

//extern std::ostream &operator<< (std::ostream &out, const glm::vec3 &vec);
std::ostream &operator<< (std::ostream &out, const Camera &cam);

#endif // CAMERA_H
