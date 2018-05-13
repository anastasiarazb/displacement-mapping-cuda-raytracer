# Displacement Mapping CUDA Raytracer

Данное приложение — курсовой проект по предмету «Компьютерная графика» на тему «Формирование рельефа поверхности методом модификации карт нормалей» (2016). Приложение представляет собой визуализатор 3D-моделей в реальном времени с возможностью наложения объемных и цветных текстур для повышения детализованности изображения. Изображение формируется методом трассировки лучей, выполняемой параллельно на GPU с использованием технологии CUDA. Сглаживание реализовано с помощью вероятностного метода (stochastic sampling). Алгоритм наложения текстур (displacement mapping) позволяет изменять рельеф поверхности модели, «приподнимая» или «опуская» базовую поверхность в каждой точке на величину, взятую из текстуры. Такой метод позволяет не только экономить память, но и накладывать разные текстуры на одну и ту же модель (примерять различные фактуры тканей на мебель или черты лица на голову персонажа), что может быть полезно, например, художникам или разработчикам игр. Примеры генерируемых изображений приведены в папке [screenshots](https://github.com/nastusha-merry/displacement-mapping-cuda-raytracer/tree/master/screenshots).

Алгоритм подробно описывается в файле «[Расчетно-пояснительная записка.pdf](https://github.com/nastusha-merry/displacement-mapping-cuda-raytracer/blob/master/%D0%A0%D0%B0%D1%81%D1%87%D0%B5%D1%82%D0%BD%D0%BE-%D0%BF%D0%BE%D1%8F%D1%81%D0%BD%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F-%D0%B7%D0%B0%D0%BF%D0%B8%D1%81%D0%BA%D0%B0.pdf)».

Системные требования (запуск):
- bumblebee (optirun + primus)

Системные требования (компиляция):
- GLFW3 и GLEW
- CUDA Toolkit 8.0
- Qt5
- CMake >= 3
- GCC >= 4.8
- графическая карта NVIDIA, поддерживающая архитектуру CUDA c compute_capability 2.0
- ОС GNU/Linux


Запуск приложения:
```bash
optirun -b primus  bin/cuda-raytracer <path_to_model(.obj)> <path_to_color_texture(.jpg .png .gif .bmp)> <path_to_displacement_map(.jpg .png .gif .bmp)> <zero_value(float point)> <displace_amount(float point)> <subdivision_parameter(integer)>
```

Пример (из папки [examples](https://github.com/nastusha-merry/displacement-mapping-cuda-raytracer/tree/master/examples)):
```bash
optirun -b primus bin/cuda models/monsterfrog/monsterfrog_mapped.obj models/monsterfrog/monsterfrog-n.bmp models/monsterfrog/monsterfrog-d.bmp 0.0 0.5 8
``` 
