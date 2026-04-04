@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cd /d "C:\Users\famil\Desktop\fastenhancer"
cl.exe /O2 /TC /D_CRT_SECURE_NO_WARNINGS /IC:\Users\famil\Desktop\fastenhancer\tests\engine\unity /IC:\Users\famil\Desktop\fastenhancer\src\engine\common /IC:\Users\famil\Desktop\fastenhancer\src\engine /IC:\Users\famil\Desktop\fastenhancer\src\engine\configs /FoC:\Users\famil\Desktop\fastenhancer\obj_temp\ /c C:\Users\famil\Desktop\fastenhancer\tests\engine\unity\unity.c C:\Users\famil\Desktop\fastenhancer\tests\engine\test_fft.c C:\Users\famil\Desktop\fastenhancer\src\engine\common\fft.c 2>&1
