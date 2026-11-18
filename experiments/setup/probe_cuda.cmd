@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64 >nul
if errorlevel 1 exit /b 1
set "THESIS_CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0"
set "THESIS_CUDNN_ROOT=%THESIS_CUDA_ROOT%\cuda"
set "PATH=%THESIS_CUDNN_ROOT%\bin;%THESIS_CUDA_ROOT%\bin;%PATH%"
"%THESIS_CUDA_ROOT%\bin\nvcc.exe" -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -I"%THESIS_CUDNN_ROOT%\include" "%~dp0probe_cuda.cu" -L"%THESIS_CUDNN_ROOT%\lib\x64" -lcudnn -lcublas -o "%~dp0probe_cuda.exe"
if errorlevel 1 exit /b 1
"%~dp0probe_cuda.exe"
exit /b %errorlevel%
