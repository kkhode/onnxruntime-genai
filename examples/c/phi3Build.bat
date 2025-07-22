echo "------------ Script initiatialization ------------"
set BUILD_FOLDER=C:\onnxruntime-genai-dev-tools\build_tools\BuildArtifacts-20250624_145252
set OVEP_INSTALL=%BUILD_FOLDER%\onnxruntime-install
set GENAIINTERFACES_LOC=%BUILD_FOLDER%\genai-interfaces

echo "------------ Syncing example headers to ORT GenAI src ------------"
cd %BUILD_FOLDER%\onnxruntime-genai
copy /y examples\c\include\ort_genai.h src\ort_genai.h
copy /y examples\c\include\ort_genai_c.h src\ort_genai_c.h

echo "------------ Syncing GenAI headers to ORT GenAI src & example ------------"
cd %BUILD_FOLDER%\onnxruntime-genai
del /s /q .\src\pipelines\
mkdir .\src\pipelines\
copy %GENAIINTERFACES_LOC%\include\pipelines\* .\src\pipelines\
del /s /q .\examples\c\include\pipelines\
mkdir .\examples\c\include\pipelines\
copy %GENAIINTERFACES_LOC%\include\pipelines\* .\examples\c\include\pipelines\

echo "------------ Building onnxruntime-genai ------------"
cd %BUILD_FOLDER%\onnxruntime-genai
python build.py --config RelWithDebInfo --parallel --skip_tests --ort_home %OVEP_INSTALL% || exit /b 1

echo "------------ Building onnxruntime-genai Phi3 C samples ------------"
copy /y %OVEP_INSTALL%\include\* examples\c\include
copy /y %OVEP_INSTALL%\lib\* examples\c\lib
copy /y build\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime-genai.lib examples\c\lib\.
copy /y build\Windows\RelWithDebInfo\RelWithDebInfo\onnxruntime-genai.dll examples\c\lib\.

cd examples/c
mkdir build
cd build
cmake -DPHI3=ON -DPHI3_QA=ON ..
cmake --build . --config RelWithDebInfo
