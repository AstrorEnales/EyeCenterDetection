cmake_minimum_required(VERSION 2.8)
add_compile_options(-std=gnu++11)
project( EyeCenter )
find_package( OpenCV REQUIRED )
include("$ENV{DLIB_ROOT}/lib/cmake/dlib/dlib.cmake")
add_executable( EyeCenter Main.cpp Utils.cpp Gradient.cpp EyeCenterNaive.cpp EyeCenterAscend.cpp EyeCenterAscendFit.cpp EyeCenterEvolAlg.cpp EyeCenterAscendPaul.cpp face_landmark_detection.cpp )
target_link_libraries( EyeCenter ${OpenCV_LIBS} dlib::dlib )
