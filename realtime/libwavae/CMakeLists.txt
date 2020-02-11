cmake_minimum_required(VERSION 3.0)
project(libwavae)

find_package(Torch REQUIRED)

file(GLOB SRC src/*.cpp)

add_library(wavae SHARED ${SRC})
target_link_libraries(wavae "${TORCH_LIBRARIES}")
set_property(TARGET wavae PROPERTY CXX_STANDARD 11)