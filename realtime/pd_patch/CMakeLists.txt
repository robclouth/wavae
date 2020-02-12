cmake_minimum_required(VERSION 3.0)

project(WAVAE)

file(GLOB SRC src/*.cpp)

add_library(wavae_encoder SHARED ${SRC})
set_target_properties(wavae_encoder PROPERTIES PREFIX "" SUFFIX "~.pd_linux")
