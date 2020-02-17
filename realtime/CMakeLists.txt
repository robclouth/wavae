cmake_minimum_required(VERSION 3.0)

project(WAVAE)

add_subdirectory("./libwavae")

add_library(wavae_encoder SHARED src/wavae_encoder.cpp)
set_target_properties(wavae_encoder PROPERTIES PREFIX "" SUFFIX "~.pd_linux")

add_library(wavae_decoder SHARED src/wavae_decoder.cpp)
set_target_properties(wavae_decoder PROPERTIES PREFIX "" SUFFIX "~.pd_linux")
