cmake_minimum_required(VERSION 3.16)
project(llm_bin CXX)

set(LLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    ${LLM_ROOT}/include
    ${LLM_ROOT}/include/common
)

add_library(hiai_llm_engine SHARED IMPORTED)
set_target_properties(hiai_llm_engine PROPERTIES
    IMPORTED_LOCATION ${LLM_ROOT}/lib64/libhiai_llm_engine.so)

add_library(neural_network_runtime SHARED IMPORTED)
set_target_properties(neural_network_runtime PROPERTIES
    IMPORTED_LOCATION ${LLM_ROOT}/lib64/libneural_network_runtime.so)

add_executable(llm_bin llm_bin.cpp)
target_link_libraries(llm_bin PRIVATE
    hiai_llm_engine
    neural_network_runtime
)

add_executable(llm_eval llm_eval.cpp)
target_link_libraries(llm_eval PRIVATE
    hiai_llm_engine
    neural_network_runtime
)

# Embed rpath so binaries find libs next to deploy layout (lib64/ sibling)
set(_llm_rpath "\$ORIGIN/../lib64:\$ORIGIN/lib64")
set_target_properties(llm_bin llm_eval PROPERTIES
    BUILD_RPATH "${_llm_rpath}"
    INSTALL_RPATH "${_llm_rpath}"
)

install(TARGETS llm_bin llm_eval RUNTIME DESTINATION bin)
