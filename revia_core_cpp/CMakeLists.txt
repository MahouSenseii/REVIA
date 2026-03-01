cmake_minimum_required(VERSION 3.20)
project(revia_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

set(USE_TLS OFF CACHE BOOL "" FORCE)
set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG        v11.4.5
)
FetchContent_MakeAvailable(ixwebsocket)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(revia_core ${SOURCES})

target_include_directories(revia_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(revia_core PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ixwebsocket
)

if(WIN32)
    target_link_libraries(revia_core PRIVATE ws2_32 shlwapi)
    target_compile_definitions(revia_core PRIVATE _WIN32_WINNT=0x0A00 NOMINMAX)
endif()
