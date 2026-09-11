cmake_minimum_required(VERSION 3.20)
project(Obsidianalys VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Frontend bäddas in om den är byggd (cd web && npm run build)
set(WEB_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/core/generated/web_assets.cpp)
if(NOT EXISTS ${WEB_ASSETS})
  message(WARNING "Frontend är inte byggd – binären serverar bara via --web DIR. Kör 'npm install && npm run build' i web/.")
  set(WEB_ASSETS ${CMAKE_CURRENT_SOURCE_DIR}/core/src/web_assets_stub.cpp)
endif()

add_executable(obsidianalys
  core/src/main.cpp
  core/src/analysis.cpp
  core/src/text.cpp
  core/src/report.cpp
  core/src/platform.cpp
  core/src/server.cpp
  ${WEB_ASSETS}
)
target_include_directories(obsidianalys PRIVATE core/src)
target_include_directories(obsidianalys SYSTEM PRIVATE core/third_party)

find_package(Threads REQUIRED)
target_link_libraries(obsidianalys PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(obsidianalys PRIVATE /utf-8 /W3 /permissive-)
else()
  target_compile_options(obsidianalys PRIVATE -Wall -Wextra)
endif()

if(WIN32)
  option(OBSIDIANALYS_WINDOWED "Bygg utan konsolfönster" ON)
  set(OBSIDIANALYS_ICON ${CMAKE_CURRENT_SOURCE_DIR}/obkol.ico)
  configure_file(core/app.rc.in ${CMAKE_CURRENT_BINARY_DIR}/app.rc @ONLY)
  target_sources(obsidianalys PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/app.rc)
  target_compile_definitions(obsidianalys PRIVATE _WIN32_WINNT=0x0A00 WIN32_LEAN_AND_MEAN NOMINMAX)
  target_link_libraries(obsidianalys PRIVATE ws2_32 shell32)
  if(OBSIDIANALYS_WINDOWED)
    set_target_properties(obsidianalys PROPERTIES WIN32_EXECUTABLE ON)
    if(MSVC)
      target_link_options(obsidianalys PRIVATE /ENTRY:mainCRTStartup)
    endif()
  endif()
  if(MINGW)
    target_link_options(obsidianalys PRIVATE -static)
  endif()
endif()
