cmake_minimum_required (VERSION 2.8)

# projectname is the same as the main-executable
project(xflow)

add_definitions('-g')
add_definitions('-Wall')
add_definitions('-std=c++11')

include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/ps-lite/include"
  "${PROJECT_SOURCE_DIR}/ps-lite/deps/include"
  "${PROJECT_BINARY_DIR}"
)
message(${PROJECT_SOURCE_DIR})
message(${PROJECT_BINARY_DIR})

link_directories(
  "${PROJECT_BINARY_DIR}"
  "${PROJECT_SOURCE_DIR}/ps-lite"
  "${PROJECT_SOURCE_DIR}/ps-lite/deps/lib"
  "${PROJECT_SOURCE_DIR}/ps-lite/build"
)

#add_subdirectory(ps-lite)
add_subdirectory(src/io)
add_subdirectory(src/model)
#add_subdirectory(src/c_api)
