project(HW8_P1)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 11)
aux_source_directory(. SRC_LIST)
#add_executable(${PROJECT_NAME} ${SRC_LIST})

find_package( OpenCV 3 REQUIRED)


set(project_sources
    main.cpp
    )

add_executable(${PROJECT_NAME} ${project_sources})

include_directories(${OpenCV_INCLUDE_DIRS})


#add_executable( task_1 task_1.cpp )
target_link_libraries(HW8_P1 ${OpenCV_LIBS} )
