### setup project ###
cmake_minimum_required(VERSION 3.17.3) # 3.17 > for Python3_SOABI
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(fibby
  VERSION 1.0
  DESCRIPTION "FIB module"
  LANGUAGES C Fortran
  )

# Safety net
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.\n"
  )
endif()

# Grab Python
find_package(Python3 3.9 REQUIRED
  COMPONENTS Interpreter Development NumPy)

# Grab the variables from a local Python installation
# F2PY headers
execute_process(
  COMMAND "${Python3_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Project scope; consider using target_include_directories instead
include_directories(
  BEFORE
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
  ${F2PY_INCLUDE_DIR}
  )

message(STATUS ${Python3_INCLUDE_DIRS})
message(STATUS ${F2PY_INCLUDE_DIR})
message(STATUS ${Python3_NumPy_INCLUDE_DIRS})

# Vars
set(f2py_module_name "fibby")
set(fortran_src_file "${CMAKE_SOURCE_DIR}/fib1.f")
set(f2py_module_c "${f2py_module_name}module.c")
set(generated_module_file "${f2py_module_name}${Python3_SOABI}")

# Generate sources
add_custom_target(
  genpyf
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
  )
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
  COMMAND ${Python3_EXECUTABLE}  -m "numpy.f2py"
                   "${fortran_src_file}"
                   -m "fibby"
                   --lower # Important
  DEPENDS fib1.f # Fortran source
  )

# Set up target
add_library(${CMAKE_PROJECT_NAME} SHARED
  "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}" # Generated
  "${F2PY_INCLUDE_DIR}/fortranobject.c" # From NumPy
  "${fortran_src_file}" # Fortran source(s)
  )

# Depend on sources
add_dependencies(${CMAKE_PROJECT_NAME} genpyf)

set_target_properties(
     ${CMAKE_PROJECT_NAME}
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${CMAKE_PROJECT_NAME}"
        LINKER_LANGUAGE C
    )
