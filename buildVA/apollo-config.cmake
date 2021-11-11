
# Copyright (c) 2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# This file is part of Apollo.
# OCEC-17-092
# All rights reserved.
#
# Apollo is currently developed by Chad Wood, wood67@llnl.gov, with the help
# of many collaborators.
#
# Apollo was originally created by David Beckingsale, david@llnl.gov
#
# For details, see https://github.com/LLNL/apollo.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

#=============================================================================
#
# Apollo is an online machine learning utility designed to support
# performance portability of HPC codes.
#
#=== Usage ===================================================================
#
# This file allows Apollo to be automatically detected by other libraries
# using CMake.  To build with Apollo, you can do one of two things:
#
#   1. Set the apollo_DIR environment variable to the root of the Apollo
#      installation.  If you loaded apollo through a dotkit, this may already
#      be set, and apollo will be autodetected by CMake.
#
#   2. Configure your project with this option:
#      -Dapollo_DIR=<apollo install prefix>/share/
#
# If you have done either of these things, then CMake should automatically
# find and include this file when you call find_package(apollo) from your
# CMakeLists.txt file.
#
#=== Components ==============================================================
#
# To link against these, just do, for example:
#
#   find_package(apollo REQUIRED)
#   add_executable(foo foo.c)
#   target_link_libraries(foo apollo)
#
#
if (NOT apollo_CONFIG_LOADED)
  set(apollo_CONFIG_LOADED TRUE)

  # Install layout
  set(apollo_INSTALL_PREFIX /g/g15/bolet1/workspace/apollo/buildVA/install)
  set(apollo_INCLUDE_DIR    ${apollo_INSTALL_PREFIX}/include)
  set(apollo_LIB_DIR        ${apollo_INSTALL_PREFIX}/lib)
  set(apollo_CMAKE_DIR      ${apollo_INSTALL_PREFIX}/share/cmake/apollo)

  # Includes needed to use apollo
  set(apollo_INCLUDE_PATH ${apollo_INCLUDE_DIR})
  set(apollo_LIB_PATH     ${apollo_LIB_DIR})

  # Library targets imported from file
  include(${apollo_CMAKE_DIR}/apollo.cmake)
endif()
