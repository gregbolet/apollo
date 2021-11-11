
// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// This file is part of Apollo.
// OCEC-17-092
// All rights reserved.
//
// Apollo is currently developed by Chad Wood, wood67@llnl.gov, with the help
// of many collaborators.
//
// Apollo was originally created by David Beckingsale, david@llnl.gov
//
// For details, see https://github.com/LLNL/apollo.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.


#ifndef APOLLO_CONFIG_H
#define APOLLO_CONFIG_H


// Version information -- numerical and a version string
#define APOLLO_MAJOR_VERSION 2
#define APOLLO_MINOR_VERSION 0
#define APOLLO_PATCH_VERSION 0
#define APOLLO_VERSION         "2.0.0"
#define APOLLO_GIT_SHA1        "3e378cbca3d1dc759c3bdd4496a58fdd02d76a18"

#define APOLLO_BUILD_TYPE      "Release"

#define APOLLO_HOST_NODE_NAME  "quartz1916"
#define APOLLO_HOST_KNOWN_AS   "quartz1916"
#define APOLLO_HOST_DETAILED   "GNU/Linux 3.10.0-1160.36.2.1chaos.ch6.x86_64 x86_64"

#define APOLLO_CXX_COMPILER    "/usr/tce/packages/clang/clang-12.0.0/bin/clang++ "
#define APOLLO_CXX_FLAGS       " -Wno-unused-variable -fPIC  "
#define APOLLO_LINK_FLAGS      "  "




#endif // APOLLO_CONFIG_H
