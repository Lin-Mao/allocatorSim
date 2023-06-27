#!/bin/bash
# This is used to replace the sanitizer_api.cpp file in the source code,
# since the cuda version is too old in the SCM compilation container.

# current directory should be "patches"!!!

mv sanitizer_api.cpp ../src/utils