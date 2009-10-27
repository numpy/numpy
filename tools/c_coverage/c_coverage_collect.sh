#!/usr/bin/env bash
valgrind --tool=callgrind --compress-strings=no --compress-pos=no --collect-jumps=yes "$@"
