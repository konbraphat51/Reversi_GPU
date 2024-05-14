#pragma once

#include "cuda_runtime.h"
#include "board.h"

extern "C" int mcGPU_move(BoardState *state, int threads);