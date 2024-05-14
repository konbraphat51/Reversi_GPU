#pragma once

#include "cuda_runtime.h"
#include "board.h"

int mcGPU_move(BoardState *state, int threads);