#include "flatten_layer.h"

// =========================
// Forward
// =========================

void FlattenLayer::forward(const Tensor& input, Tensor& output)
{
   /*if (input.ndims() != 4) {
        throw std::runtime_error("FlattenLayer::forward expects 4D input [B,C,H,W]");
    }*/

    const int B = input.batch();
    const int C = input.channels();
    const int H = input.height();
    const int W = input.width();

    const int flattened_size = C * H * W;

    // Vérifier que output est bien [B, C*H*W]
    /*if (output.ndims() != 2 ||
        output.batch() != B ||
        output.channels() != flattened_size) {
        throw std::runtime_error("FlattenLayer::forward: output shape mismatch");
    }*/

    // Copie en respectant le stockage row-major
    for (int b = 0; b < B; ++b) {
        int index = 0;

        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    output(b, index++) = input(b, c, h, w);
                }
            }
        }
    }
}

