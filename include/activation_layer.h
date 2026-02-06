#ifndef ACTIVATION_LAYERS_H
#define ACTIVATION_LAYERS_H
#include "tensor.h"

// =========================
// Activation Layers
// =========================

class ReLULayer{
public:
    ReLULayer();

    void forward(const Tensor& input, Tensor& output) override;
};

class SigmoidLayer{
public:
    SigmoidLayer();

    void forward(const Tensor& input, Tensor& output) override;

private:
    // Paramètres éventuels (ex: clamp) si tu veux
};

class TanhLayer{
public:
    TanhLayer();

    void forward(const Tensor& input, Tensor& output) override;
};

class SoftmaxLayer{
public:
    SoftmaxLayer();

    void forward(const Tensor& input, Tensor& output) override;

};

#endif