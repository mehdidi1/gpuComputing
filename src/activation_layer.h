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
    Shape get_output_shape(const Shape& input_shape) const override;
};

class SigmoidLayer{
public:
    SigmoidLayer();

    void forward(const Tensor& input, Tensor& output) override;
    Shape get_output_shape(const Shape& input_shape) const override;

private:
    // Paramètres éventuels (ex: clamp) si tu veux
};

class TanhLayer{
public:
    TanhLayer();

    void forward(const Tensor& input, Tensor& output) override;
    Shape get_output_shape(const Shape& input_shape) const override;
};

class SoftmaxLayer{
public:
    SoftmaxLayer();

    void forward(const Tensor& input, Tensor& output) override;
    Shape get_output_shape(const Shape& input_shape) const override;

private:
    // Softmax: généralement sur la dernière dimension
    // si tu veux, tu peux ajouter axis_ (par défaut -1)
    int axis_ = -1;
};

#endif