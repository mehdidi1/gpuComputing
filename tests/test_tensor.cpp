#include "tensor.h"

#include <cassert>
#include <iostream>
#include <vector>

static void test_constructor_4d()
{
    Tensor t(2, 3, 4, 5);
    assert(t.batch() == 2);
    assert(t.channels() == 3);
    assert(t.height() == 4);
    assert(t.width() == 5);
    assert(t.size() == 2 * 3 * 4 * 5);

    // Ensure zero-initialized
    assert(t(0, 0, 0, 0) == 0.0f);
    assert(t(1, 2, 3, 4) == 0.0f);
}

static void test_constructor_2d()
{
    Tensor t(3, 7);
    assert(t.batch() == 3);
    assert(t.channels() == 7); // For 2D tensor, channels() returns dims_[1]
    assert(t.height() == 1);
    assert(t.width() == 1);
    assert(t.size() == 3 * 7);

    // Ensure zero-initialized
    assert(t(0, 0) == 0.0f);
    assert(t(2, 6) == 0.0f);
}

static void test_indexing_4d()
{
    Tensor t(2, 3, 4, 5);
    t(1, 2, 3, 4) = 42.0f;
    assert(t(1, 2, 3, 4) == 42.0f);

    // Check that another element remains zero
    assert(t(0, 0, 0, 0) == 0.0f);
}

static void test_indexing_2d()
{
    Tensor t(2, 5);
    t(1, 4) = -3.5f;
    assert(t(1, 4) == -3.5f);

    // Check that another element remains zero
    assert(t(0, 0) == 0.0f);
}

static void test_data_constructor_copy()
{
    const int batch = 1;
    const int channels = 1;
    const int height = 2;
    const int width = 3;
    const int total = batch * channels * height * width;

    std::vector<float> src(total);
    for (int i = 0; i < total; ++i) {
        src[i] = static_cast<float>(i + 1);
    }

    Tensor t(batch, channels, height, width, src.data());
    for (int i = 0; i < total; ++i) {
        assert(t.data()[i] == src[i]);
    }

    // Mutate source and ensure tensor data is unchanged (deep copy)
    src[0] = 999.0f;
    assert(t.data()[0] != src[0]);
}

static void test_fill_and_zeros()
{
    Tensor t(2, 2, 2, 2);
    t.fill(3.14f);
    for (int i = 0; i < t.size(); ++i) {
        assert(t.data()[i] == 3.14f);
    }

    t.zeros();
    for (int i = 0; i < t.size(); ++i) {
        assert(t.data()[i] == 0.0f);
    }
}

int main()
{
    test_constructor_4d();
    test_constructor_2d();
    test_indexing_4d();
    test_indexing_2d();
    test_data_constructor_copy();
    test_fill_and_zeros();

    std::cout << "[OK] Tensor tests passed." << std::endl;
    return 0;
}
