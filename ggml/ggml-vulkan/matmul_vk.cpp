// Matrix multiplication on the ggml Vulkan backend, checked against a CPU reference.
//
// ggml_mul_mat(a, b) with a = [K, M] and b = [K, N] (ne0 = K) gives c = [M, N].
// In plain row-major terms: A is MxK, B is NxK, and C(NxM) = B * A^T,
// i.e. C[j][i] = sum_k A[i][k] * B[j][k].

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 512;
    const int N = argc > 2 ? atoi(argv[2]) : 512;
    const int K = argc > 3 ? atoi(argv[3]) : 512;
    const int dev = argc > 4 ? atoi(argv[4]) : 0;

    // 1. Create the Vulkan backend explicitly (fails loudly if there is no device).
    const int n_dev = ggml_backend_vk_get_device_count();
    printf("Vulkan devices found: %d\n", n_dev);
    if (n_dev == 0) {
        fprintf(stderr, "No Vulkan device. Check the driver, vulkaninfo, and NVIDIA_DRIVER_CAPABILITIES.\n");
        return 1;
    }
    for (int i = 0; i < n_dev; i++) {
        char desc[256];
        ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
        printf("  [%d] %s%s\n", i, desc, i == dev ? "  <-- using" : "");
    }
    ggml_backend_t backend = ggml_backend_vk_init(dev);
    if (!backend) {
        fprintf(stderr, "ggml_backend_vk_init(%d) failed\n", dev);
        return 1;
    }
    printf("Backend: %s\n", ggml_backend_name(backend));

    // 2. Build the graph in a no_alloc context (tensor metadata only).
    ggml_init_params ip = {};
    ip.mem_size   = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b); // [M, N]

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    // 3. Allocate all tensors in Vulkan device memory.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "Failed to allocate tensors on the Vulkan device\n");
        return 1;
    }
    printf("Device buffer: %s, %.1f MiB\n", ggml_backend_buffer_name(buf),
           ggml_backend_buffer_get_size(buf) / 1048576.0);

    // 4. Upload inputs.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> ha((size_t) M * K), hb((size_t) N * K);
    for (auto & x : ha) x = dist(rng);
    for (auto & x : hb) x = dist(rng);
    ggml_backend_tensor_set(a, ha.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, hb.data(), 0, ggml_nbytes(b));

    // 5. Run on the GPU (first call includes shader/pipeline setup, so warm up).
    ggml_backend_graph_compute(backend, gf);
    const int iters = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double gflops = 2.0 * M * N * K / (ms * 1e6);
    printf("mul_mat %dx%dx%d: %.3f ms/iter, %.1f GFLOPS\n", M, N, K, ms, gflops);

    // 6. Download the result and verify against a CPU reference (sampled for large sizes).
    std::vector<float> hc((size_t) M * N);
    ggml_backend_tensor_get(c, hc.data(), 0, ggml_nbytes(c));

    double max_err = 0.0, sum_sq = 0.0;
    const int step_i = M > 128 ? M / 64 : 1;
    const int step_j = N > 128 ? N / 64 : 1;
    int checked = 0;
    for (int j = 0; j < N; j += step_j) {
        for (int i = 0; i < M; i += step_i) {
            double ref = 0.0;
            for (int k = 0; k < K; k++) {
                ref += (double) ha[(size_t) i * K + k] * (double) hb[(size_t) j * K + k];
            }
            const double err = std::fabs(ref - hc[(size_t) j * M + i]);
            if (err > max_err) max_err = err;
            sum_sq += ref * ref;
            checked++;
        }
    }
    // Vulkan mul_mat may use fp16 tensor-core math, so compare relative to the output scale.
    const double rms = std::sqrt(sum_sq / checked);
    const double rel = max_err / rms;
    const double tol = 0.02; // 2% of RMS
    const bool ok = rel <= tol;
    printf("Checked %d outputs: max abs error = %.3e, RMS(ref) = %.3f, relative = %.3e (tol %.0e) -> %sP0+r\P0+r\P0+r\P0+r\P0+r\\n",
           checked, max_err, rms, rel, tol, ok ? "PASS" : "FAIL");

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 2;
}
