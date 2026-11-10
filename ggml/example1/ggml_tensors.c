// ggml tensor basics: create, inspect, build a graph, compute on CPU.
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <string.h>

static void print_shape(const char *label, const struct ggml_tensor *t) {
    printf("%s: type=%s ne=[%lld, %lld, %lld, %lld] nb=[%zu, %zu, %zu, %zu]\n",
           label, ggml_type_name(t->type),
           (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
           t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
}

int main(void) {
    // 1. A context owns the memory that tensors and the graph live in.
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,   // allocate tensor data inside the context
    };
    struct ggml_context *ctx = ggml_init(params);

    // 2. Tensors. ne[0] is the FASTEST-varying dimension (columns), ne[1] is rows.
    //    A: 4 columns x 3 rows,  B: 4 columns x 2 rows
    struct ggml_tensor *A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    struct ggml_tensor *B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);

    float a_data[12] = { 2, 8, 5, 1,
                         4, 2, 8, 6,
                         3, 2, 1, 7 };
    float b_data[8]  = { 10, 5, 9, 9,
                         5, 9, 4, 6 };
    memcpy(A->data, a_data, ggml_nbytes(A));   // data lives in the context: write it directly
    memcpy(B->data, b_data, ggml_nbytes(B));

    print_shape("A", A);
    print_shape("B", B);

    // 3. Operations only DESCRIBE work: they create new tensors recording the op.
    //    mul_mat(A, B) needs A->ne[0] == B->ne[0]; result ne = [A->ne[1], B->ne[1]] = [3, 2]
    struct ggml_tensor *C = ggml_mul_mat(ctx, A, B);
    print_shape("C", C);
    printf("C->op = %s, src0 = A? %d, src1 = B? %d\n",
           ggml_op_name(C->op), C->src[0] == A, C->src[1] == B);

    // 4. Collect the ops into a graph, then run it.
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/2);

    // 5. Read results back.
    const float *out = (const float *)C->data;
    printf("C (2 rows x 3 cols):\n");
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) printf("%8.1f", out[r * 3 + c]);
        printf("\n");
    }
    
    ggml_graph_print(gf);
    ggml_graph_dump_dot(gf, NULL, "./debug.dot");

    ggml_free(ctx);
    return 0;
}
