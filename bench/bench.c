/*
 * RV-Sparse benchmark.
 *
 * Runs five candidate sparse_multiply implementations on a sweep of
 * matrix sizes and densities, verifies they all agree, and prints a
 * CSV record per (method, size, density) configuration to stdout.
 *
 *   gcc -O3 -march=native -lm -o bench/run_bench bench/bench.c
 *   ./bench/run_bench > results/timings.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ---------------------------------------------------------------- */
/* Five candidate implementations                                    */
/* ---------------------------------------------------------------- */

typedef void (*spmul_fn)(int rows, int cols,
                         const double* A, const double* x,
                         int* out_nnz,
                         double* values, int* col_indices, int* row_ptrs,
                         double* y);

/*
 * M1 - Dense direct.
 * Skip CSR entirely and compute y = A*x in place. Violates the challenge
 * (no CSR is built) but serves as a theoretical floor for memory-bound
 * dense traffic.
 */
static void m1_dense_direct(int rows, int cols,
                            const double* A, const double* x,
                            int* out_nnz,
                            double* values, int* col_indices, int* row_ptrs,
                            double* y) {
    (void) values; (void) col_indices; (void) row_ptrs;
    *out_nnz = 0;
    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        const double* row = A + (size_t) i * cols;
        for (int j = 0; j < cols; ++j) s += row[j] * x[j];
        y[i] = s;
    }
}

/*
 * M2 - Two-phase naive.
 * Phase 1 builds CSR with a data-dependent branch on v != 0.
 * Phase 2 walks CSR for the SpMV. No restrict qualifiers.
 */
static void m2_two_phase_naive(int rows, int cols,
                               const double* A, const double* x,
                               int* out_nnz,
                               double* values, int* col_indices, int* row_ptrs,
                               double* y) {
    int nnz = 0;
    for (int i = 0; i < rows; ++i) {
        row_ptrs[i] = nnz;
        for (int j = 0; j < cols; ++j) {
            double v = A[(size_t) i * cols + j];
            if (v != 0.0) {
                values[nnz]      = v;
                col_indices[nnz] = j;
                ++nnz;
            }
        }
    }
    row_ptrs[rows] = nnz;
    *out_nnz = nnz;

    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k)
            s += values[k] * x[col_indices[k]];
        y[i] = s;
    }
}

/*
 * M3 - Two-phase branchless (current submission).
 * Predicated store: always write, advance the cursor with a cmov.
 * restrict-qualified aliases let -O3 SIMDize the dot product.
 */
static void m3_two_phase_branchless(int rows, int cols,
                                    const double* A, const double* x,
                                    int* out_nnz,
                                    double* values, int* col_indices, int* row_ptrs,
                                    double* y) {
    const double* __restrict__ pA = A;
    const double* __restrict__ px = x;
    double*       __restrict__ pv = values;
    int*          __restrict__ pc = col_indices;
    int*          __restrict__ pr = row_ptrs;
    double*       __restrict__ py = y;

    int nnz = 0;
    for (int i = 0; i < rows; ++i) {
        pr[i] = nnz;
        const double* row = pA + (size_t) i * cols;
        for (int j = 0; j < cols; ++j) {
            double v = row[j];
            pv[nnz] = v;
            pc[nnz] = j;
            nnz += (v != 0.0);
        }
    }
    pr[rows] = nnz;
    *out_nnz = nnz;

    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        int row_start = pr[i];
        int row_end   = pr[i + 1];
        for (int k = row_start; k < row_end; ++k)
            s += pv[k] * px[pc[k]];
        py[i] = s;
    }
}

/*
 * M4 - Fused single-pass, branchy.
 * Build CSR and accumulate y in the same row scan, so values[] is
 * touched once. Inner branch on v != 0 is preserved.
 */
static void m4_fused_branchy(int rows, int cols,
                             const double* A, const double* x,
                             int* out_nnz,
                             double* values, int* col_indices, int* row_ptrs,
                             double* y) {
    int nnz = 0;
    for (int i = 0; i < rows; ++i) {
        row_ptrs[i] = nnz;
        double s = 0.0;
        const double* row = A + (size_t) i * cols;
        for (int j = 0; j < cols; ++j) {
            double v = row[j];
            if (v != 0.0) {
                values[nnz]      = v;
                col_indices[nnz] = j;
                ++nnz;
                s += v * x[j];
            }
        }
        y[i] = s;
    }
    row_ptrs[rows] = nnz;
    *out_nnz = nnz;
}

/*
 * M5 - Fused single-pass, branchless.
 * Predicated CSR store plus an unconditional FMA into the row sum.
 * Multiplying by zero is harmless for correctness, and the dense FMA
 * loop autovectorizes cleanly.
 */
static void m5_fused_branchless(int rows, int cols,
                                const double* A, const double* x,
                                int* out_nnz,
                                double* values, int* col_indices, int* row_ptrs,
                                double* y) {
    const double* __restrict__ pA = A;
    const double* __restrict__ px = x;
    double*       __restrict__ pv = values;
    int*          __restrict__ pc = col_indices;
    int*          __restrict__ pr = row_ptrs;
    double*       __restrict__ py = y;

    int nnz = 0;
    for (int i = 0; i < rows; ++i) {
        pr[i] = nnz;
        double s = 0.0;
        const double* row = pA + (size_t) i * cols;
        for (int j = 0; j < cols; ++j) {
            double v = row[j];
            pv[nnz] = v;
            pc[nnz] = j;
            nnz += (v != 0.0);
            s   += v * px[j];
        }
        py[i] = s;
    }
    pr[rows] = nnz;
    *out_nnz = nnz;
}

/* ---------------------------------------------------------------- */
/* Harness                                                            */
/* ---------------------------------------------------------------- */

typedef struct {
    const char*  name;
    spmul_fn     fn;
    int          builds_csr;   /* 1 if the method actually populates CSR */
} method_t;

static method_t METHODS[] = {
    { "M1_dense_direct",       m1_dense_direct,       0 },
    { "M2_two_phase_naive",    m2_two_phase_naive,    1 },
    { "M3_two_phase_branchless", m3_two_phase_branchless, 1 },
    { "M4_fused_branchy",      m4_fused_branchy,      1 },
    { "M5_fused_branchless",   m5_fused_branchless,   1 },
};
enum { NMETHODS = sizeof(METHODS) / sizeof(METHODS[0]) };

/* Linear-congruential RNG so timings are reproducible. */
static uint32_t g_rng = 0xC0FFEEu;
static uint32_t rng_u32(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng; }
static double   rng_unit(void){ return (rng_u32() >> 8) / (double) (1u << 24); }

static void gen_matrix(double* A, int rows, int cols, double density) {
    size_t n = (size_t) rows * cols;
    for (size_t i = 0; i < n; ++i)
        A[i] = (rng_unit() < density) ? (rng_unit() * 20.0 - 10.0) : 0.0;
}
static void gen_vector(double* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = rng_unit() * 20.0 - 10.0;
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e9 + (double) ts.tv_nsec;
}

static int verify(const double* a, const double* b, int n) {
    for (int i = 0; i < n; ++i) {
        double d = fabs(a[i] - b[i]);
        double t = 1e-6 + 1e-6 * fabs(b[i]);
        if (d > t) return 0;
    }
    return 1;
}

/* Time one method by running it `iters` times, return best ns / iter. */
static double time_method(spmul_fn fn,
                          int rows, int cols, int density_pct,
                          const double* A, const double* x,
                          double* values, int* col_indices, int* row_ptrs,
                          double* y, int iters)
{
    (void) density_pct;
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        double t0 = now_ns();
        for (int it = 0; it < iters; ++it) {
            int out_nnz = 0;
            fn(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y);
        }
        double t1 = now_ns();
        double per = (t1 - t0) / iters;
        if (per < best) best = per;
    }
    return best;
}

static int calibrate_iters(spmul_fn fn,
                           int rows, int cols,
                           const double* A, const double* x,
                           double* values, int* col_indices, int* row_ptrs,
                           double* y)
{
    int iters = 1;
    for (;;) {
        double t0 = now_ns();
        for (int it = 0; it < iters; ++it) {
            int out_nnz = 0;
            fn(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y);
        }
        double dt = now_ns() - t0;
        if (dt > 5e7) return iters;       /* 50 ms */
        if (iters > 1 << 22) return iters;
        iters *= 2;
    }
}

static void run_config(int rows, int cols, double density)
{
    size_t mat = (size_t) rows * cols;
    double* A    = malloc(mat * sizeof(double));
    double* x    = malloc((size_t) cols * sizeof(double));
    double* y    = malloc((size_t) rows * sizeof(double));
    double* yref = malloc((size_t) rows * sizeof(double));
    double* vals = malloc(mat * sizeof(double));
    int*    cols_idx = malloc(mat * sizeof(int));
    int*    rptr  = malloc(((size_t) rows + 1) * sizeof(int));

    gen_matrix(A, rows, cols, density);
    gen_vector(x, cols);

    /* Reference: dense product. */
    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += A[(size_t) i * cols + j] * x[j];
        yref[i] = s;
    }

    int true_nnz = 0;
    for (size_t i = 0; i < mat; ++i) if (A[i] != 0.0) ++true_nnz;

    for (int m = 0; m < NMETHODS; ++m) {
        memset(y,    0, (size_t) rows * sizeof(double));
        memset(vals, 0, mat * sizeof(double));
        memset(cols_idx, 0, mat * sizeof(int));
        memset(rptr, 0, ((size_t) rows + 1) * sizeof(int));

        int iters = calibrate_iters(METHODS[m].fn, rows, cols, A, x,
                                    vals, cols_idx, rptr, y);
        double ns = time_method(METHODS[m].fn, rows, cols, (int)(density*100),
                                A, x, vals, cols_idx, rptr, y, iters);

        if (!verify(y, yref, rows)) {
            fprintf(stderr,
                    "VERIFY FAIL: %s rows=%d cols=%d density=%.2f\n",
                    METHODS[m].name, rows, cols, density);
            exit(2);
        }
        double gflops = 2.0 * true_nnz / ns;             /* 2 ops per nz in SpMV */
        printf("%s,%d,%d,%.4f,%d,%d,%.2f,%.4f\n",
               METHODS[m].name, rows, cols, density, true_nnz, iters, ns, gflops);
        fflush(stdout);
    }

    free(A); free(x); free(y); free(yref);
    free(vals); free(cols_idx); free(rptr);
}

int main(void)
{
    printf("method,rows,cols,density,nnz,iters,ns_per_run,gflops_spmv\n");

    /* Sweep 1: density at fixed 256x256 (working set ~512 KiB, L2 range). */
    int density_sizes_pct[] = { 1, 2, 5, 10, 20, 35, 50, 70, 90 };
    for (size_t k = 0; k < sizeof(density_sizes_pct)/sizeof(int); ++k) {
        g_rng = 0xC0FFEEu;     /* deterministic per config */
        run_config(256, 256, density_sizes_pct[k] / 100.0);
    }

    /* Sweep 2: size at fixed density 0.10. */
    int sizes[] = { 32, 64, 128, 256, 512, 1024 };
    for (size_t k = 0; k < sizeof(sizes)/sizeof(int); ++k) {
        g_rng = 0xC0FFEEu;
        run_config(sizes[k], sizes[k], 0.10);
    }

    /* Sweep 3: rectangular shapes at density 0.15. */
    int shapes[][2] = { {64, 1024}, {1024, 64}, {128, 512}, {512, 128} };
    for (size_t k = 0; k < sizeof(shapes)/sizeof(shapes[0]); ++k) {
        g_rng = 0xC0FFEEu;
        run_config(shapes[k][0], shapes[k][1], 0.15);
    }

    return 0;
}
