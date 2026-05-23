#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kMaxDims = 8;
constexpr int kThreads = 256;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct ShapeInfo {
    int ndim;
    int64_t numel;
    int64_t sizes[kMaxDims];
};

ShapeInfo make_shape_info(const std::vector<int64_t>& sizes) {
    TORCH_CHECK(!sizes.empty(), "CRESSO5 CUDA ops require at least one dimension");
    TORCH_CHECK(static_cast<int>(sizes.size()) <= kMaxDims, "CRESSO5 CUDA ops support tensors up to 8 dimensions");
    ShapeInfo info{};
    info.ndim = static_cast<int>(sizes.size());
    info.numel = 1;
    for (int i = 0; i < info.ndim; ++i) {
        TORCH_CHECK(sizes[i] > 0, "CRESSO5 CUDA ops require positive dimensions");
        info.sizes[i] = sizes[i];
        info.numel *= sizes[i];
    }
    return info;
}

int choose_chunks(int64_t numel) {
    int64_t chunks = (numel + 4095) / 4096;
    chunks = std::max<int64_t>(1, chunks);
    chunks = std::min<int64_t>(2048, chunks);
    return static_cast<int>(chunks);
}

__device__ __forceinline__ double hash_unit_device(int64_t n, int64_t salt) {
    uint64_t x = static_cast<uint64_t>(n + 1) * 0x9E3779B1ULL +
                 static_cast<uint64_t>(salt + 11) * 0x85EBCA77ULL;
    x = x ^ (x >> 16);
    x = (x * 0x7FEB352DULL) & 0xFFFFFFFFULL;
    x = x ^ (x >> 15);
    x = (x * 0x846CA68BULL) & 0xFFFFFFFFULL;
    x = x ^ (x >> 16);
    return static_cast<double>(x & 0xFFFFFFULL) / static_cast<double>(1 << 24);
}

__device__ __forceinline__ int64_t wrapped_mul_add(int64_t a, int64_t b, int64_t c) {
    uint64_t u = static_cast<uint64_t>(a) * static_cast<uint64_t>(b) + static_cast<uint64_t>(c);
    return static_cast<int64_t>(u);
}

__device__ __forceinline__ int64_t cresso_hash_index(int64_t i, int64_t bins, int64_t salt) {
    const int64_t salt_i = salt + 1;
    int64_t h = wrapped_mul_add(i, 1103515245LL + 97LL * salt_i, 12345LL + 0x9E3779B1LL * salt_i);
    h = h ^ (h >> 16);
    h = wrapped_mul_add(h, 2246822519LL + 1315423911LL * salt_i, 0LL);
    h = h ^ (h >> 13);
    h = wrapped_mul_add(h, 3266489917LL + 374761393LL * salt_i, 0LL);
    h = h ^ (h >> 16);
    int64_t r = h % bins;
    return r < 0 ? r + bins : r;
}

__device__ __forceinline__ int64_t freq_value(const int64_t* freqs, int r, int axis, int ndim) {
    return freqs[static_cast<int64_t>(r) * ndim + axis];
}

__device__ bool eval_basis_raw(
    int64_t linear,
    const int64_t* freqs,
    int r,
    ShapeInfo shape,
    double& cos_raw,
    double& sin_raw) {
    int64_t coords[kMaxDims];
    int64_t tmp = linear;
    for (int axis = shape.ndim - 1; axis >= 0; --axis) {
        const int64_t dim = shape.sizes[axis];
        coords[axis] = tmp % dim;
        tmp /= dim;
    }

    int64_t freq_sum = 0;
    for (int axis = 0; axis < shape.ndim; ++axis) {
        const int64_t f = freq_value(freqs, r, axis, shape.ndim);
        freq_sum += f < 0 ? -f : f;
    }
    if (freq_sum == 0) {
        cos_raw = 1.0;
        sin_raw = 0.0;
        return true;
    }

    const int kind = r & 3;
    if (kind == 0) {
        double phase = 0.0;
        for (int axis = 0; axis < shape.ndim; ++axis) {
            const int64_t f = freq_value(freqs, r, axis, shape.ndim);
            if (f == 0) {
                continue;
            }
            const double u = (static_cast<double>(coords[axis]) + 0.5) / static_cast<double>(shape.sizes[axis]);
            phase += kPi * static_cast<double>(f) * u;
        }
        cos_raw = cos(phase);
        sin_raw = sin(phase);
        return false;
    }

    if (kind == 1) {
        double cprod = 1.0;
        double sprod = 1.0;
        int active = 0;
        for (int axis = 0; axis < shape.ndim; ++axis) {
            const int64_t f = freq_value(freqs, r, axis, shape.ndim);
            if (f == 0) {
                continue;
            }
            ++active;
            const double u = (static_cast<double>(coords[axis]) + 0.5) / static_cast<double>(shape.sizes[axis]);
            const double phase = kPi * static_cast<double>(f) * u;
            cprod *= cos(phase);
            sprod *= sin(phase);
        }
        cos_raw = active == 0 ? 1.0 : cprod;
        sin_raw = active == 0 ? 0.0 : sprod;
        return active == 0;
    }

    double phase = 0.0;
    for (int axis = 0; axis < shape.ndim; ++axis) {
        const int64_t f = freq_value(freqs, r, axis, shape.ndim);
        const double u = (static_cast<double>(coords[axis]) + 0.5) / static_cast<double>(shape.sizes[axis]);
        const double centered = u - 0.5;
        const double chirp = 0.25 + 1.75 * hash_unit_device(r, 17 + axis);
        phase += kPi * static_cast<double>(f) * u;
        phase += kPi * chirp * static_cast<double>(f > 1 ? f : 1) * centered * centered;
    }

    if (kind == 2) {
        cos_raw = cos(phase);
        sin_raw = sin(phase);
        return false;
    }

    double window = 1.0;
    for (int axis = 0; axis < shape.ndim; ++axis) {
        const double u = (static_cast<double>(coords[axis]) + 0.5) / static_cast<double>(shape.sizes[axis]);
        const double center = 0.15 + 0.70 * hash_unit_device(r, 101 + axis);
        const double width = 0.18 + 0.20 * hash_unit_device(r, 151 + axis);
        const double z = (u - center) / width;
        window *= exp(-0.5 * z * z);
    }
    cos_raw = window * cos(phase);
    sin_raw = window * sin(phase);
    return false;
}

__device__ bool eval_basis_cached(
    int64_t linear,
    const int64_t* freqs,
    const float* axis_cache,
    int max_dim,
    int r,
    ShapeInfo shape,
    double& cos_raw,
    double& sin_raw) {
    int64_t coords[kMaxDims];
    int64_t tmp = linear;
    for (int axis = shape.ndim - 1; axis >= 0; --axis) {
        const int64_t dim = shape.sizes[axis];
        coords[axis] = tmp % dim;
        tmp /= dim;
    }

    int64_t freq_sum = 0;
    for (int axis = 0; axis < shape.ndim; ++axis) {
        const int64_t f = freq_value(freqs, r, axis, shape.ndim);
        freq_sum += f < 0 ? -f : f;
    }
    if (freq_sum == 0) {
        cos_raw = 1.0;
        sin_raw = 0.0;
        return true;
    }

    const int kind = r & 3;
    if (kind == 1) {
        double cprod = 1.0;
        double sprod = 1.0;
        int active = 0;
        for (int axis = 0; axis < shape.ndim; ++axis) {
            const int64_t f = freq_value(freqs, r, axis, shape.ndim);
            if (f == 0) {
                continue;
            }
            ++active;
            const int64_t base =
                (((static_cast<int64_t>(r) * shape.ndim + axis) * max_dim + coords[axis]) * 3);
            cprod *= static_cast<double>(axis_cache[base]);
            sprod *= static_cast<double>(axis_cache[base + 1]);
        }
        cos_raw = active == 0 ? 1.0 : cprod;
        sin_raw = active == 0 ? 0.0 : sprod;
        return active == 0;
    }

    double csum = 1.0;
    double ssum = 0.0;
    double window = 1.0;
    for (int axis = 0; axis < shape.ndim; ++axis) {
        const int64_t base =
            (((static_cast<int64_t>(r) * shape.ndim + axis) * max_dim + coords[axis]) * 3);
        const double ca = static_cast<double>(axis_cache[base]);
        const double sa = static_cast<double>(axis_cache[base + 1]);
        const double next_c = csum * ca - ssum * sa;
        const double next_s = ssum * ca + csum * sa;
        csum = next_c;
        ssum = next_s;
        if (kind == 3) {
            window *= static_cast<double>(axis_cache[base + 2]);
        }
    }
    cos_raw = window * csum;
    sin_raw = window * ssum;
    return false;
}

__device__ __forceinline__ void reduce4(double* shared, double v0, double v1, double v2, double v3) {
    const int tid = threadIdx.x;
    double* s0 = shared;
    double* s1 = shared + blockDim.x;
    double* s2 = shared + 2 * blockDim.x;
    double* s3 = shared + 3 * blockDim.x;
    s0[tid] = v0;
    s1[tid] = v1;
    s2[tid] = v2;
    s3[tid] = v3;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            s2[tid] += s2[tid + stride];
            s3[tid] += s3[tid + stride];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void reduce2(double* shared, double v0, double v1) {
    const int tid = threadIdx.x;
    double* s0 = shared;
    double* s1 = shared + blockDim.x;
    s0[tid] = v0;
    s1[tid] = v1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
        }
        __syncthreads();
    }
}

__global__ void basis_axis_cache_kernel(const int64_t* freqs, int rank, ShapeInfo shape, int max_dim, float* cache) {
    const int64_t total = static_cast<int64_t>(rank) * shape.ndim * max_dim;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int coord = static_cast<int>(idx % max_dim);
        const int axis = static_cast<int>((idx / max_dim) % shape.ndim);
        const int r = static_cast<int>(idx / (static_cast<int64_t>(max_dim) * shape.ndim));
        const int64_t out = idx * 3;
        if (coord >= shape.sizes[axis]) {
            cache[out] = 1.0f;
            cache[out + 1] = 0.0f;
            cache[out + 2] = 1.0f;
            continue;
        }
        const int kind = r & 3;
        const int64_t f = freq_value(freqs, r, axis, shape.ndim);
        const double u = (static_cast<double>(coord) + 0.5) / static_cast<double>(shape.sizes[axis]);
        double phase = 0.0;
        double window = 1.0;
        if (kind == 0) {
            phase = f == 0 ? 0.0 : kPi * static_cast<double>(f) * u;
        } else if (kind == 1) {
            if (f == 0) {
                cache[out] = 1.0f;
                cache[out + 1] = 1.0f;
                cache[out + 2] = 1.0f;
                continue;
            }
            phase = kPi * static_cast<double>(f) * u;
        } else {
            const double centered = u - 0.5;
            const double chirp = 0.25 + 1.75 * hash_unit_device(r, 17 + axis);
            phase = kPi * static_cast<double>(f) * u;
            phase += kPi * chirp * static_cast<double>(f > 1 ? f : 1) * centered * centered;
            if (kind == 3) {
                const double center = 0.15 + 0.70 * hash_unit_device(r, 101 + axis);
                const double width = 0.18 + 0.20 * hash_unit_device(r, 151 + axis);
                const double z = (u - center) / width;
                window = exp(-0.5 * z * z);
            }
        }
        cache[out] = static_cast<float>(cos(phase));
        cache[out + 1] = static_cast<float>(sin(phase));
        cache[out + 2] = static_cast<float>(window);
    }
}

__global__ void basis_stats_partial_kernel(
    const int64_t* freqs,
    int rank,
    ShapeInfo shape,
    int chunks,
    double* partial) {
    const int r = blockIdx.x;
    const int chunk = blockIdx.y;
    double cos_sum = 0.0;
    double cos_sq_sum = 0.0;
    double sin_sum = 0.0;
    double sin_sq_sum = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < shape.numel;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        double c = 0.0;
        double s = 0.0;
        eval_basis_raw(i, freqs, r, shape, c, s);
        cos_sum += c;
        cos_sq_sum += c * c;
        sin_sum += s;
        sin_sq_sum += s * s;
    }
    extern __shared__ double shared4[];
    reduce4(shared4, cos_sum, cos_sq_sum, sin_sum, sin_sq_sum);
    if (threadIdx.x == 0) {
        double* row = partial + (static_cast<int64_t>(r) * chunks + chunk) * 4;
        row[0] = shared4[0];
        row[1] = shared4[blockDim.x];
        row[2] = shared4[2 * blockDim.x];
        row[3] = shared4[3 * blockDim.x];
    }
}

__global__ void basis_stats_partial_cached_kernel(
    const int64_t* freqs,
    const float* axis_cache,
    int max_dim,
    int rank,
    ShapeInfo shape,
    int chunks,
    double* partial) {
    const int r = blockIdx.x;
    const int chunk = blockIdx.y;
    double cos_sum = 0.0;
    double cos_sq_sum = 0.0;
    double sin_sum = 0.0;
    double sin_sq_sum = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < shape.numel;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        double c = 0.0;
        double s = 0.0;
        eval_basis_cached(i, freqs, axis_cache, max_dim, r, shape, c, s);
        cos_sum += c;
        cos_sq_sum += c * c;
        sin_sum += s;
        sin_sq_sum += s * s;
    }
    extern __shared__ double shared4[];
    reduce4(shared4, cos_sum, cos_sq_sum, sin_sum, sin_sq_sum);
    if (threadIdx.x == 0) {
        double* row = partial + (static_cast<int64_t>(r) * chunks + chunk) * 4;
        row[0] = shared4[0];
        row[1] = shared4[blockDim.x];
        row[2] = shared4[2 * blockDim.x];
        row[3] = shared4[3 * blockDim.x];
    }
}

__global__ void basis_stats_finalize_kernel(const double* partial, int rank, int chunks, int64_t n, double* stats) {
    const int r = blockIdx.x;
    double sums[4] = {0.0, 0.0, 0.0, 0.0};
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        const double* row = partial + (static_cast<int64_t>(r) * chunks + chunk) * 4;
        sums[0] += row[0];
        sums[1] += row[1];
        sums[2] += row[2];
        sums[3] += row[3];
    }
    extern __shared__ double shared4[];
    reduce4(shared4, sums[0], sums[1], sums[2], sums[3]);
    if (threadIdx.x == 0) {
        const double inv_n = 1.0 / static_cast<double>(n);
        const double cos_mean = shared4[0] * inv_n;
        const double sin_mean = shared4[2 * blockDim.x] * inv_n;
        double cos_var = shared4[blockDim.x] * inv_n - cos_mean * cos_mean;
        double sin_var = shared4[3 * blockDim.x] * inv_n - sin_mean * sin_mean;
        cos_var = cos_var < 0.0 ? 0.0 : cos_var;
        sin_var = sin_var < 0.0 ? 0.0 : sin_var;
        const double cos_rms = sqrt(cos_var + 1.0e-12);
        const double sin_rms = sqrt(sin_var + 1.0e-12);
        double* out = stats + static_cast<int64_t>(r) * 4;
        out[0] = cos_mean;
        out[1] = 1.0 / (cos_rms < 1.0e-6 ? 1.0e-6 : cos_rms);
        out[2] = sin_mean;
        out[3] = 1.0 / (sin_rms < 1.0e-6 ? 1.0e-6 : sin_rms);
    }
}

template <typename scalar_t>
__global__ void basis_project_partial_kernel(
    const scalar_t* x,
    const int64_t* freqs,
    const double* stats,
    int rank,
    ShapeInfo shape,
    int chunks,
    double* partial) {
    const int r = blockIdx.x;
    const int chunk = blockIdx.y;
    const double* st = stats + static_cast<int64_t>(r) * 4;
    double cos_sum = 0.0;
    double sin_sum = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < shape.numel;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        double c = 0.0;
        double s = 0.0;
        const bool zero_mode = eval_basis_raw(i, freqs, r, shape, c, s);
        if (!zero_mode) {
            c = (c - st[0]) * st[1];
            s = (s - st[2]) * st[3];
        }
        const double xv = static_cast<double>(x[i]);
        cos_sum += xv * c;
        sin_sum += xv * s;
    }
    extern __shared__ double shared2[];
    reduce2(shared2, cos_sum, sin_sum);
    if (threadIdx.x == 0) {
        double* row = partial + (static_cast<int64_t>(r) * chunks + chunk) * 2;
        row[0] = shared2[0];
        row[1] = shared2[blockDim.x];
    }
}

template <typename scalar_t>
__global__ void basis_project_partial_cached_kernel(
    const scalar_t* x,
    const int64_t* freqs,
    const double* stats,
    const float* axis_cache,
    int max_dim,
    int rank,
    ShapeInfo shape,
    int chunks,
    double* partial) {
    const int r = blockIdx.x;
    const int chunk = blockIdx.y;
    const double* st = stats + static_cast<int64_t>(r) * 4;
    double cos_sum = 0.0;
    double sin_sum = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < shape.numel;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        double c = 0.0;
        double s = 0.0;
        const bool zero_mode = eval_basis_cached(i, freqs, axis_cache, max_dim, r, shape, c, s);
        if (!zero_mode) {
            c = (c - st[0]) * st[1];
            s = (s - st[2]) * st[3];
        }
        const double xv = static_cast<double>(x[i]);
        cos_sum += xv * c;
        sin_sum += xv * s;
    }
    extern __shared__ double shared2[];
    reduce2(shared2, cos_sum, sin_sum);
    if (threadIdx.x == 0) {
        double* row = partial + (static_cast<int64_t>(r) * chunks + chunk) * 2;
        row[0] = shared2[0];
        row[1] = shared2[blockDim.x];
    }
}

template <typename scalar_t>
__global__ void basis_project_finalize_kernel(
    const double* partial,
    int rank,
    int chunks,
    int64_t n,
    scalar_t* cos_out,
    scalar_t* sin_out) {
    const int r = blockIdx.x;
    double cos_sum = 0.0;
    double sin_sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        const double* row = partial + (static_cast<int64_t>(r) * chunks + chunk) * 2;
        cos_sum += row[0];
        sin_sum += row[1];
    }
    extern __shared__ double shared2[];
    reduce2(shared2, cos_sum, sin_sum);
    if (threadIdx.x == 0) {
        const double inv_n = 1.0 / static_cast<double>(n);
        cos_out[r] = static_cast<scalar_t>(shared2[0] * inv_n);
        sin_out[r] = static_cast<scalar_t>(shared2[blockDim.x] * inv_n);
    }
}

template <typename scalar_t>
__global__ void basis_project_small_cached_kernel(
    const scalar_t* x,
    const int64_t* freqs,
    const double* stats,
    const float* axis_cache,
    int max_dim,
    int rank,
    ShapeInfo shape,
    scalar_t* cos_out,
    scalar_t* sin_out) {
    const int r = blockIdx.x;
    const double* st = stats + static_cast<int64_t>(r) * 4;
    double cos_sum = 0.0;
    double sin_sum = 0.0;
    for (int64_t i = threadIdx.x; i < shape.numel; i += blockDim.x) {
        double c = 0.0;
        double s = 0.0;
        const bool zero_mode = eval_basis_cached(i, freqs, axis_cache, max_dim, r, shape, c, s);
        if (!zero_mode) {
            c = (c - st[0]) * st[1];
            s = (s - st[2]) * st[3];
        }
        const double xv = static_cast<double>(x[i]);
        cos_sum += xv * c;
        sin_sum += xv * s;
    }
    extern __shared__ double shared2[];
    reduce2(shared2, cos_sum, sin_sum);
    if (threadIdx.x == 0) {
        const double inv_n = 1.0 / static_cast<double>(shape.numel);
        cos_out[r] = static_cast<scalar_t>(shared2[0] * inv_n);
        sin_out[r] = static_cast<scalar_t>(shared2[blockDim.x] * inv_n);
    }
}

template <typename scalar_t>
__global__ void basis_reconstruct_kernel(
    scalar_t* out,
    const scalar_t* cos_coeff,
    const scalar_t* sin_coeff,
    const scalar_t* gates,
    bool has_gates,
    const int64_t* freqs,
    const double* stats,
    int rank,
    ShapeInfo shape,
    double shrink) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < shape.numel;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        double acc = 0.0;
        for (int r = 0; r < rank; ++r) {
            const double gate = has_gates ? static_cast<double>(gates[r]) : 1.0;
            if (gate == 0.0) {
                continue;
            }
            double c = 0.0;
            double s = 0.0;
            const bool zero_mode = eval_basis_raw(i, freqs, r, shape, c, s);
            if (!zero_mode) {
                const double* st = stats + static_cast<int64_t>(r) * 4;
                c = (c - st[0]) * st[1];
                s = (s - st[2]) * st[3];
            }
            acc += static_cast<double>(cos_coeff[r]) * shrink * gate * c;
            acc += static_cast<double>(sin_coeff[r]) * shrink * gate * s;
        }
        out[i] = static_cast<scalar_t>(acc);
    }
}

template <typename scalar_t>
__global__ void basis_reconstruct_cached_kernel(
    scalar_t* out,
    const scalar_t* cos_coeff,
    const scalar_t* sin_coeff,
    const scalar_t* gates,
    bool has_gates,
    const int64_t* freqs,
    const double* stats,
    const float* axis_cache,
    int max_dim,
    int rank,
    ShapeInfo shape,
    double shrink) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < shape.numel;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        double acc = 0.0;
        for (int r = 0; r < rank; ++r) {
            const double gate = has_gates ? static_cast<double>(gates[r]) : 1.0;
            if (gate == 0.0) {
                continue;
            }
            double c = 0.0;
            double s = 0.0;
            const bool zero_mode = eval_basis_cached(i, freqs, axis_cache, max_dim, r, shape, c, s);
            if (!zero_mode) {
                const double* st = stats + static_cast<int64_t>(r) * 4;
                c = (c - st[0]) * st[1];
                s = (s - st[2]) * st[3];
            }
            acc += static_cast<double>(cos_coeff[r]) * shrink * gate * c;
            acc += static_cast<double>(sin_coeff[r]) * shrink * gate * s;
        }
        out[i] = static_cast<scalar_t>(acc);
    }
}

template <typename scalar_t>
__global__ void hash_reduce_sum_kernel(
    const scalar_t* x,
    int64_t n,
    int64_t bins,
    int64_t salt,
    scalar_t* sums,
    int32_t* counts) {
    extern __shared__ unsigned char smem[];
    scalar_t* local_sums = reinterpret_cast<scalar_t*>(smem);
    int32_t* local_counts = reinterpret_cast<int32_t*>(local_sums + bins);

    for (int64_t b = threadIdx.x; b < bins; b += blockDim.x) {
        local_sums[b] = static_cast<scalar_t>(0);
        local_counts[b] = 0;
    }
    __syncthreads();

    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t b = cresso_hash_index(i, bins, salt);
        atomicAdd(local_sums + b, x[i]);
        atomicAdd(local_counts + b, 1);
    }
    __syncthreads();

    for (int64_t b = threadIdx.x; b < bins; b += blockDim.x) {
        if (local_counts[b] != 0) {
            atomicAdd(sums + b, local_sums[b]);
            atomicAdd(counts + b, local_counts[b]);
        }
    }
}

template <typename scalar_t>
__global__ void hash_finalize_mean_kernel(const scalar_t* sums, const int32_t* counts, int64_t bins, scalar_t* out) {
    for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; b < bins;
         b += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t count = counts[b];
        out[b] = count > 0 ? sums[b] / static_cast<scalar_t>(count) : static_cast<scalar_t>(0);
    }
}

template <typename scalar_t>
__global__ void hash_gather_kernel(
    const scalar_t* values,
    scalar_t* out,
    int64_t n,
    int64_t bins,
    int64_t salt) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        out[i] = values[cresso_hash_index(i, bins, salt)];
    }
}

void check_freqs(torch::Tensor freqs, const ShapeInfo& shape) {
    TORCH_CHECK(freqs.is_cuda(), "freqs must be a CUDA tensor");
    TORCH_CHECK(freqs.scalar_type() == at::kLong, "freqs must be int64");
    TORCH_CHECK(freqs.dim() == 2, "freqs must have shape [rank, ndim]");
    TORCH_CHECK(freqs.size(1) == shape.ndim, "freqs ndim does not match shape");
}

void check_stats(torch::Tensor stats, int64_t rank) {
    TORCH_CHECK(stats.is_cuda(), "stats must be a CUDA tensor");
    TORCH_CHECK(stats.scalar_type() == at::kDouble, "stats must be float64");
    TORCH_CHECK(stats.dim() == 2 && stats.size(0) == rank && stats.size(1) == 4, "stats must have shape [rank, 4]");
}

void check_axis_cache(torch::Tensor axis_cache, int64_t rank, const ShapeInfo& shape) {
    TORCH_CHECK(axis_cache.is_cuda(), "axis_cache must be a CUDA tensor");
    TORCH_CHECK(axis_cache.scalar_type() == at::kFloat, "axis_cache must be float32");
    TORCH_CHECK(axis_cache.dim() == 4, "axis_cache must have shape [rank, ndim, max_dim, 3]");
    TORCH_CHECK(axis_cache.size(0) == rank, "axis_cache rank mismatch");
    TORCH_CHECK(axis_cache.size(1) == shape.ndim, "axis_cache ndim mismatch");
    TORCH_CHECK(axis_cache.size(3) == 3, "axis_cache channel size must be 3");
}

}  // namespace

torch::Tensor basis_axis_cache_cuda(torch::Tensor freqs, std::vector<int64_t> sizes) {
    const ShapeInfo shape = make_shape_info(sizes);
    check_freqs(freqs, shape);
    const c10::cuda::CUDAGuard device_guard(freqs.device());
    auto freqs_c = freqs.contiguous();
    const int64_t rank = freqs_c.size(0);
    int64_t max_dim = 1;
    for (int i = 0; i < shape.ndim; ++i) {
        max_dim = std::max<int64_t>(max_dim, shape.sizes[i]);
    }
    auto cache = torch::empty({rank, shape.ndim, max_dim, 3}, freqs_c.options().dtype(torch::kFloat32));
    if (rank == 0) {
        return cache;
    }
    const int64_t total = rank * shape.ndim * max_dim;
    const int blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (total + kThreads - 1) / kThreads));
    basis_axis_cache_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        freqs_c.data_ptr<int64_t>(),
        static_cast<int>(rank),
        shape,
        static_cast<int>(max_dim),
        cache.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return cache;
}

torch::Tensor basis_stats_cuda(torch::Tensor freqs, std::vector<int64_t> sizes) {
    const ShapeInfo shape = make_shape_info(sizes);
    check_freqs(freqs, shape);
    const c10::cuda::CUDAGuard device_guard(freqs.device());
    auto freqs_c = freqs.contiguous();
    const int rank = static_cast<int>(freqs_c.size(0));
    auto stats = torch::empty({rank, 4}, freqs_c.options().dtype(torch::kFloat64));
    if (rank == 0) {
        return stats;
    }
    const int chunks = choose_chunks(shape.numel);
    auto partial = torch::empty({rank, chunks, 4}, stats.options());
    const dim3 grid(rank, chunks);
    basis_stats_partial_kernel<<<grid, kThreads, 4 * kThreads * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
        freqs_c.data_ptr<int64_t>(),
        rank,
        shape,
        chunks,
        partial.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    basis_stats_finalize_kernel<<<rank, kThreads, 4 * kThreads * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<double>(),
        rank,
        chunks,
        shape.numel,
        stats.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return stats;
}

torch::Tensor basis_stats_with_cache_cuda(torch::Tensor freqs, std::vector<int64_t> sizes, torch::Tensor axis_cache) {
    const ShapeInfo shape = make_shape_info(sizes);
    check_freqs(freqs, shape);
    const int64_t rank64 = freqs.size(0);
    check_axis_cache(axis_cache, rank64, shape);
    const c10::cuda::CUDAGuard device_guard(freqs.device());
    auto freqs_c = freqs.contiguous();
    auto cache_c = axis_cache.contiguous();
    const int rank = static_cast<int>(rank64);
    auto stats = torch::empty({rank, 4}, freqs_c.options().dtype(torch::kFloat64));
    if (rank == 0) {
        return stats;
    }
    const int chunks = choose_chunks(shape.numel);
    const int max_dim = static_cast<int>(cache_c.size(2));
    auto partial = torch::empty({rank, chunks, 4}, stats.options());
    const dim3 grid(rank, chunks);
    basis_stats_partial_cached_kernel<<<grid, kThreads, 4 * kThreads * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
        freqs_c.data_ptr<int64_t>(),
        cache_c.data_ptr<float>(),
        max_dim,
        rank,
        shape,
        chunks,
        partial.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    basis_stats_finalize_kernel<<<rank, kThreads, 4 * kThreads * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<double>(),
        rank,
        chunks,
        shape.numel,
        stats.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return stats;
}

std::vector<torch::Tensor> basis_project_with_stats_cuda(
    torch::Tensor x,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor stats) {
    const ShapeInfo shape = make_shape_info(sizes);
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.numel() == shape.numel, "x numel does not match shape");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble, "x must be float32 or float64");
    check_freqs(freqs, shape);
    const int64_t rank64 = freqs.size(0);
    TORCH_CHECK(rank64 <= 65535, "rank is too large for CRESSO5 CUDA projection");
    check_stats(stats, rank64);
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto x_c = x.contiguous();
    auto freqs_c = freqs.contiguous();
    auto stats_c = stats.contiguous();
    auto cos_out = torch::empty({rank64}, x_c.options());
    auto sin_out = torch::empty({rank64}, x_c.options());
    if (rank64 == 0) {
        return {cos_out, sin_out};
    }

    const int rank = static_cast<int>(rank64);
    const int chunks = choose_chunks(shape.numel);
    auto partial = torch::empty({rank, chunks, 2}, stats_c.options());
    const dim3 grid(rank, chunks);
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "basis_project_partial_cuda", [&] {
        basis_project_partial_kernel<scalar_t><<<
            grid,
            kThreads,
            2 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            x_c.data_ptr<scalar_t>(),
            freqs_c.data_ptr<int64_t>(),
            stats_c.data_ptr<double>(),
            rank,
            shape,
            chunks,
            partial.data_ptr<double>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "basis_project_finalize_cuda", [&] {
        basis_project_finalize_kernel<scalar_t><<<
            rank,
            kThreads,
            2 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            partial.data_ptr<double>(),
            rank,
            chunks,
            shape.numel,
            cos_out.data_ptr<scalar_t>(),
            sin_out.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cos_out, sin_out};
}

std::vector<torch::Tensor> basis_project_with_cache_cuda(
    torch::Tensor x,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor stats,
    torch::Tensor axis_cache) {
    const ShapeInfo shape = make_shape_info(sizes);
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.numel() == shape.numel, "x numel does not match shape");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble, "x must be float32 or float64");
    check_freqs(freqs, shape);
    const int64_t rank64 = freqs.size(0);
    TORCH_CHECK(rank64 <= 65535, "rank is too large for CRESSO5 CUDA projection");
    check_stats(stats, rank64);
    check_axis_cache(axis_cache, rank64, shape);
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto x_c = x.contiguous();
    auto freqs_c = freqs.contiguous();
    auto stats_c = stats.contiguous();
    auto cache_c = axis_cache.contiguous();
    auto cos_out = torch::empty({rank64}, x_c.options());
    auto sin_out = torch::empty({rank64}, x_c.options());
    if (rank64 == 0) {
        return {cos_out, sin_out};
    }

    const int rank = static_cast<int>(rank64);
    const int max_dim = static_cast<int>(cache_c.size(2));
    if (shape.numel <= 8192) {
        AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "basis_project_small_cached_cuda", [&] {
            basis_project_small_cached_kernel<scalar_t><<<
                rank,
                kThreads,
                2 * kThreads * sizeof(double),
                at::cuda::getCurrentCUDAStream()>>>(
                x_c.data_ptr<scalar_t>(),
                freqs_c.data_ptr<int64_t>(),
                stats_c.data_ptr<double>(),
                cache_c.data_ptr<float>(),
                max_dim,
                rank,
                shape,
                cos_out.data_ptr<scalar_t>(),
                sin_out.data_ptr<scalar_t>());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return {cos_out, sin_out};
    }

    const int chunks = choose_chunks(shape.numel);
    auto partial = torch::empty({rank, chunks, 2}, stats_c.options());
    const dim3 grid(rank, chunks);
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "basis_project_partial_cached_cuda", [&] {
        basis_project_partial_cached_kernel<scalar_t><<<
            grid,
            kThreads,
            2 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            x_c.data_ptr<scalar_t>(),
            freqs_c.data_ptr<int64_t>(),
            stats_c.data_ptr<double>(),
            cache_c.data_ptr<float>(),
            max_dim,
            rank,
            shape,
            chunks,
            partial.data_ptr<double>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "basis_project_finalize_cached_cuda", [&] {
        basis_project_finalize_kernel<scalar_t><<<
            rank,
            kThreads,
            2 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            partial.data_ptr<double>(),
            rank,
            chunks,
            shape.numel,
            cos_out.data_ptr<scalar_t>(),
            sin_out.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cos_out, sin_out};
}

std::vector<torch::Tensor> basis_project_cuda(torch::Tensor x, torch::Tensor freqs, std::vector<int64_t> sizes) {
    auto stats = basis_stats_cuda(freqs, sizes);
    return basis_project_with_stats_cuda(x, freqs, sizes, stats);
}

torch::Tensor basis_reconstruct_with_stats_cuda(
    torch::Tensor cos_coeff,
    torch::Tensor sin_coeff,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor gates,
    torch::Tensor stats) {
    const ShapeInfo shape = make_shape_info(sizes);
    TORCH_CHECK(cos_coeff.is_cuda() && sin_coeff.is_cuda(), "coefficients must be CUDA tensors");
    TORCH_CHECK(cos_coeff.scalar_type() == at::kFloat || cos_coeff.scalar_type() == at::kDouble, "coefficients must be float32 or float64");
    TORCH_CHECK(sin_coeff.scalar_type() == cos_coeff.scalar_type(), "cos/sin coefficient dtype mismatch");
    TORCH_CHECK(cos_coeff.dim() == 1 && sin_coeff.dim() == 1 && cos_coeff.numel() == sin_coeff.numel(), "coefficients must be matching 1D tensors");
    check_freqs(freqs, shape);
    const int64_t rank64 = cos_coeff.numel();
    TORCH_CHECK(freqs.size(0) == rank64, "freqs rank does not match coefficients");
    check_stats(stats, rank64);
    const bool has_gates = gates.numel() > 0;
    if (has_gates) {
        TORCH_CHECK(gates.is_cuda(), "gates must be a CUDA tensor");
        TORCH_CHECK(gates.scalar_type() == cos_coeff.scalar_type(), "gates dtype must match coefficients");
        TORCH_CHECK(gates.dim() == 1 && gates.numel() == rank64, "gates must have shape [rank]");
    }
    const c10::cuda::CUDAGuard device_guard(cos_coeff.device());
    auto cos_c = cos_coeff.contiguous();
    auto sin_c = sin_coeff.contiguous();
    auto freqs_c = freqs.contiguous();
    auto stats_c = stats.contiguous();
    auto gates_c = has_gates ? gates.contiguous() : torch::empty({0}, cos_c.options());
    auto out = torch::empty(sizes, cos_c.options());
    if (rank64 == 0) {
        out.zero_();
        return out;
    }
    const int rank = static_cast<int>(rank64);
    const int blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (shape.numel + kThreads - 1) / kThreads));
    const double shrink = 1.0 / sqrt(std::max(1.0, 0.7 * log2(static_cast<double>(rank) + 1.0)));
    AT_DISPATCH_FLOATING_TYPES(cos_c.scalar_type(), "basis_reconstruct_cuda", [&] {
        basis_reconstruct_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(),
            cos_c.data_ptr<scalar_t>(),
            sin_c.data_ptr<scalar_t>(),
            gates_c.data_ptr<scalar_t>(),
            has_gates,
            freqs_c.data_ptr<int64_t>(),
            stats_c.data_ptr<double>(),
            rank,
            shape,
            shrink);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor basis_reconstruct_with_cache_cuda(
    torch::Tensor cos_coeff,
    torch::Tensor sin_coeff,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor gates,
    torch::Tensor stats,
    torch::Tensor axis_cache) {
    const ShapeInfo shape = make_shape_info(sizes);
    TORCH_CHECK(cos_coeff.is_cuda() && sin_coeff.is_cuda(), "coefficients must be CUDA tensors");
    TORCH_CHECK(cos_coeff.scalar_type() == at::kFloat || cos_coeff.scalar_type() == at::kDouble, "coefficients must be float32 or float64");
    TORCH_CHECK(sin_coeff.scalar_type() == cos_coeff.scalar_type(), "cos/sin coefficient dtype mismatch");
    TORCH_CHECK(cos_coeff.dim() == 1 && sin_coeff.dim() == 1 && cos_coeff.numel() == sin_coeff.numel(), "coefficients must be matching 1D tensors");
    check_freqs(freqs, shape);
    const int64_t rank64 = cos_coeff.numel();
    TORCH_CHECK(freqs.size(0) == rank64, "freqs rank does not match coefficients");
    check_stats(stats, rank64);
    check_axis_cache(axis_cache, rank64, shape);
    const bool has_gates = gates.numel() > 0;
    if (has_gates) {
        TORCH_CHECK(gates.is_cuda(), "gates must be a CUDA tensor");
        TORCH_CHECK(gates.scalar_type() == cos_coeff.scalar_type(), "gates dtype must match coefficients");
        TORCH_CHECK(gates.dim() == 1 && gates.numel() == rank64, "gates must have shape [rank]");
    }
    const c10::cuda::CUDAGuard device_guard(cos_coeff.device());
    auto cos_c = cos_coeff.contiguous();
    auto sin_c = sin_coeff.contiguous();
    auto freqs_c = freqs.contiguous();
    auto stats_c = stats.contiguous();
    auto cache_c = axis_cache.contiguous();
    auto gates_c = has_gates ? gates.contiguous() : torch::empty({0}, cos_c.options());
    auto out = torch::empty(sizes, cos_c.options());
    if (rank64 == 0) {
        out.zero_();
        return out;
    }
    const int rank = static_cast<int>(rank64);
    const int max_dim = static_cast<int>(cache_c.size(2));
    const int blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (shape.numel + kThreads - 1) / kThreads));
    const double shrink = 1.0 / sqrt(std::max(1.0, 0.7 * log2(static_cast<double>(rank) + 1.0)));
    AT_DISPATCH_FLOATING_TYPES(cos_c.scalar_type(), "basis_reconstruct_cached_cuda", [&] {
        basis_reconstruct_cached_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(),
            cos_c.data_ptr<scalar_t>(),
            sin_c.data_ptr<scalar_t>(),
            gates_c.data_ptr<scalar_t>(),
            has_gates,
            freqs_c.data_ptr<int64_t>(),
            stats_c.data_ptr<double>(),
            cache_c.data_ptr<float>(),
            max_dim,
            rank,
            shape,
            shrink);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor basis_reconstruct_cuda(
    torch::Tensor cos_coeff,
    torch::Tensor sin_coeff,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor gates) {
    auto stats = basis_stats_cuda(freqs, sizes);
    return basis_reconstruct_with_stats_cuda(cos_coeff, sin_coeff, freqs, sizes, gates, stats);
}

torch::Tensor hash_reduce_mean_cuda(torch::Tensor x, int64_t bins, int64_t salt) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble, "x must be float32 or float64");
    TORCH_CHECK(bins > 0, "bins must be positive");
    TORCH_CHECK(bins <= 4096, "CRESSO5 CUDA hash reduce supports up to 4096 bins");
    const c10::cuda::CUDAGuard device_guard(x.device());
    auto x_c = x.contiguous();
    auto sums = torch::zeros({bins}, x_c.options());
    auto counts = torch::zeros({bins}, x_c.options().dtype(torch::kInt32));
    auto out = torch::empty({bins}, x_c.options());
    const int blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (x_c.numel() + 1023) / 1024));
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "hash_reduce_mean_cuda", [&] {
        const size_t shared_bytes = static_cast<size_t>(bins) * (sizeof(scalar_t) + sizeof(int32_t));
        hash_reduce_sum_kernel<scalar_t><<<blocks, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_c.data_ptr<scalar_t>(),
            x_c.numel(),
            bins,
            salt,
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const int norm_blocks = std::min<int64_t>(1024, std::max<int64_t>(1, (bins + kThreads - 1) / kThreads));
        hash_finalize_mean_kernel<scalar_t><<<norm_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>(),
            bins,
            out.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor hash_gather_cuda(torch::Tensor values, std::vector<int64_t> sizes, int64_t salt) {
    const ShapeInfo shape = make_shape_info(sizes);
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(values.scalar_type() == at::kFloat || values.scalar_type() == at::kDouble, "values must be float32 or float64");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");
    TORCH_CHECK(values.numel() > 0, "values must be non-empty");
    const int64_t bins = values.numel();
    const c10::cuda::CUDAGuard device_guard(values.device());
    auto values_c = values.contiguous();
    auto out = torch::empty(sizes, values_c.options());
    const int blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (shape.numel + kThreads - 1) / kThreads));
    AT_DISPATCH_FLOATING_TYPES(values_c.scalar_type(), "hash_gather_cuda", [&] {
        hash_gather_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            values_c.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            shape.numel,
            bins,
            salt);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

namespace {

template <typename scalar_t>
__global__ void hash_reset_accum_kernel(scalar_t* sums, int32_t* counts, int64_t bins) {
    for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; b < bins;
         b += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        sums[b] = static_cast<scalar_t>(0);
        counts[b] = 0;
    }
}

template <typename scalar_t>
__global__ void hash_metric_finalize_update_kernel(
    const scalar_t* sums,
    const int32_t* counts,
    int64_t bins,
    int64_t step,
    double hash_decay,
    scalar_t* hash_energy,
    double* mean_out) {
    double local = 0.0;
    for (int64_t b = threadIdx.x; b < bins; b += blockDim.x) {
        const int32_t count = counts[b];
        const double cell = count > 0 ? static_cast<double>(sums[b]) / static_cast<double>(count) : 0.0;
        const double updated =
            step <= 1 ? cell : hash_decay * static_cast<double>(hash_energy[b]) + (1.0 - hash_decay) * cell;
        hash_energy[b] = static_cast<scalar_t>(updated);
        local += updated;
    }
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    shared[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        mean_out[0] = shared[0] / static_cast<double>(bins);
    }
}

template <typename scalar_t>
__global__ void hash_metric_log_gather_kernel(
    const scalar_t* hash_energy,
    const double* mean,
    scalar_t* out,
    int64_t n,
    int64_t bins,
    int64_t salt,
    double exponent,
    double eps) {
    const double mean_v = mean[0] < eps ? eps : mean[0];
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t b = cresso_hash_index(i, bins, salt);
        double hm = static_cast<double>(hash_energy[b]) / mean_v;
        hm = hm < eps ? eps : hm;
        double hmt = pow(hm, exponent);
        hmt = hmt < 0.08 ? 0.08 : (hmt > 12.0 ? 12.0 : hmt);
        out[i] = static_cast<scalar_t>(log(hmt < eps ? eps : hmt));
    }
}

template <typename scalar_t>
__global__ void hash_impulse_update_kernel(
    const scalar_t* sums,
    const int32_t* counts,
    int64_t bins,
    double hash_decay,
    double hash_impulse_decay,
    double refractory_gain,
    scalar_t* hash_refractory,
    scalar_t* hash_impulse) {
    for (int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; b < bins;
         b += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int32_t count = counts[b];
        const double cell = count > 0 ? static_cast<double>(sums[b]) / static_cast<double>(count) : 0.0;
        const double ref =
            hash_decay * static_cast<double>(hash_refractory[b]) + (1.0 - hash_decay) * fabs(cell);
        hash_refractory[b] = static_cast<scalar_t>(ref);
        const double gate = 1.0 / (1.0 + refractory_gain * ref);
        const double impulse =
            hash_impulse_decay * static_cast<double>(hash_impulse[b]) + (1.0 - hash_impulse_decay) * cell * gate;
        hash_impulse[b] = static_cast<scalar_t>(impulse);
    }
}

template <typename scalar_t>
__global__ void metric_update_kernel(
    scalar_t* metric_cos,
    scalar_t* metric_sin,
    scalar_t* metric_p_cos,
    scalar_t* metric_p_sin,
    const scalar_t* mcos_port,
    const scalar_t* msin_port,
    const scalar_t* omega,
    int64_t rank,
    double metric_dt,
    double metric_friction) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rank) {
        return;
    }
    const double damp = fmax(0.0, 1.0 - metric_dt * metric_friction);
    const double w2 = static_cast<double>(omega[i]) * static_cast<double>(omega[i]);
    double pc = static_cast<double>(metric_p_cos[i]) * damp;
    pc += metric_dt * (static_cast<double>(mcos_port[i]) - w2 * static_cast<double>(metric_cos[i]));
    metric_p_cos[i] = static_cast<scalar_t>(pc);
    metric_cos[i] = static_cast<scalar_t>(fmin(3.0, fmax(-3.0, static_cast<double>(metric_cos[i]) + metric_dt * pc)));

    double ps = static_cast<double>(metric_p_sin[i]) * damp;
    ps += metric_dt * (static_cast<double>(msin_port[i]) - w2 * static_cast<double>(metric_sin[i]));
    metric_p_sin[i] = static_cast<scalar_t>(ps);
    metric_sin[i] = static_cast<scalar_t>(fmin(3.0, fmax(-3.0, static_cast<double>(metric_sin[i]) + metric_dt * ps)));
}

template <typename scalar_t>
__global__ void confidence_update_kernel(
    scalar_t* confidence,
    const scalar_t* pcos_port,
    const scalar_t* psin_port,
    const scalar_t* qcos,
    const scalar_t* qsin,
    int64_t rank,
    double confidence_decay,
    double eps) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rank) {
        return;
    }
    const double pc = static_cast<double>(pcos_port[i]);
    const double ps = static_cast<double>(psin_port[i]);
    const double qc = static_cast<double>(qcos[i]);
    const double qs = static_cast<double>(qsin[i]);
    const double port_energy = sqrt(pc * pc + ps * ps + eps);
    const double q_energy = sqrt(qc * qc + qs * qs + eps);
    double support = (pc * qc + ps * qs) / (port_energy * q_energy + eps);
    support = fmin(1.0, fmax(-1.0, support));
    const double updated = confidence_decay * static_cast<double>(confidence[i]) + (1.0 - confidence_decay) * support;
    confidence[i] = static_cast<scalar_t>(fmin(4.0, fmax(-4.0, updated)));
}

template <typename scalar_t>
__global__ void contact_update_pair_kernel(
    scalar_t* q,
    scalar_t* p,
    const scalar_t* port,
    const scalar_t* omega,
    scalar_t* calcium,
    scalar_t* action,
    const scalar_t* surprise,
    int64_t rank,
    double dt,
    double refractory_decay,
    double refractory_gain,
    double surprise_gain,
    double surprise_brake,
    double impulse_friction,
    double contact_gain,
    double restoring,
    double cubic) {
    extern __shared__ double shared[];
    double* gated = shared;
    double* reduce = shared + rank;
    const int tid = threadIdx.x;
    double energy_p = 0.0;
    double energy_q = 0.0;
    double port_power = 0.0;
    const double s = static_cast<double>(*surprise);
    const double plasticity = 1.0 + surprise_gain * s / (1.0 + s);
    const double brake = 1.0 / (1.0 + surprise_brake * s);
    for (int64_t i = tid; i < rank; i += blockDim.x) {
        const double port_i = static_cast<double>(port[i]);
        const double c = refractory_decay * static_cast<double>(calcium[i]) + (1.0 - refractory_decay) * fabs(port_i);
        calcium[i] = static_cast<scalar_t>(c);
        const double gate = 1.0 / (1.0 + refractory_gain * c);
        const double gp = port_i * gate * plasticity * brake;
        gated[i] = gp;
        const double p_i = static_cast<double>(p[i]);
        const double oq = static_cast<double>(omega[i]) * static_cast<double>(q[i]);
        energy_p += p_i * p_i;
        energy_q += oq * oq;
        port_power += gp * p_i;
    }
    reduce[tid] = energy_p;
    reduce[blockDim.x + tid] = energy_q;
    reduce[2 * blockDim.x + tid] = port_power;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
            reduce[blockDim.x + tid] += reduce[blockDim.x + tid + stride];
            reduce[2 * blockDim.x + tid] += reduce[2 * blockDim.x + tid + stride];
        }
        __syncthreads();
    }
    __shared__ double damp;
    if (tid == 0) {
        const double inv_rank = rank > 0 ? 1.0 / static_cast<double>(rank) : 0.0;
        const double energy = 0.5 * (reduce[0] * inv_rank + reduce[blockDim.x] * inv_rank);
        const double power = reduce[2 * blockDim.x] * inv_rank;
        const double action_value = 0.985 * static_cast<double>(*action) + dt * (power - energy);
        *action = static_cast<scalar_t>(action_value);
        const double friction = impulse_friction + contact_gain * tanh(fabs(action_value));
        damp = fmax(0.0, 1.0 - dt * friction);
    }
    __syncthreads();
    for (int64_t i = tid; i < rank; i += blockDim.x) {
        const double w2 = static_cast<double>(omega[i]) * static_cast<double>(omega[i]);
        double p_i = static_cast<double>(p[i]) * damp;
        const double q_i = static_cast<double>(q[i]);
        p_i += dt * (gated[i] - restoring * w2 * q_i - cubic * q_i * q_i * q_i);
        double q_next = q_i + dt * p_i;
        q_next = fmin(8.0, fmax(-8.0, q_next));
        p[i] = static_cast<scalar_t>(p_i);
        q[i] = static_cast<scalar_t>(q_next);
    }
}

void check_rank_tensor(torch::Tensor t, int64_t rank, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dim() == 1 && t.numel() == rank, name, " shape mismatch");
    TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kDouble, name, " must be float32 or float64");
}

}  // namespace

torch::Tensor hash_metric_update_log_cuda(
    torch::Tensor density_source,
    torch::Tensor hash_energy,
    std::vector<int64_t> sizes,
    int64_t salt,
    int64_t step,
    double hash_decay,
    double hash_metric_coupling,
    double energy_power,
    double eps) {
    const ShapeInfo shape = make_shape_info(sizes);
    TORCH_CHECK(density_source.is_cuda(), "density_source must be CUDA");
    TORCH_CHECK(density_source.scalar_type() == at::kFloat || density_source.scalar_type() == at::kDouble, "density_source must be float32 or float64");
    TORCH_CHECK(density_source.numel() == shape.numel, "density_source numel mismatch");
    TORCH_CHECK(hash_energy.is_cuda(), "hash_energy must be CUDA");
    TORCH_CHECK(hash_energy.dim() == 1 && hash_energy.numel() > 0, "hash_energy must be non-empty 1D");
    TORCH_CHECK(hash_energy.scalar_type() == density_source.scalar_type(), "hash_energy dtype must match density_source");
    const c10::cuda::CUDAGuard device_guard(density_source.device());
    auto x_c = density_source.contiguous();
    const int64_t bins = hash_energy.numel();
    TORCH_CHECK(bins <= 4096, "CRESSO5 CUDA hash metric supports up to 4096 bins");
    auto sums = torch::empty({bins}, x_c.options());
    auto counts = torch::empty({bins}, x_c.options().dtype(torch::kInt32));
    auto mean = torch::empty({1}, x_c.options().dtype(torch::kFloat64));
    auto out = torch::empty(sizes, x_c.options());
    const int bin_blocks = std::min<int64_t>(1024, std::max<int64_t>(1, (bins + kThreads - 1) / kThreads));
    const int value_blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (x_c.numel() + 1023) / 1024));
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "hash_metric_update_log_cuda", [&] {
        hash_reset_accum_kernel<scalar_t><<<bin_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>(),
            bins);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const size_t shared_bytes = static_cast<size_t>(bins) * (sizeof(scalar_t) + sizeof(int32_t));
        hash_reduce_sum_kernel<scalar_t><<<value_blocks, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_c.data_ptr<scalar_t>(),
            x_c.numel(),
            bins,
            salt,
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hash_metric_finalize_update_kernel<scalar_t><<<
            1,
            kThreads,
            kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>(),
            bins,
            step,
            hash_decay,
            hash_energy.data_ptr<scalar_t>(),
            mean.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const double exponent = hash_metric_coupling / std::max(0.25, energy_power);
        hash_metric_log_gather_kernel<scalar_t><<<value_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            hash_energy.data_ptr<scalar_t>(),
            mean.data_ptr<double>(),
            out.data_ptr<scalar_t>(),
            x_c.numel(),
            bins,
            salt,
            exponent,
            eps);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor hash_impulse_update_gather_cuda(
    torch::Tensor port_source,
    torch::Tensor hash_refractory,
    torch::Tensor hash_impulse,
    std::vector<int64_t> sizes,
    int64_t salt,
    double hash_decay,
    double hash_impulse_decay,
    double refractory_gain) {
    const ShapeInfo shape = make_shape_info(sizes);
    TORCH_CHECK(port_source.is_cuda(), "port_source must be CUDA");
    TORCH_CHECK(port_source.scalar_type() == at::kFloat || port_source.scalar_type() == at::kDouble, "port_source must be float32 or float64");
    TORCH_CHECK(port_source.numel() == shape.numel, "port_source numel mismatch");
    TORCH_CHECK(hash_refractory.is_cuda() && hash_impulse.is_cuda(), "hash state must be CUDA");
    TORCH_CHECK(hash_refractory.dim() == 1 && hash_impulse.dim() == 1 && hash_refractory.numel() == hash_impulse.numel(), "hash state shape mismatch");
    TORCH_CHECK(hash_refractory.scalar_type() == port_source.scalar_type() && hash_impulse.scalar_type() == port_source.scalar_type(), "hash state dtype mismatch");
    const c10::cuda::CUDAGuard device_guard(port_source.device());
    auto x_c = port_source.contiguous();
    const int64_t bins = hash_impulse.numel();
    TORCH_CHECK(bins <= 4096, "CRESSO5 CUDA hash impulse supports up to 4096 bins");
    auto sums = torch::empty({bins}, x_c.options());
    auto counts = torch::empty({bins}, x_c.options().dtype(torch::kInt32));
    auto out = torch::empty(sizes, x_c.options());
    const int bin_blocks = std::min<int64_t>(1024, std::max<int64_t>(1, (bins + kThreads - 1) / kThreads));
    const int value_blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (x_c.numel() + 1023) / 1024));
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "hash_impulse_update_gather_cuda", [&] {
        hash_reset_accum_kernel<scalar_t><<<bin_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>(),
            bins);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        const size_t shared_bytes = static_cast<size_t>(bins) * (sizeof(scalar_t) + sizeof(int32_t));
        hash_reduce_sum_kernel<scalar_t><<<value_blocks, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            x_c.data_ptr<scalar_t>(),
            x_c.numel(),
            bins,
            salt,
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hash_impulse_update_kernel<scalar_t><<<bin_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            sums.data_ptr<scalar_t>(),
            counts.data_ptr<int32_t>(),
            bins,
            hash_decay,
            hash_impulse_decay,
            refractory_gain,
            hash_refractory.data_ptr<scalar_t>(),
            hash_impulse.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hash_gather_kernel<scalar_t><<<value_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            hash_impulse.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            x_c.numel(),
            bins,
            salt);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

namespace {

template <typename scalar_t>
__global__ void axis_metric_reset_kernel(scalar_t* rows, scalar_t* cols, int64_t row_count, int64_t col_count) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < row_count + col_count;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        if (i < row_count) {
            rows[i] = static_cast<scalar_t>(0);
        } else {
            cols[i - row_count] = static_cast<scalar_t>(0);
        }
    }
}

template <typename scalar_t>
__global__ void axis_metric_sum_kernel(
    const scalar_t* rel,
    scalar_t* rows,
    scalar_t* cols,
    int64_t row_count,
    int64_t col_count,
    double power,
    double eps) {
    const int64_t n = row_count * col_count;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        double v = static_cast<double>(rel[i]);
        v = v < eps ? eps : v;
        v = pow(v, power);
        v = v > 1.0e6 ? 1.0e6 : v;
        const int64_t row = i / col_count;
        const int64_t col = i - row * col_count;
        atomicAdd(rows + row, static_cast<scalar_t>(v));
        atomicAdd(cols + col, static_cast<scalar_t>(v));
    }
}

__device__ __forceinline__ double axis_block_sum_local(double value, double* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename scalar_t>
__global__ void axis_metric_global_kernel(const scalar_t* rows, int64_t row_count, int64_t col_count, double* global_mean) {
    double local = 0.0;
    for (int64_t row = threadIdx.x; row < row_count; row += blockDim.x) {
        local += static_cast<double>(rows[row]);
    }
    extern __shared__ double shared[];
    const double total = axis_block_sum_local(local, shared);
    if (threadIdx.x == 0) {
        global_mean[0] = total / static_cast<double>(row_count * col_count);
    }
}

template <typename scalar_t>
__global__ void axis_metric_gather_kernel(
    const scalar_t* rows,
    const scalar_t* cols,
    const double* global_mean,
    scalar_t* out,
    int64_t row_count,
    int64_t col_count,
    double power,
    double coupling,
    double eps) {
    const int64_t n = row_count * col_count;
    const double g = global_mean[0] < eps ? eps : global_mean[0];
    const double coeff = coupling / (2.0 * power);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = i / col_count;
        const int64_t col = i - row * col_count;
        double row_norm = (static_cast<double>(rows[row]) / static_cast<double>(col_count)) / g;
        double col_norm = (static_cast<double>(cols[col]) / static_cast<double>(row_count)) / g;
        row_norm = row_norm < eps ? eps : row_norm;
        col_norm = col_norm < eps ? eps : col_norm;
        double metric = exp(coeff * (log(row_norm) + log(col_norm)));
        metric = metric < 0.08 ? 0.08 : (metric > 12.0 ? 12.0 : metric);
        out[i] = static_cast<scalar_t>(metric);
    }
}

}  // namespace

torch::Tensor instant_axis_metric_2d_cuda(torch::Tensor rel, double power, double coupling, double eps) {
    TORCH_CHECK(rel.is_cuda(), "rel must be CUDA");
    TORCH_CHECK(rel.dim() == 2, "instant_axis_metric_2d requires a 2D tensor");
    TORCH_CHECK(rel.scalar_type() == at::kFloat || rel.scalar_type() == at::kDouble, "rel must be float32 or float64");
    const c10::cuda::CUDAGuard device_guard(rel.device());
    auto rel_c = rel.contiguous();
    const int64_t rows_n = rel_c.size(0);
    const int64_t cols_n = rel_c.size(1);
    power = std::max(power, 0.25);
    auto rows = torch::empty({rows_n}, rel_c.options());
    auto cols = torch::empty({cols_n}, rel_c.options());
    auto global = torch::empty({1}, rel_c.options().dtype(torch::kFloat64));
    auto out = torch::empty_like(rel_c);
    const int reset_blocks = std::min<int64_t>(1024, std::max<int64_t>(1, (rows_n + cols_n + kThreads - 1) / kThreads));
    const int value_blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (rel_c.numel() + kThreads - 1) / kThreads));
    AT_DISPATCH_FLOATING_TYPES(rel_c.scalar_type(), "instant_axis_metric_2d_cuda", [&] {
        axis_metric_reset_kernel<scalar_t><<<reset_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            rows.data_ptr<scalar_t>(),
            cols.data_ptr<scalar_t>(),
            rows_n,
            cols_n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        axis_metric_sum_kernel<scalar_t><<<value_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            rel_c.data_ptr<scalar_t>(),
            rows.data_ptr<scalar_t>(),
            cols.data_ptr<scalar_t>(),
            rows_n,
            cols_n,
            power,
            eps);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        axis_metric_global_kernel<scalar_t><<<1, kThreads, kThreads * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
            rows.data_ptr<scalar_t>(),
            rows_n,
            cols_n,
            global.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        axis_metric_gather_kernel<scalar_t><<<value_blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            rows.data_ptr<scalar_t>(),
            cols.data_ptr<scalar_t>(),
            global.data_ptr<double>(),
            out.data_ptr<scalar_t>(),
            rows_n,
            cols_n,
            power,
            coupling,
            eps);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void metric_update_cuda(
    torch::Tensor metric_cos,
    torch::Tensor metric_sin,
    torch::Tensor metric_p_cos,
    torch::Tensor metric_p_sin,
    torch::Tensor mcos_port,
    torch::Tensor msin_port,
    torch::Tensor omega,
    double metric_dt,
    double metric_friction) {
    const int64_t rank = metric_cos.numel();
    check_rank_tensor(metric_cos, rank, "metric_cos");
    check_rank_tensor(metric_sin, rank, "metric_sin");
    check_rank_tensor(metric_p_cos, rank, "metric_p_cos");
    check_rank_tensor(metric_p_sin, rank, "metric_p_sin");
    check_rank_tensor(mcos_port, rank, "mcos_port");
    check_rank_tensor(msin_port, rank, "msin_port");
    check_rank_tensor(omega, rank, "omega");
    TORCH_CHECK(metric_sin.scalar_type() == metric_cos.scalar_type(), "metric dtype mismatch");
    const c10::cuda::CUDAGuard device_guard(metric_cos.device());
    const int threads = 128;
    const int blocks = std::max<int64_t>(1, (rank + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(metric_cos.scalar_type(), "metric_update_cuda", [&] {
        metric_update_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            metric_cos.data_ptr<scalar_t>(),
            metric_sin.data_ptr<scalar_t>(),
            metric_p_cos.data_ptr<scalar_t>(),
            metric_p_sin.data_ptr<scalar_t>(),
            mcos_port.data_ptr<scalar_t>(),
            msin_port.data_ptr<scalar_t>(),
            omega.data_ptr<scalar_t>(),
            rank,
            metric_dt,
            metric_friction);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void confidence_update_cuda(
    torch::Tensor confidence,
    torch::Tensor pcos_port,
    torch::Tensor psin_port,
    torch::Tensor qcos,
    torch::Tensor qsin,
    double confidence_decay,
    double eps) {
    const int64_t rank = confidence.numel();
    check_rank_tensor(confidence, rank, "confidence");
    check_rank_tensor(pcos_port, rank, "pcos_port");
    check_rank_tensor(psin_port, rank, "psin_port");
    check_rank_tensor(qcos, rank, "qcos");
    check_rank_tensor(qsin, rank, "qsin");
    const c10::cuda::CUDAGuard device_guard(confidence.device());
    const int threads = 128;
    const int blocks = std::max<int64_t>(1, (rank + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(confidence.scalar_type(), "confidence_update_cuda", [&] {
        confidence_update_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            confidence.data_ptr<scalar_t>(),
            pcos_port.data_ptr<scalar_t>(),
            psin_port.data_ptr<scalar_t>(),
            qcos.data_ptr<scalar_t>(),
            qsin.data_ptr<scalar_t>(),
            rank,
            confidence_decay,
            eps);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void contact_update_pair_cuda(
    torch::Tensor q,
    torch::Tensor p,
    torch::Tensor port,
    torch::Tensor omega,
    torch::Tensor calcium,
    torch::Tensor action,
    torch::Tensor surprise,
    double dt,
    double refractory_decay,
    double refractory_gain,
    double surprise_gain,
    double surprise_brake,
    double impulse_friction,
    double contact_gain,
    double restoring,
    double cubic) {
    const int64_t rank = q.numel();
    check_rank_tensor(q, rank, "q");
    check_rank_tensor(p, rank, "p");
    check_rank_tensor(port, rank, "port");
    check_rank_tensor(omega, rank, "omega");
    check_rank_tensor(calcium, rank, "calcium");
    TORCH_CHECK(action.is_cuda() && action.dim() == 0, "action must be a CUDA scalar");
    TORCH_CHECK(surprise.is_cuda() && surprise.dim() == 0, "surprise must be a CUDA scalar");
    TORCH_CHECK(rank <= 4096, "contact_update_pair supports rank <= 4096");
    const c10::cuda::CUDAGuard device_guard(q.device());
    const int threads = 128;
    const size_t shared_bytes = static_cast<size_t>(rank + 3 * threads) * sizeof(double);
    AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "contact_update_pair_cuda", [&] {
        contact_update_pair_kernel<scalar_t><<<1, threads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(),
            p.data_ptr<scalar_t>(),
            port.data_ptr<scalar_t>(),
            omega.data_ptr<scalar_t>(),
            calcium.data_ptr<scalar_t>(),
            action.data_ptr<scalar_t>(),
            surprise.data_ptr<scalar_t>(),
            rank,
            dt,
            refractory_decay,
            refractory_gain,
            surprise_gain,
            surprise_brake,
            impulse_friction,
            contact_gain,
            restoring,
            cubic);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

__device__ __forceinline__ void reduce6_local(double* shared, double v0, double v1, double v2, double v3, double v4, double v5) {
    const int tid = threadIdx.x;
    double* s0 = shared;
    double* s1 = shared + blockDim.x;
    double* s2 = shared + 2 * blockDim.x;
    double* s3 = shared + 3 * blockDim.x;
    double* s4 = shared + 4 * blockDim.x;
    double* s5 = shared + 5 * blockDim.x;
    s0[tid] = v0;
    s1[tid] = v1;
    s2[tid] = v2;
    s3[tid] = v3;
    s4[tid] = v4;
    s5[tid] = v5;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            s2[tid] += s2[tid + stride];
            s3[tid] += s3[tid + stride];
            s4[tid] += s4[tid + stride];
            s5[tid] += s5[tid + stride];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ double block_sum_local(double value, double* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename scalar_t>
__global__ void drive_stats_partial_kernel(
    const scalar_t* tangent,
    const scalar_t* pred,
    const scalar_t* error,
    int64_t n,
    int chunks,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    double tf2 = 0.0;
    double pred2 = 0.0;
    double dot = 0.0;
    double rational2 = 0.0;
    double root2 = 0.0;
    double error2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double t = static_cast<double>(tangent[i]);
        const double p = static_cast<double>(pred[i]);
        const double e = static_cast<double>(error[i]);
        const double rational = t / (1.0 + fabs(t) / clip);
        const double root_raw = (t < 0.0 ? -1.0 : (t > 0.0 ? 1.0 : 0.0)) * sqrt(fmin(fabs(t), clip * clip) + eps);
        tf2 += t * t;
        pred2 += p * p;
        dot += t * p;
        rational2 += rational * rational;
        root2 += root_raw * root_raw;
        error2 += e * e;
    }
    extern __shared__ double shared6[];
    reduce6_local(shared6, tf2, pred2, dot, rational2, root2, error2);
    if (threadIdx.x == 0) {
        double* row = partial + static_cast<int64_t>(chunk) * 6;
        row[0] = shared6[0];
        row[1] = shared6[blockDim.x];
        row[2] = shared6[2 * blockDim.x];
        row[3] = shared6[3 * blockDim.x];
        row[4] = shared6[4 * blockDim.x];
        row[5] = shared6[5 * blockDim.x];
    }
}

template <typename scalar_t>
__global__ void drive_stats_finalize_kernel(
    const double* partial,
    int chunks,
    int64_t n,
    const scalar_t* surprise,
    double prediction_mix,
    int64_t warmup_steps,
    int64_t step,
    double eps,
    double* stats) {
    double sums[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        const double* row = partial + static_cast<int64_t>(chunk) * 6;
        sums[0] += row[0];
        sums[1] += row[1];
        sums[2] += row[2];
        sums[3] += row[3];
        sums[4] += row[4];
        sums[5] += row[5];
    }
    extern __shared__ double shared6[];
    reduce6_local(shared6, sums[0], sums[1], sums[2], sums[3], sums[4], sums[5]);
    if (threadIdx.x == 0) {
        const double inv_n = 1.0 / static_cast<double>(n);
        const double force_rms = sqrt(shared6[0] * inv_n + eps);
        const double pred_rms = sqrt(shared6[blockDim.x] * inv_n + eps);
        const double rational_rms = sqrt(shared6[3 * blockDim.x] * inv_n + eps);
        const double root_rms = sqrt(shared6[4 * blockDim.x] * inv_n + eps);
        const double error_rms = sqrt(shared6[5 * blockDim.x] * inv_n + eps);
        double align_gate = 0.0;
        if (pred_rms > sqrt(eps)) {
            const double alignment = fmin(1.0, fmax(-1.0, (shared6[2 * blockDim.x] * inv_n) / (force_rms * pred_rms + eps)));
            align_gate = fmin(1.0, fmax(0.0, (alignment + 1.0) * 0.5));
        }
        const double warmup = warmup_steps == 0 ? 1.0 : (1.0 - exp(-static_cast<double>(step) / fmax(1.0, static_cast<double>(warmup_steps))));
        const double surprise_value = static_cast<double>(surprise[0]);
        const double mix = prediction_mix * warmup * align_gate / (1.0 + surprise_value);
        stats[0] = force_rms;
        stats[1] = pred_rms;
        stats[2] = rational_rms;
        stats[3] = root_rms;
        stats[4] = error_rms;
        stats[5] = mix;
        stats[6] = surprise_value;
    }
}

template <typename scalar_t>
__device__ __forceinline__ double shaped_value_device(
    double t,
    const double* stats,
    double root_channel_mix,
    double spike_mix,
    bool hard_enabled,
    double clip,
    double eps) {
    const double rational = t / (1.0 + fabs(t) / clip);
    const double bounded = tanh(t / clip) * clip;
    const double root_raw = (t < 0.0 ? -1.0 : (t > 0.0 ? 1.0 : 0.0)) * sqrt(fmin(fabs(t), clip * clip) + eps);
    const double root = root_raw * (stats[2] / fmax(stats[3], eps));
    const double surprise = stats[6];
    const double spike = spike_mix / (1.0 + surprise);
    const double base = (1.0 - spike) * rational + spike * bounded;
    const double root_mix = hard_enabled ? root_channel_mix / (1.0 + 0.5 * surprise) : 0.0;
    return (1.0 - root_mix) * base + root_mix * root;
}

template <typename scalar_t>
__global__ void shaped_stats_partial_kernel(
    const scalar_t* tangent,
    const scalar_t* echo,
    const scalar_t* hash_pred,
    bool has_echo,
    bool has_hash_pred,
    bool hard_enabled,
    int64_t n,
    int chunks,
    const double* stats,
    double root_channel_mix,
    double spike_mix,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    double shaped2 = 0.0;
    double echo2 = 0.0;
    double hash2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double shaped = shaped_value_device<scalar_t>(
            static_cast<double>(tangent[i]), stats, root_channel_mix, spike_mix, hard_enabled, clip, eps);
        shaped2 += shaped * shaped;
        if (has_echo) {
            const double v = static_cast<double>(echo[i]);
            echo2 += v * v;
        }
        if (has_hash_pred) {
            const double v = static_cast<double>(hash_pred[i]);
            hash2 += v * v;
        }
    }
    extern __shared__ double shared[];
    reduce4(shared, shaped2, echo2, hash2, 0.0);
    if (threadIdx.x == 0) {
        double* row = partial + static_cast<int64_t>(chunk) * 3;
        row[0] = shared[0];
        row[1] = shared[blockDim.x];
        row[2] = shared[2 * blockDim.x];
    }
}

__global__ void shaped_stats_finalize_kernel(const double* partial, int chunks, int64_t n, double eps, double* stats) {
    double shaped = 0.0;
    double echo = 0.0;
    double hash = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        const double* row = partial + static_cast<int64_t>(chunk) * 3;
        shaped += row[0];
        echo += row[1];
        hash += row[2];
    }
    extern __shared__ double shared[];
    reduce4(shared, shaped, echo, hash, 0.0);
    if (threadIdx.x == 0) {
        const double inv_n = 1.0 / static_cast<double>(n);
        stats[7] = sqrt(shared[0] * inv_n + eps);
        stats[8] = sqrt(shared[blockDim.x] * inv_n + eps);
        stats[9] = sqrt(shared[2 * blockDim.x] * inv_n + eps);
    }
}

template <typename scalar_t>
__global__ void core_stats_partial_kernel(
    const scalar_t* tangent,
    bool hard_enabled,
    int64_t n,
    int chunks,
    const double* stats,
    double direct_force_mix,
    double root_channel_mix,
    double spike_mix,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    double core2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double t = static_cast<double>(tangent[i]);
        const double shaped = shaped_value_device<scalar_t>(t, stats, root_channel_mix, spike_mix, hard_enabled, clip, eps);
        const double direct_mix = direct_force_mix / (1.0 + 0.35 * stats[6]);
        const double direct = t * (stats[7] / fmax(stats[0], eps));
        const double core = (1.0 - direct_mix) * shaped + direct_mix * direct;
        core2 += core * core;
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(core2, shared);
    if (threadIdx.x == 0) {
        partial[chunk] = total;
    }
}

__global__ void core_stats_finalize_kernel(const double* partial, int chunks, int64_t n, double eps, double* stats) {
    double sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        sum += partial[chunk];
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(sum, shared);
    if (threadIdx.x == 0) {
        stats[10] = sqrt(total / static_cast<double>(n) + eps);
    }
}

template <typename scalar_t>
__global__ void drive_partial_kernel(
    const scalar_t* tangent,
    const scalar_t* pred,
    const scalar_t* error,
    const scalar_t* echo,
    const scalar_t* hash_pred,
    scalar_t* drive,
    bool has_echo,
    bool has_hash_pred,
    bool hard_enabled,
    int64_t n,
    int chunks,
    const double* stats,
    double novelty_mix,
    double echo_mix,
    double hash_drive_mix,
    double direct_force_mix,
    double residual_feedback_mix,
    double root_channel_mix,
    double spike_mix,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    double drive2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double t = static_cast<double>(tangent[i]);
        const double e = static_cast<double>(error[i]);
        const double shaped = shaped_value_device<scalar_t>(t, stats, root_channel_mix, spike_mix, hard_enabled, clip, eps);
        const double direct_mix = direct_force_mix / (1.0 + 0.35 * stats[6]);
        const double direct = t * (stats[7] / fmax(stats[0], eps));
        const double core = (1.0 - direct_mix) * shaped + direct_mix * direct;
        const double residual_mix = residual_feedback_mix / (1.0 + stats[6]);
        const double residual = e * (stats[10] / fmax(stats[4], eps));
        const double novelty = tanh(e / (clip * (1.0 + stats[6]))) * clip;
        double d = (1.0 - stats[5]) * core + stats[5] * static_cast<double>(pred[i]) + novelty_mix * novelty +
                   residual_mix * residual;
        if (hard_enabled && has_echo) {
            d += echo_mix * static_cast<double>(echo[i]) * (stats[7] / fmax(stats[8], eps));
        }
        if (hard_enabled && has_hash_pred) {
            d += hash_drive_mix * static_cast<double>(hash_pred[i]) * (stats[7] / fmax(stats[9], eps));
        }
        drive[i] = static_cast<scalar_t>(d);
        drive2 += d * d;
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(drive2, shared);
    if (threadIdx.x == 0) {
        partial[chunk] = total;
    }
}

template <typename scalar_t>
__global__ void drive_gain_update_kernel(
    const double* partial,
    int chunks,
    int64_t n,
    int64_t step,
    scalar_t* drive_energy,
    double reservoir_decay,
    double target_update_rms,
    double min_gain,
    double max_gain,
    double eps,
    double* gain_out) {
    double sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        sum += partial[chunk];
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(sum, shared);
    if (threadIdx.x == 0) {
        const double now = total / static_cast<double>(n);
        const double updated = step <= 1 ? now : static_cast<double>(*drive_energy) * reservoir_decay + now * (1.0 - reservoir_decay);
        *drive_energy = static_cast<scalar_t>(updated);
        double gain = target_update_rms / sqrt(updated + eps);
        gain = gain < min_gain ? min_gain : (gain > max_gain ? max_gain : gain);
        gain_out[0] = gain;
    }
}

template <typename scalar_t>
__global__ void param_apply_drive_kernel(scalar_t* param, const scalar_t* drive, const double* gain, int64_t n, double lr) {
    const double g = gain[0];
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        param[i] = static_cast<scalar_t>(static_cast<double>(param[i]) - lr * static_cast<double>(drive[i]) * g);
    }
}

__device__ __forceinline__ double clamp_double_local(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

template <typename scalar_t>
__device__ __forceinline__ double scalar_force_value(
    const scalar_t* param,
    const scalar_t* grad,
    int64_t i,
    double weight_decay) {
    double f = static_cast<double>(grad[i]);
    if (weight_decay != 0.0) {
        f += weight_decay * static_cast<double>(param[i]);
    }
    return f;
}

template <typename scalar_t>
__device__ __forceinline__ double scalar_tangent_value_2d(
    const scalar_t* param,
    const scalar_t* grad,
    const double* row_sums,
    const double* col_sums,
    const double* stats,
    int64_t rows,
    int64_t cols,
    int64_t i,
    double weight_decay,
    double local_sharpness,
    double instant_metric_coupling,
    double instant_metric_power,
    double eps) {
    const double force = scalar_force_value(param, grad, i, weight_decay);
    const double scale = fmax(stats[0], eps);
    const double rel = fmin(fabs(force) / scale + eps, 1.0e4);
    double local_metric = 1.0;
    if (local_sharpness != 0.0) {
        local_metric = clamp_double_local(1.0 + local_sharpness * sqrt(fmin(rel, 100.0)), 1.0, 8.0);
    }
    double instant_metric = 1.0;
    const int64_t n = rows * cols;
    if (instant_metric_coupling != 0.0 && n > 1) {
        const int64_t row = i / cols;
        const int64_t col = i - row * cols;
        const double axis_mean = fmax(stats[14], eps);
        const double row_mean = row_sums[row] / static_cast<double>(cols);
        const double col_mean = col_sums[col] / static_cast<double>(rows);
        const double row_norm = row_mean / axis_mean;
        const double col_norm = col_mean / axis_mean;
        const double power = fmax(instant_metric_power, 0.25);
        const double log_metric = log(fmax(row_norm, eps)) + log(fmax(col_norm, eps));
        instant_metric = clamp_double_local(exp((instant_metric_coupling / (2.0 * power)) * log_metric), 0.08, 12.0);
    }
    const double denom = scale * stats[1] * local_metric * instant_metric + eps;
    return clamp_double_local(force / denom, -1.0e4, 1.0e4);
}

template <typename scalar_t>
__global__ void scalar_force_partial_kernel(
    const scalar_t* param,
    const scalar_t* grad,
    int64_t n,
    int chunks,
    double weight_decay,
    double* partial) {
    const int chunk = blockIdx.x;
    double sum = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double f = scalar_force_value(param, grad, i, weight_decay);
        sum += f * f;
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(sum, shared);
    if (threadIdx.x == 0) {
        partial[chunk] = total;
    }
}

template <typename scalar_t>
__global__ void scalar_force_finalize_kernel(
    const double* partial,
    int chunks,
    int64_t n,
    int64_t step,
    scalar_t* force_rms,
    double reservoir_decay,
    double eps,
    double* stats) {
    double sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        sum += partial[chunk];
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(sum, shared);
    if (threadIdx.x == 0) {
        const double now = sqrt(total / static_cast<double>(n) + eps);
        const double old = static_cast<double>(*force_rms);
        const double updated = step <= 1 ? now : old * reservoir_decay + now * (1.0 - reservoir_decay);
        *force_rms = static_cast<scalar_t>(updated);
        stats[0] = fmax(updated, eps);
    }
}

template <typename scalar_t>
__global__ void scalar_axis_density_kernel(
    const scalar_t* param,
    const scalar_t* grad,
    double* row_sums,
    double* col_sums,
    int64_t rows,
    int64_t cols,
    int chunks,
    const double* stats,
    double weight_decay,
    double energy_power,
    double instant_metric_power,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    const int64_t n = rows * cols;
    double log_sum = 0.0;
    double axis_sum = 0.0;
    const double scale = fmax(stats[0], eps);
    const double axis_power = fmax(instant_metric_power, 0.25);
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double force = scalar_force_value(param, grad, i, weight_decay);
        const double rel = fmin(fabs(force) / scale + eps, 1.0e4);
        const double density = fmin(pow(fmax(rel, eps), axis_power), 1.0e6);
        const int64_t row = i / cols;
        const int64_t col = i - row * cols;
        atomicAdd(row_sums + row, density);
        atomicAdd(col_sums + col, density);
        axis_sum += density;
        log_sum += log1p(pow(rel, energy_power));
    }
    extern __shared__ double shared[];
    reduce4(shared, log_sum, axis_sum, 0.0, 0.0);
    if (threadIdx.x == 0) {
        double* row = partial + static_cast<int64_t>(chunk) * 2;
        row[0] = shared[0];
        row[1] = shared[blockDim.x];
    }
}

template <typename scalar_t>
__global__ void scalar_metric_finalize_kernel(
    const double* partial,
    int chunks,
    int64_t n,
    scalar_t* metric_q,
    scalar_t* metric_impulse,
    double metric_dt,
    double metric_friction,
    double metric_coupling,
    double* stats) {
    double log_sum = 0.0;
    double axis_sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        const double* row = partial + static_cast<int64_t>(chunk) * 2;
        log_sum += row[0];
        axis_sum += row[1];
    }
    extern __shared__ double shared[];
    reduce4(shared, log_sum, axis_sum, 0.0, 0.0);
    if (threadIdx.x == 0) {
        double mq = static_cast<double>(*metric_q);
        double mp = static_cast<double>(*metric_impulse);
        const double density = shared[0] / static_cast<double>(n);
        const double metric_port = density - tanh(mq);
        mp *= fmax(0.0, 1.0 - metric_dt * metric_friction);
        mp += (metric_port - mq) * metric_dt;
        mq += mp * metric_dt;
        mq = clamp_double_local(mq, -3.0, 3.0);
        *metric_q = static_cast<scalar_t>(mq);
        *metric_impulse = static_cast<scalar_t>(mp);
        stats[1] = clamp_double_local(exp(metric_coupling * mq), 0.06, 16.0);
        stats[14] = shared[blockDim.x] / static_cast<double>(n);
    }
}

template <typename scalar_t>
__global__ void scalar_pre_contact_partial_kernel(
    const scalar_t* param,
    const scalar_t* grad,
    const double* row_sums,
    const double* col_sums,
    scalar_t* tangent_out,
    const scalar_t* q,
    int64_t rows,
    int64_t cols,
    int chunks,
    const double* stats,
    double weight_decay,
    double local_sharpness,
    double instant_metric_coupling,
    double instant_metric_power,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    const int64_t n = rows * cols;
    const double qv = static_cast<double>(*q);
    double err_sum = 0.0;
    double err2_sum = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double tangent = scalar_tangent_value_2d(
            param,
            grad,
            row_sums,
            col_sums,
            stats,
            rows,
            cols,
            i,
            weight_decay,
            local_sharpness,
            instant_metric_coupling,
            instant_metric_power,
            eps);
        tangent_out[i] = static_cast<scalar_t>(tangent);
        const double error = tangent - qv;
        err_sum += error;
        err2_sum += error * error;
    }
    extern __shared__ double shared[];
    reduce4(shared, err_sum, err2_sum, 0.0, 0.0);
    if (threadIdx.x == 0) {
        double* row = partial + static_cast<int64_t>(chunk) * 2;
        row[0] = shared[0];
        row[1] = shared[blockDim.x];
    }
}

template <typename scalar_t>
__global__ void scalar_contact_update_kernel(
    const double* partial,
    int chunks,
    int64_t n,
    scalar_t* q,
    scalar_t* impulse,
    scalar_t* surprise,
    scalar_t* action,
    int64_t step,
    double dt,
    double impulse_friction,
    double contact_gain,
    double restoring,
    double cubic,
    double refractory_gain,
    double surprise_gain,
    double surprise_decay,
    double surprise_brake,
    double eps,
    double* stats) {
    double err_sum = 0.0;
    double err2_sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        const double* row = partial + static_cast<int64_t>(chunk) * 2;
        err_sum += row[0];
        err2_sum += row[1];
    }
    extern __shared__ double shared[];
    reduce4(shared, err_sum, err2_sum, 0.0, 0.0);
    if (threadIdx.x == 0) {
        double qv = static_cast<double>(*q);
        double iv = static_cast<double>(*impulse);
        double sv = static_cast<double>(*surprise);
        double av = static_cast<double>(*action);
        const double surprise_now = sqrt(shared[blockDim.x] / static_cast<double>(n) + eps);
        sv = step <= 1 ? surprise_now : sv * surprise_decay + surprise_now * (1.0 - surprise_decay);
        const double port = shared[0] / static_cast<double>(n);
        const double plasticity = 1.0 + surprise_gain * sv / (1.0 + sv);
        const double brake = 1.0 / (1.0 + surprise_brake * sv);
        const double energy = 0.5 * (iv * iv + qv * qv);
        av = av * 0.985 + (port * iv - energy) * dt;
        const double friction = impulse_friction + contact_gain * tanh(fabs(av));
        iv *= fmax(0.0, 1.0 - dt * friction);
        iv += (plasticity * brake * port - restoring * qv - cubic * qv * qv * qv) * dt;
        qv += iv * dt;
        qv = clamp_double_local(qv, -8.0, 8.0);
        *q = static_cast<scalar_t>(qv);
        *impulse = static_cast<scalar_t>(iv);
        *surprise = static_cast<scalar_t>(sv);
        *action = static_cast<scalar_t>(av);
        stats[2] = qv;
        stats[3] = sv;
    }
}

template <typename scalar_t>
__global__ void scalar_drive_stats_partial_kernel(
    const scalar_t* tangent,
    int64_t rows,
    int64_t cols,
    int chunks,
    const double* stats,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    const int64_t n = rows * cols;
    const double qv = stats[2];
    double tf2 = 0.0;
    double pred2 = 0.0;
    double dot = 0.0;
    double rational2 = 0.0;
    double root2 = 0.0;
    double error2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double t = static_cast<double>(tangent[i]);
        const double e = t - qv;
        const double rational = t / (1.0 + fabs(t) / clip);
        const double root_raw = (t < 0.0 ? -1.0 : (t > 0.0 ? 1.0 : 0.0)) * sqrt(fmin(fabs(t), clip * clip) + eps);
        tf2 += t * t;
        pred2 += qv * qv;
        dot += t * qv;
        rational2 += rational * rational;
        root2 += root_raw * root_raw;
        error2 += e * e;
    }
    extern __shared__ double shared6[];
    reduce6_local(shared6, tf2, pred2, dot, rational2, root2, error2);
    if (threadIdx.x == 0) {
        double* row = partial + static_cast<int64_t>(chunk) * 6;
        row[0] = shared6[0];
        row[1] = shared6[blockDim.x];
        row[2] = shared6[2 * blockDim.x];
        row[3] = shared6[3 * blockDim.x];
        row[4] = shared6[4 * blockDim.x];
        row[5] = shared6[5 * blockDim.x];
    }
}

__global__ void scalar_drive_stats_finalize_kernel(
    const double* partial,
    int chunks,
    int64_t n,
    int64_t step,
    double prediction_mix,
    int64_t warmup_steps,
    double eps,
    double* stats) {
    double sums[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        const double* row = partial + static_cast<int64_t>(chunk) * 6;
        sums[0] += row[0];
        sums[1] += row[1];
        sums[2] += row[2];
        sums[3] += row[3];
        sums[4] += row[4];
        sums[5] += row[5];
    }
    extern __shared__ double shared6[];
    reduce6_local(shared6, sums[0], sums[1], sums[2], sums[3], sums[4], sums[5]);
    if (threadIdx.x == 0) {
        const double inv_n = 1.0 / static_cast<double>(n);
        const double force_rms = sqrt(shared6[0] * inv_n + eps);
        const double pred_rms = sqrt(shared6[blockDim.x] * inv_n + eps);
        const double rational_rms = sqrt(shared6[3 * blockDim.x] * inv_n + eps);
        const double root_rms = sqrt(shared6[4 * blockDim.x] * inv_n + eps);
        const double error_rms = sqrt(shared6[5 * blockDim.x] * inv_n + eps);
        double align_gate = 0.0;
        if (pred_rms > sqrt(eps)) {
            const double alignment = clamp_double_local((shared6[2 * blockDim.x] * inv_n) / (force_rms * pred_rms + eps), -1.0, 1.0);
            align_gate = clamp_double_local((alignment + 1.0) * 0.5, 0.0, 1.0);
        }
        const double warmup = warmup_steps == 0 ? 1.0 : (1.0 - exp(-static_cast<double>(step) / fmax(1.0, static_cast<double>(warmup_steps))));
        const double surprise = stats[3];
        stats[4] = force_rms;
        stats[5] = pred_rms;
        stats[6] = rational_rms;
        stats[7] = root_rms;
        stats[8] = error_rms;
        stats[9] = prediction_mix * warmup * align_gate / (1.0 + surprise);
    }
}

template <typename scalar_t>
__device__ __forceinline__ double scalar_shaped_value(double t, const double* stats, double root_channel_mix, double spike_mix, double clip, double eps) {
    const double rational = t / (1.0 + fabs(t) / clip);
    const double bounded = tanh(t / clip) * clip;
    const double root_raw = (t < 0.0 ? -1.0 : (t > 0.0 ? 1.0 : 0.0)) * sqrt(fmin(fabs(t), clip * clip) + eps);
    const double root = root_raw * (stats[6] / fmax(stats[7], eps));
    const double spike = spike_mix / (1.0 + stats[3]);
    const double base = (1.0 - spike) * rational + spike * bounded;
    const double root_mix = root_channel_mix / (1.0 + 0.5 * stats[3]);
    return (1.0 - root_mix) * base + root_mix * root;
}

template <typename scalar_t>
__global__ void scalar_shaped_partial_kernel(
    const scalar_t* tangent,
    int64_t rows,
    int64_t cols,
    int chunks,
    const double* stats,
    double root_channel_mix,
    double spike_mix,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    const int64_t n = rows * cols;
    double shaped2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double t = static_cast<double>(tangent[i]);
        const double shaped = scalar_shaped_value<scalar_t>(t, stats, root_channel_mix, spike_mix, clip, eps);
        shaped2 += shaped * shaped;
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(shaped2, shared);
    if (threadIdx.x == 0) {
        partial[chunk] = total;
    }
}

__global__ void scalar_one_stat_finalize_kernel(const double* partial, int chunks, int64_t n, double eps, double* stats, int64_t index) {
    double sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        sum += partial[chunk];
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(sum, shared);
    if (threadIdx.x == 0) {
        stats[index] = sqrt(total / static_cast<double>(n) + eps);
    }
}

template <typename scalar_t>
__global__ void scalar_core_partial_kernel(
    const scalar_t* tangent,
    int64_t rows,
    int64_t cols,
    int chunks,
    const double* stats,
    double direct_force_mix,
    double root_channel_mix,
    double spike_mix,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    const int64_t n = rows * cols;
    const double direct_mix = direct_force_mix / (1.0 + 0.35 * stats[3]);
    double core2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double t = static_cast<double>(tangent[i]);
        const double shaped = scalar_shaped_value<scalar_t>(t, stats, root_channel_mix, spike_mix, clip, eps);
        const double direct = t * (stats[10] / fmax(stats[4], eps));
        const double core = (1.0 - direct_mix) * shaped + direct_mix * direct;
        core2 += core * core;
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(core2, shared);
    if (threadIdx.x == 0) {
        partial[chunk] = total;
    }
}

template <typename scalar_t>
__global__ void scalar_drive_partial_kernel(
    const scalar_t* tangent,
    scalar_t* drive,
    int64_t rows,
    int64_t cols,
    int chunks,
    const double* stats,
    double novelty_mix,
    double direct_force_mix,
    double residual_feedback_mix,
    double root_channel_mix,
    double spike_mix,
    double clip,
    double eps,
    double* partial) {
    const int chunk = blockIdx.x;
    const int64_t n = rows * cols;
    const double qv = stats[2];
    const double direct_mix = direct_force_mix / (1.0 + 0.35 * stats[3]);
    const double residual_mix = residual_feedback_mix / (1.0 + stats[3]);
    double drive2 = 0.0;
    for (int64_t i = static_cast<int64_t>(chunk) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(chunks) * blockDim.x) {
        const double t = static_cast<double>(tangent[i]);
        const double error = t - qv;
        const double shaped = scalar_shaped_value<scalar_t>(t, stats, root_channel_mix, spike_mix, clip, eps);
        const double direct = t * (stats[10] / fmax(stats[4], eps));
        const double core = (1.0 - direct_mix) * shaped + direct_mix * direct;
        const double residual = error * (stats[11] / fmax(stats[8], eps));
        const double novelty = tanh(error / (clip * (1.0 + stats[3]))) * clip;
        const double d = (1.0 - stats[9]) * core + stats[9] * qv + novelty_mix * novelty + residual_mix * residual;
        drive[i] = static_cast<scalar_t>(d);
        drive2 += d * d;
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(drive2, shared);
    if (threadIdx.x == 0) {
        partial[chunk] = total;
    }
}

template <typename scalar_t>
__global__ void scalar_gain_finalize_kernel(
    const double* partial,
    int chunks,
    int64_t n,
    int64_t step,
    scalar_t* drive_energy,
    double reservoir_decay,
    double target_update_rms,
    double min_gain,
    double max_gain,
    double eps,
    double* stats) {
    double sum = 0.0;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += blockDim.x) {
        sum += partial[chunk];
    }
    extern __shared__ double shared[];
    const double total = block_sum_local(sum, shared);
    if (threadIdx.x == 0) {
        const double now = total / static_cast<double>(n);
        const double old = static_cast<double>(*drive_energy);
        const double updated = step <= 1 ? now : old * reservoir_decay + now * (1.0 - reservoir_decay);
        *drive_energy = static_cast<scalar_t>(updated);
        stats[12] = clamp_double_local(target_update_rms / sqrt(updated + eps), min_gain, max_gain);
    }
}

template <typename scalar_t>
__global__ void scalar_param_apply_kernel(scalar_t* param, const scalar_t* drive, const double* stats, int64_t n, double lr) {
    const double gain = stats[12];
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        param[i] = static_cast<scalar_t>(static_cast<double>(param[i]) - lr * static_cast<double>(drive[i]) * gain);
    }
}

}  // namespace

void drive_update_cuda(
    torch::Tensor param,
    torch::Tensor tangent_force,
    torch::Tensor pred,
    torch::Tensor error,
    torch::Tensor echo,
    torch::Tensor hash_pred,
    torch::Tensor surprise,
    torch::Tensor drive_energy,
    int64_t step,
    bool hard_enabled,
    double lr,
    double reservoir_decay,
    double prediction_mix,
    double novelty_mix,
    double echo_mix,
    double hash_drive_mix,
    double direct_force_mix,
    double residual_feedback_mix,
    double root_channel_mix,
    double spike_mix,
    int64_t warmup_steps,
    double target_update_rms,
    double min_gain,
    double max_gain,
    double drive_clip,
    double eps) {
    TORCH_CHECK(param.is_cuda() && tangent_force.is_cuda() && pred.is_cuda() && error.is_cuda(), "drive_update tensors must be CUDA");
    TORCH_CHECK(param.is_contiguous() && tangent_force.is_contiguous() && pred.is_contiguous() && error.is_contiguous(), "drive_update requires contiguous tensors");
    TORCH_CHECK(param.scalar_type() == tangent_force.scalar_type() && pred.scalar_type() == tangent_force.scalar_type() &&
                    error.scalar_type() == tangent_force.scalar_type(),
                "drive_update dtype mismatch");
    TORCH_CHECK(param.scalar_type() == at::kFloat || param.scalar_type() == at::kDouble, "drive_update supports float32 and float64 params");
    TORCH_CHECK(param.numel() == tangent_force.numel() && pred.numel() == tangent_force.numel() && error.numel() == tangent_force.numel(),
                "drive_update numel mismatch");
    TORCH_CHECK(surprise.is_cuda() && surprise.dim() == 0 && drive_energy.is_cuda() && drive_energy.dim() == 0, "drive_update scalar state must be CUDA scalars");
    TORCH_CHECK(surprise.scalar_type() == tangent_force.scalar_type() && drive_energy.scalar_type() == tangent_force.scalar_type(),
                "drive_update scalar state dtype mismatch");
    const bool has_echo = echo.numel() > 0;
    const bool has_hash_pred = hash_pred.numel() > 0;
    if (has_echo) {
        TORCH_CHECK(echo.is_cuda() && echo.is_contiguous() && echo.numel() == tangent_force.numel() &&
                        echo.scalar_type() == tangent_force.scalar_type(),
                    "echo shape/dtype mismatch");
    }
    if (has_hash_pred) {
        TORCH_CHECK(hash_pred.is_cuda() && hash_pred.is_contiguous() && hash_pred.numel() == tangent_force.numel() &&
                        hash_pred.scalar_type() == tangent_force.scalar_type(),
                    "hash_pred shape/dtype mismatch");
    }
    const c10::cuda::CUDAGuard device_guard(param.device());
    const int64_t n = tangent_force.numel();
    const int chunks = choose_chunks(n);
    const int blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (n + kThreads - 1) / kThreads));
    auto partial6 = torch::empty({chunks, 6}, tangent_force.options().dtype(torch::kFloat64));
    auto partial3 = torch::empty({chunks, 3}, tangent_force.options().dtype(torch::kFloat64));
    auto partial1 = torch::empty({chunks}, tangent_force.options().dtype(torch::kFloat64));
    auto stats = torch::empty({11}, tangent_force.options().dtype(torch::kFloat64));
    auto gain = torch::empty({1}, tangent_force.options().dtype(torch::kFloat64));
    auto drive = torch::empty_like(tangent_force);
    auto empty = tangent_force;
    AT_DISPATCH_FLOATING_TYPES(tangent_force.scalar_type(), "drive_update_cuda", [&] {
        drive_stats_partial_kernel<scalar_t><<<
            chunks,
            kThreads,
            6 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            tangent_force.data_ptr<scalar_t>(),
            pred.data_ptr<scalar_t>(),
            error.data_ptr<scalar_t>(),
            n,
            chunks,
            drive_clip,
            eps,
            partial6.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        drive_stats_finalize_kernel<scalar_t><<<
            1,
            kThreads,
            6 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            partial6.data_ptr<double>(),
            chunks,
            n,
            surprise.data_ptr<scalar_t>(),
            prediction_mix,
            warmup_steps,
            step,
            eps,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        shaped_stats_partial_kernel<scalar_t><<<
            chunks,
            kThreads,
            4 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            tangent_force.data_ptr<scalar_t>(),
            has_echo ? echo.data_ptr<scalar_t>() : empty.data_ptr<scalar_t>(),
            has_hash_pred ? hash_pred.data_ptr<scalar_t>() : empty.data_ptr<scalar_t>(),
            has_echo,
            has_hash_pred,
            hard_enabled,
            n,
            chunks,
            stats.data_ptr<double>(),
            root_channel_mix,
            spike_mix,
            drive_clip,
            eps,
            partial3.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        shaped_stats_finalize_kernel<<<
            1,
            kThreads,
            4 * kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            partial3.data_ptr<double>(),
            chunks,
            n,
            eps,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        core_stats_partial_kernel<scalar_t><<<
            chunks,
            kThreads,
            kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            tangent_force.data_ptr<scalar_t>(),
            hard_enabled,
            n,
            chunks,
            stats.data_ptr<double>(),
            direct_force_mix,
            root_channel_mix,
            spike_mix,
            drive_clip,
            eps,
            partial1.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        core_stats_finalize_kernel<<<1, kThreads, kThreads * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
            partial1.data_ptr<double>(),
            chunks,
            n,
            eps,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        drive_partial_kernel<scalar_t><<<
            chunks,
            kThreads,
            kThreads * sizeof(double),
            at::cuda::getCurrentCUDAStream()>>>(
            tangent_force.data_ptr<scalar_t>(),
            pred.data_ptr<scalar_t>(),
            error.data_ptr<scalar_t>(),
            has_echo ? echo.data_ptr<scalar_t>() : empty.data_ptr<scalar_t>(),
            has_hash_pred ? hash_pred.data_ptr<scalar_t>() : empty.data_ptr<scalar_t>(),
            drive.data_ptr<scalar_t>(),
            has_echo,
            has_hash_pred,
            hard_enabled,
            n,
            chunks,
            stats.data_ptr<double>(),
            novelty_mix,
            echo_mix,
            hash_drive_mix,
            direct_force_mix,
            residual_feedback_mix,
            root_channel_mix,
            spike_mix,
            drive_clip,
            eps,
            partial1.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        drive_gain_update_kernel<scalar_t><<<1, kThreads, kThreads * sizeof(double), at::cuda::getCurrentCUDAStream()>>>(
            partial1.data_ptr<double>(),
            chunks,
            n,
            step,
            drive_energy.data_ptr<scalar_t>(),
            reservoir_decay,
            target_update_rms,
            min_gain,
            max_gain,
            eps,
            gain.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        param_apply_drive_kernel<scalar_t><<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            param.data_ptr<scalar_t>(),
            drive.data_ptr<scalar_t>(),
            gain.data_ptr<double>(),
            n,
            lr);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scalar_update_2d_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor q,
    torch::Tensor impulse,
    torch::Tensor metric_q,
    torch::Tensor metric_impulse,
    torch::Tensor force_rms,
    torch::Tensor drive_energy,
    torch::Tensor surprise,
    torch::Tensor action,
    int64_t step,
    double lr,
    double weight_decay,
    double energy_power,
    double metric_dt,
    double metric_friction,
    double metric_coupling,
    double local_sharpness,
    double instant_metric_coupling,
    double instant_metric_power,
    double dt,
    double impulse_friction,
    double contact_gain,
    double restoring,
    double cubic,
    double reservoir_decay,
    double refractory_gain,
    double surprise_gain,
    double surprise_decay,
    double prediction_mix,
    double novelty_mix,
    double warmup_steps,
    double surprise_brake,
    double target_update_rms,
    double min_gain,
    double max_gain,
    double drive_clip,
    double spike_mix,
    double root_channel_mix,
    double direct_force_mix,
    double residual_feedback_mix,
    double eps) {
    TORCH_CHECK(param.is_cuda() && grad.is_cuda(), "scalar_update_2d tensors must be CUDA");
    TORCH_CHECK(param.dim() == 2 && grad.dim() == 2, "scalar_update_2d requires 2D tensors");
    TORCH_CHECK(param.is_contiguous() && grad.is_contiguous(), "scalar_update_2d requires contiguous tensors");
    TORCH_CHECK(param.scalar_type() == grad.scalar_type(), "scalar_update_2d dtype mismatch");
    TORCH_CHECK(param.scalar_type() == at::kFloat || param.scalar_type() == at::kDouble, "scalar_update_2d supports float32 and float64 params");
    TORCH_CHECK(param.sizes() == grad.sizes(), "scalar_update_2d shape mismatch");
    TORCH_CHECK(q.is_cuda() && impulse.is_cuda() && metric_q.is_cuda() && metric_impulse.is_cuda() && force_rms.is_cuda() &&
                    drive_energy.is_cuda() && surprise.is_cuda() && action.is_cuda(),
                "scalar_update_2d state must be CUDA");
    TORCH_CHECK(q.dim() == 0 && impulse.dim() == 0 && metric_q.dim() == 0 && metric_impulse.dim() == 0 && force_rms.dim() == 0 &&
                    drive_energy.dim() == 0 && surprise.dim() == 0 && action.dim() == 0,
                "scalar_update_2d state must be scalar tensors");
    TORCH_CHECK(q.scalar_type() == param.scalar_type() && impulse.scalar_type() == param.scalar_type() &&
                    metric_q.scalar_type() == param.scalar_type() && metric_impulse.scalar_type() == param.scalar_type() &&
                    force_rms.scalar_type() == param.scalar_type() && drive_energy.scalar_type() == param.scalar_type() &&
                    surprise.scalar_type() == param.scalar_type() && action.scalar_type() == param.scalar_type(),
                "scalar_update_2d state dtype mismatch");

    const c10::cuda::CUDAGuard device_guard(param.device());
    const int64_t rows = param.size(0);
    const int64_t cols = param.size(1);
    const int64_t n = param.numel();
    const int chunks = choose_chunks(n);
    const int blocks = std::min<int64_t>(65535, std::max<int64_t>(1, (n + kThreads - 1) / kThreads));
    auto partial1 = torch::empty({chunks}, param.options().dtype(torch::kFloat64));
    auto partial2 = torch::empty({chunks, 2}, param.options().dtype(torch::kFloat64));
    auto partial6 = torch::empty({chunks, 6}, param.options().dtype(torch::kFloat64));
    auto row_sums = torch::empty({rows}, param.options().dtype(torch::kFloat64));
    auto col_sums = torch::empty({cols}, param.options().dtype(torch::kFloat64));
    auto stats = torch::empty({16}, param.options().dtype(torch::kFloat64));
    auto tangent = torch::empty_like(param);
    auto drive = torch::empty_like(param);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "scalar_update_2d_cuda", [&] {
        scalar_force_partial_kernel<scalar_t><<<chunks, kThreads, kThreads * sizeof(double), stream>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            n,
            chunks,
            weight_decay,
            partial1.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_force_finalize_kernel<scalar_t><<<1, kThreads, kThreads * sizeof(double), stream>>>(
            partial1.data_ptr<double>(),
            chunks,
            n,
            step,
            force_rms.data_ptr<scalar_t>(),
            reservoir_decay,
            eps,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        C10_CUDA_CHECK(cudaMemsetAsync(row_sums.data_ptr<double>(), 0, static_cast<size_t>(rows) * sizeof(double), stream));
        C10_CUDA_CHECK(cudaMemsetAsync(col_sums.data_ptr<double>(), 0, static_cast<size_t>(cols) * sizeof(double), stream));
        scalar_axis_density_kernel<scalar_t><<<
            chunks,
            kThreads,
            4 * kThreads * sizeof(double),
            stream>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            row_sums.data_ptr<double>(),
            col_sums.data_ptr<double>(),
            rows,
            cols,
            chunks,
            stats.data_ptr<double>(),
            weight_decay,
            energy_power,
            instant_metric_power,
            eps,
            partial2.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_metric_finalize_kernel<scalar_t><<<1, kThreads, 4 * kThreads * sizeof(double), stream>>>(
            partial2.data_ptr<double>(),
            chunks,
            n,
            metric_q.data_ptr<scalar_t>(),
            metric_impulse.data_ptr<scalar_t>(),
            metric_dt,
            metric_friction,
            metric_coupling,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_pre_contact_partial_kernel<scalar_t><<<
            chunks,
            kThreads,
            4 * kThreads * sizeof(double),
            stream>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            row_sums.data_ptr<double>(),
            col_sums.data_ptr<double>(),
            tangent.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            rows,
            cols,
            chunks,
            stats.data_ptr<double>(),
            weight_decay,
            local_sharpness,
            instant_metric_coupling,
            instant_metric_power,
            eps,
            partial2.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_contact_update_kernel<scalar_t><<<1, kThreads, 4 * kThreads * sizeof(double), stream>>>(
            partial2.data_ptr<double>(),
            chunks,
            n,
            q.data_ptr<scalar_t>(),
            impulse.data_ptr<scalar_t>(),
            surprise.data_ptr<scalar_t>(),
            action.data_ptr<scalar_t>(),
            step,
            dt,
            impulse_friction,
            contact_gain,
            restoring,
            cubic,
            refractory_gain,
            surprise_gain,
            surprise_decay,
            surprise_brake,
            eps,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_drive_stats_partial_kernel<scalar_t><<<
            chunks,
            kThreads,
            6 * kThreads * sizeof(double),
            stream>>>(
            tangent.data_ptr<scalar_t>(),
            rows,
            cols,
            chunks,
            stats.data_ptr<double>(),
            drive_clip,
            eps,
            partial6.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_drive_stats_finalize_kernel<<<1, kThreads, 6 * kThreads * sizeof(double), stream>>>(
            partial6.data_ptr<double>(),
            chunks,
            n,
            step,
            prediction_mix,
            static_cast<int64_t>(warmup_steps),
            eps,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_shaped_partial_kernel<scalar_t><<<chunks, kThreads, kThreads * sizeof(double), stream>>>(
            tangent.data_ptr<scalar_t>(),
            rows,
            cols,
            chunks,
            stats.data_ptr<double>(),
            root_channel_mix,
            spike_mix,
            drive_clip,
            eps,
            partial1.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_one_stat_finalize_kernel<<<1, kThreads, kThreads * sizeof(double), stream>>>(
            partial1.data_ptr<double>(),
            chunks,
            n,
            eps,
            stats.data_ptr<double>(),
            10);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_core_partial_kernel<scalar_t><<<chunks, kThreads, kThreads * sizeof(double), stream>>>(
            tangent.data_ptr<scalar_t>(),
            rows,
            cols,
            chunks,
            stats.data_ptr<double>(),
            direct_force_mix,
            root_channel_mix,
            spike_mix,
            drive_clip,
            eps,
            partial1.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_one_stat_finalize_kernel<<<1, kThreads, kThreads * sizeof(double), stream>>>(
            partial1.data_ptr<double>(),
            chunks,
            n,
            eps,
            stats.data_ptr<double>(),
            11);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_drive_partial_kernel<scalar_t><<<chunks, kThreads, kThreads * sizeof(double), stream>>>(
            tangent.data_ptr<scalar_t>(),
            drive.data_ptr<scalar_t>(),
            rows,
            cols,
            chunks,
            stats.data_ptr<double>(),
            novelty_mix,
            direct_force_mix,
            residual_feedback_mix,
            root_channel_mix,
            spike_mix,
            drive_clip,
            eps,
            partial1.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_gain_finalize_kernel<scalar_t><<<1, kThreads, kThreads * sizeof(double), stream>>>(
            partial1.data_ptr<double>(),
            chunks,
            n,
            step,
            drive_energy.data_ptr<scalar_t>(),
            reservoir_decay,
            target_update_rms,
            min_gain,
            max_gain,
            eps,
            stats.data_ptr<double>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        scalar_param_apply_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
            param.data_ptr<scalar_t>(),
            drive.data_ptr<scalar_t>(),
            stats.data_ptr<double>(),
            n,
            lr);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
