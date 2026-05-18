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
