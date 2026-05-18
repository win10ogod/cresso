#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor basis_stats_cuda(torch::Tensor freqs, std::vector<int64_t> sizes);
torch::Tensor basis_axis_cache_cuda(torch::Tensor freqs, std::vector<int64_t> sizes);
torch::Tensor basis_stats_with_cache_cuda(torch::Tensor freqs, std::vector<int64_t> sizes, torch::Tensor axis_cache);
std::vector<torch::Tensor> basis_project_cuda(torch::Tensor x, torch::Tensor freqs, std::vector<int64_t> sizes);
std::vector<torch::Tensor> basis_project_with_stats_cuda(
    torch::Tensor x,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor stats);
std::vector<torch::Tensor> basis_project_with_cache_cuda(
    torch::Tensor x,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor stats,
    torch::Tensor axis_cache);
torch::Tensor basis_reconstruct_cuda(
    torch::Tensor cos_coeff,
    torch::Tensor sin_coeff,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor gates);
torch::Tensor basis_reconstruct_with_stats_cuda(
    torch::Tensor cos_coeff,
    torch::Tensor sin_coeff,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor gates,
    torch::Tensor stats);
torch::Tensor basis_reconstruct_with_cache_cuda(
    torch::Tensor cos_coeff,
    torch::Tensor sin_coeff,
    torch::Tensor freqs,
    std::vector<int64_t> sizes,
    torch::Tensor gates,
    torch::Tensor stats,
    torch::Tensor axis_cache);
torch::Tensor hash_reduce_mean_cuda(torch::Tensor x, int64_t bins, int64_t salt);
torch::Tensor hash_gather_cuda(torch::Tensor values, std::vector<int64_t> sizes, int64_t salt);
void metric_update_cuda(
    torch::Tensor metric_cos,
    torch::Tensor metric_sin,
    torch::Tensor metric_p_cos,
    torch::Tensor metric_p_sin,
    torch::Tensor mcos_port,
    torch::Tensor msin_port,
    torch::Tensor omega,
    double metric_dt,
    double metric_friction);
void confidence_update_cuda(
    torch::Tensor confidence,
    torch::Tensor pcos_port,
    torch::Tensor psin_port,
    torch::Tensor qcos,
    torch::Tensor qsin,
    double confidence_decay,
    double eps);
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
    double cubic);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("basis_stats", &basis_stats_cuda, "CRESSO5 basis normalization stats (CUDA)");
    m.def("basis_axis_cache", &basis_axis_cache_cuda, "CRESSO5 compact axis basis cache (CUDA)");
    m.def("basis_stats_with_cache", &basis_stats_with_cache_cuda, "CRESSO5 basis stats from axis cache (CUDA)");
    m.def("basis_project", &basis_project_cuda, "CRESSO5 fused basis projection (CUDA)");
    m.def("basis_project_with_stats", &basis_project_with_stats_cuda, "CRESSO5 fused basis projection with stats (CUDA)");
    m.def("basis_project_with_cache", &basis_project_with_cache_cuda, "CRESSO5 fused basis projection with axis cache (CUDA)");
    m.def("basis_reconstruct", &basis_reconstruct_cuda, "CRESSO5 fused basis reconstruction (CUDA)");
    m.def(
        "basis_reconstruct_with_stats",
        &basis_reconstruct_with_stats_cuda,
        "CRESSO5 fused basis reconstruction with stats (CUDA)");
    m.def(
        "basis_reconstruct_with_cache",
        &basis_reconstruct_with_cache_cuda,
        "CRESSO5 fused basis reconstruction with axis cache (CUDA)");
    m.def("hash_reduce_mean", &hash_reduce_mean_cuda, "CRESSO5 coordinate hash reduce mean (CUDA)");
    m.def("hash_gather", &hash_gather_cuda, "CRESSO5 coordinate hash gather (CUDA)");
    m.def("metric_update", &metric_update_cuda, "CRESSO5 fused rank metric oscillator update (CUDA)");
    m.def("confidence_update", &confidence_update_cuda, "CRESSO5 fused rank confidence update (CUDA)");
    m.def("contact_update_pair", &contact_update_pair_cuda, "CRESSO5 fused rank contact oscillator update (CUDA)");
}
