#include <torch/extension.h>
#include <vector>
#include <stdexcept>

#include "config.cuh"
#include "persistent_layer.cuh"
#include "reference.cuh"
#include "persistent_multi_layer.cuh"

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

static void check_half(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dtype() == torch::kFloat16, name, " must be float16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
}

static void check_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
}

static ModelConfig make_config(int hidden_dim, int intermediate_dim,
                                int num_heads, int num_kv_heads,
                                int num_layers = 1) {
    TORCH_CHECK(hidden_dim % num_heads == 0,
                "hidden_dim must be divisible by num_heads");
    ModelConfig cfg;
    cfg.num_layers       = num_layers;
    cfg.hidden_dim       = hidden_dim;
    cfg.intermediate_dim = intermediate_dim;
    cfg.num_heads        = num_heads;
    cfg.num_kv_heads     = num_kv_heads;
    cfg.head_dim         = hidden_dim / num_heads;
    return cfg;
}

// Validate and extract LayerWeights from individual weight tensors.
static LayerWeights extract_weights(
        const torch::Tensor& q_proj,
        const torch::Tensor& k_proj,
        const torch::Tensor& v_proj,
        const torch::Tensor& o_proj,
        const torch::Tensor& gate_proj,
        const torch::Tensor& up_proj,
        const torch::Tensor& down_proj,
        const torch::Tensor& attn_norm,
        const torch::Tensor& ffn_norm,
        const ModelConfig& cfg) {

    check_half(q_proj,    "q_proj");
    check_half(k_proj,    "k_proj");
    check_half(v_proj,    "v_proj");
    check_half(o_proj,    "o_proj");
    check_half(gate_proj, "gate_proj");
    check_half(up_proj,   "up_proj");
    check_half(down_proj, "down_proj");
    check_float(attn_norm, "attn_norm");
    check_float(ffn_norm,  "ffn_norm");

    int q_cols  = cfg.num_heads    * cfg.head_dim;
    int kv_cols = cfg.num_kv_heads * cfg.head_dim;

    TORCH_CHECK(q_proj.numel()    == (int64_t)cfg.hidden_dim * q_cols,
                "q_proj shape mismatch");
    TORCH_CHECK(k_proj.numel()    == (int64_t)cfg.hidden_dim * kv_cols,
                "k_proj shape mismatch");
    TORCH_CHECK(v_proj.numel()    == (int64_t)cfg.hidden_dim * kv_cols,
                "v_proj shape mismatch");
    TORCH_CHECK(o_proj.numel()    == (int64_t)q_cols * cfg.hidden_dim,
                "o_proj shape mismatch");
    TORCH_CHECK(gate_proj.numel() == (int64_t)cfg.hidden_dim * cfg.intermediate_dim,
                "gate_proj shape mismatch");
    TORCH_CHECK(up_proj.numel()   == (int64_t)cfg.hidden_dim * cfg.intermediate_dim,
                "up_proj shape mismatch");
    TORCH_CHECK(down_proj.numel() == (int64_t)cfg.intermediate_dim * cfg.hidden_dim,
                "down_proj shape mismatch");
    TORCH_CHECK(attn_norm.numel() == cfg.hidden_dim, "attn_norm shape mismatch");
    TORCH_CHECK(ffn_norm.numel()  == cfg.hidden_dim, "ffn_norm shape mismatch");

    LayerWeights lw;
    lw.q_proj           = reinterpret_cast<const __half*>(q_proj.data_ptr<at::Half>());
    lw.k_proj           = reinterpret_cast<const __half*>(k_proj.data_ptr<at::Half>());
    lw.v_proj           = reinterpret_cast<const __half*>(v_proj.data_ptr<at::Half>());
    lw.o_proj           = reinterpret_cast<const __half*>(o_proj.data_ptr<at::Half>());
    lw.gate_proj        = reinterpret_cast<const __half*>(gate_proj.data_ptr<at::Half>());
    lw.up_proj          = reinterpret_cast<const __half*>(up_proj.data_ptr<at::Half>());
    lw.down_proj        = reinterpret_cast<const __half*>(down_proj.data_ptr<at::Half>());
    lw.attn_norm_weight = attn_norm.data_ptr<float>();
    lw.ffn_norm_weight  = ffn_norm.data_ptr<float>();
    return lw;
}

// ---------------------------------------------------------------------------
// persistent_layer_forward
// ---------------------------------------------------------------------------

torch::Tensor persistent_layer_forward(
        const torch::Tensor& hidden,
        torch::Tensor& kv_k,
        torch::Tensor& kv_v,
        int kv_seq_len,
        const torch::Tensor& q_proj,
        const torch::Tensor& k_proj,
        const torch::Tensor& v_proj,
        const torch::Tensor& o_proj,
        const torch::Tensor& gate_proj,
        const torch::Tensor& up_proj,
        const torch::Tensor& down_proj,
        const torch::Tensor& attn_norm,
        const torch::Tensor& ffn_norm,
        int hidden_dim, int intermediate_dim,
        int num_heads, int num_kv_heads) {

    check_half(hidden, "hidden");
    TORCH_CHECK(hidden.numel() == hidden_dim, "hidden dim mismatch");

    ModelConfig cfg = make_config(hidden_dim, intermediate_dim, num_heads, num_kv_heads);
    LayerWeights lw = extract_weights(q_proj, k_proj, v_proj, o_proj,
                                       gate_proj, up_proj, down_proj,
                                       attn_norm, ffn_norm, cfg);

    TORCH_CHECK(kv_k.is_contiguous() && kv_k.is_cuda() && kv_k.dtype() == torch::kFloat16,
                "kv_k must be contiguous fp16 CUDA tensor");
    TORCH_CHECK(kv_v.is_contiguous() && kv_v.is_cuda() && kv_v.dtype() == torch::kFloat16,
                "kv_v must be contiguous fp16 CUDA tensor");

    int max_seq = (int)(kv_k.numel() / (cfg.num_kv_heads * cfg.head_dim));

    KVCache kvc;
    kvc.k           = reinterpret_cast<__half*>(kv_k.data_ptr<at::Half>());
    kvc.v           = reinterpret_cast<__half*>(kv_v.data_ptr<at::Half>());
    kvc.max_seq_len = max_seq;
    kvc.kv_seq_len  = kv_seq_len;

    auto out = torch::empty({hidden_dim}, hidden.options());
    const __half* hin  = reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>());
    __half*       hout = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    launch_persistent_layer(hin, hout, lw, cfg, kvc);
    cudaDeviceSynchronize();
    return out;
}

// ---------------------------------------------------------------------------
// reference_layer_forward
// ---------------------------------------------------------------------------

torch::Tensor reference_layer_forward(
        const torch::Tensor& hidden,
        torch::Tensor& kv_k,
        torch::Tensor& kv_v,
        int kv_seq_len,
        const torch::Tensor& q_proj,
        const torch::Tensor& k_proj,
        const torch::Tensor& v_proj,
        const torch::Tensor& o_proj,
        const torch::Tensor& gate_proj,
        const torch::Tensor& up_proj,
        const torch::Tensor& down_proj,
        const torch::Tensor& attn_norm,
        const torch::Tensor& ffn_norm,
        int hidden_dim, int intermediate_dim,
        int num_heads, int num_kv_heads) {

    check_half(hidden, "hidden");
    TORCH_CHECK(hidden.numel() == hidden_dim, "hidden dim mismatch");

    ModelConfig cfg = make_config(hidden_dim, intermediate_dim, num_heads, num_kv_heads);
    LayerWeights lw = extract_weights(q_proj, k_proj, v_proj, o_proj,
                                       gate_proj, up_proj, down_proj,
                                       attn_norm, ffn_norm, cfg);

    TORCH_CHECK(kv_k.is_contiguous() && kv_k.is_cuda() && kv_k.dtype() == torch::kFloat16,
                "kv_k must be contiguous fp16 CUDA tensor");
    TORCH_CHECK(kv_v.is_contiguous() && kv_v.is_cuda() && kv_v.dtype() == torch::kFloat16,
                "kv_v must be contiguous fp16 CUDA tensor");

    int max_seq = (int)(kv_k.numel() / (cfg.num_kv_heads * cfg.head_dim));

    KVCache kvc;
    kvc.k           = reinterpret_cast<__half*>(kv_k.data_ptr<at::Half>());
    kvc.v           = reinterpret_cast<__half*>(kv_v.data_ptr<at::Half>());
    kvc.max_seq_len = max_seq;
    kvc.kv_seq_len  = kv_seq_len;

    auto out = torch::empty({hidden_dim}, hidden.options());
    const __half* hin  = reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>());
    __half*       hout = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    launch_reference_layer(hin, hout, lw, cfg, kvc);
    return out;
}

// ---------------------------------------------------------------------------
// persistent_multi_layer_forward
// ---------------------------------------------------------------------------

torch::Tensor persistent_multi_layer_forward(
        const torch::Tensor& hidden,
        std::vector<torch::Tensor>& all_kv_k,
        std::vector<torch::Tensor>& all_kv_v,
        const std::vector<int>& kv_seq_lens,
        const std::vector<torch::Tensor>& q_projs,
        const std::vector<torch::Tensor>& k_projs,
        const std::vector<torch::Tensor>& v_projs,
        const std::vector<torch::Tensor>& o_projs,
        const std::vector<torch::Tensor>& gate_projs,
        const std::vector<torch::Tensor>& up_projs,
        const std::vector<torch::Tensor>& down_projs,
        const std::vector<torch::Tensor>& attn_norms,
        const std::vector<torch::Tensor>& ffn_norms,
        int hidden_dim, int intermediate_dim,
        int num_heads, int num_kv_heads,
        int num_layers) {

    check_half(hidden, "hidden");
    TORCH_CHECK((int)q_projs.size() == num_layers, "wrong number of q_proj tensors");
    TORCH_CHECK((int)all_kv_k.size() == num_layers, "wrong number of kv_k tensors");

    ModelConfig cfg = make_config(hidden_dim, intermediate_dim, num_heads, num_kv_heads, num_layers);

    std::vector<LayerWeights> lws(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        lws[l] = extract_weights(q_projs[l], k_projs[l], v_projs[l], o_projs[l],
                                  gate_projs[l], up_projs[l], down_projs[l],
                                  attn_norms[l], ffn_norms[l], cfg);
    }

    std::vector<KVCache> kvcs(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        auto& kk = all_kv_k[l];
        auto& kv = all_kv_v[l];
        TORCH_CHECK(kk.is_contiguous() && kk.is_cuda() && kk.dtype() == torch::kFloat16,
                    "kv_k must be contiguous fp16 CUDA tensor");
        TORCH_CHECK(kv.is_contiguous() && kv.is_cuda() && kv.dtype() == torch::kFloat16,
                    "kv_v must be contiguous fp16 CUDA tensor");
        int max_seq = (int)(kk.numel() / (cfg.num_kv_heads * cfg.head_dim));
        kvcs[l].k           = reinterpret_cast<__half*>(kk.data_ptr<at::Half>());
        kvcs[l].v           = reinterpret_cast<__half*>(kv.data_ptr<at::Half>());
        kvcs[l].max_seq_len = max_seq;
        kvcs[l].kv_seq_len  = kv_seq_lens[l];
    }

    LayerWeightsArray lwa;
    lwa.layers    = lws.data();
    lwa.num_layers = num_layers;

    auto out = torch::empty({hidden_dim}, hidden.options());
    const __half* hin  = reinterpret_cast<const __half*>(hidden.data_ptr<at::Half>());
    __half*       hout = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    launch_persistent_multi_layer(hin, hout, lwa, cfg, kvcs.data());
    cudaDeviceSynchronize();
    return out;
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persistent_layer_forward", &persistent_layer_forward,
          "Single-layer persistent kernel forward");
    m.def("reference_layer_forward", &reference_layer_forward,
          "Single-layer multi-launch reference forward");
    m.def("persistent_multi_layer_forward", &persistent_multi_layer_forward,
          "Multi-layer persistent kernel forward (hidden state smem-resident across layers)");
}
