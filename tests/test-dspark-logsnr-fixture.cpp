#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "../src/llama-model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static int failures = 0;

static void expect_true(bool value, const char * what) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

static const uint32_t N_EMBD          = 32;
static const uint32_t N_HEAD          = 2;
static const uint32_t N_EMBD_HEAD     = N_EMBD / N_HEAD;
static const uint32_t N_FF            = 64;
static const uint32_t N_LAYER         = 2;
static const uint32_t N_VOCAB         = 8;
static const uint32_t N_MARKOV_RANK   = 8;
static const uint32_t N_TARGET_LAYERS = 2;
static const uint32_t N_FREQ          = 128;

struct tensor_spec {
    std::string name;
    int64_t     ne0;
    int64_t     ne1; // 0 means 1D
};

static std::vector<tensor_spec> tensor_specs() {
    const int64_t n_embd_head = N_EMBD_HEAD;
    const int64_t n_embd_inp  = (int64_t) N_TARGET_LAYERS * N_EMBD;

    std::vector<tensor_spec> specs = {
        { "fc.weight",                     n_embd_inp,    N_EMBD },
        { "enc.output_norm.weight",         N_EMBD,        0 },
        { "output_norm.weight",             N_EMBD,        0 },
        { "markov_w1.weight",               N_MARKOV_RANK, N_VOCAB },
        { "markov_w2.weight",               N_MARKOV_RANK, N_VOCAB },
        { "log_snr_fc1.weight",             N_FREQ,        N_EMBD },
        { "log_snr_fc1.bias",               N_EMBD,        0 },
        { "log_snr_fc2.weight",             N_EMBD,        N_EMBD },
        { "log_snr_fc2.bias",               N_EMBD,        0 },
    };

    for (uint32_t i = 0; i < N_LAYER; ++i) {
        const std::string bid = "blk." + std::to_string(i) + ".";
        specs.push_back({ bid + "attn_norm.weight",     N_EMBD,          0 });
        specs.push_back({ bid + "attn_q.weight",       N_EMBD,          n_embd_head * N_HEAD });
        specs.push_back({ bid + "attn_k.weight",       N_EMBD,          n_embd_head * N_HEAD });
        specs.push_back({ bid + "attn_output.weight",   n_embd_head * N_HEAD, N_EMBD });
        specs.push_back({ bid + "attn_q_norm.weight",   n_embd_head,     0 });
        specs.push_back({ bid + "attn_k_norm.weight",   n_embd_head,     0 });
        specs.push_back({ bid + "ffn_norm.weight",     N_EMBD,          0 });
        specs.push_back({ bid + "ffn_gate.weight",     N_EMBD,          N_FF });
        specs.push_back({ bid + "ffn_down.weight",     N_FF,            N_EMBD });
        specs.push_back({ bid + "ffn_up.weight",       N_EMBD,          N_FF });
    }

    return specs;
}

static bool silent_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static bool write_fixture(const std::string & path) {
    gguf_context * gctx = gguf_init_empty();

    gguf_set_val_str (gctx, "general.architecture", "dflash");
    gguf_set_val_str (gctx, "general.name", "dspark log-snr fixture");
    gguf_set_val_str (gctx, "tokenizer.ggml.model", "no_vocab");
    gguf_set_val_u32 (gctx, "dflash.vocab_size", N_VOCAB);
    gguf_set_val_u32 (gctx, "dflash.context_length", 256);
    gguf_set_val_u32 (gctx, "dflash.embedding_length", N_EMBD);
    gguf_set_val_u32 (gctx, "dflash.block_count", N_LAYER);
    gguf_set_val_u32 (gctx, "dflash.feed_forward_length", N_FF);
    gguf_set_val_u32 (gctx, "dflash.attention.head_count", N_HEAD);
    gguf_set_val_f32 (gctx, "dflash.attention.layer_norm_rms_epsilon", 1.0e-5f);
    {
        const uint32_t target_layers[N_TARGET_LAYERS] = { 0, 1 };
        gguf_set_arr_data(gctx, "dflash.target_layers", GGUF_TYPE_UINT32, target_layers, N_TARGET_LAYERS);
    }
    gguf_set_val_bool(gctx, "dflash.log_snr_conditioning", true);
    gguf_set_val_f32 (gctx, "dflash.min_log_snr", -9.0f);
    gguf_set_val_f32 (gctx, "dflash.max_log_snr",  9.0f);

    const auto specs = tensor_specs();

    size_t data_bytes = 0;
    for (const auto & spec : specs) {
        data_bytes += (spec.ne1 > 0 ? (size_t) (spec.ne0 * spec.ne1) : (size_t) spec.ne0) * sizeof(float);
    }
    std::vector<uint8_t> data(data_bytes, 0);

    size_t offset = 0;
    for (const auto & spec : specs) {
        struct ggml_tensor t;
        std::memset(&t, 0, sizeof(t));

        t.type = GGML_TYPE_F32;
        t.ne[0] = spec.ne0;
        t.ne[1] = spec.ne1 > 0 ? spec.ne1 : 1;
        t.ne[2] = 1;
        t.ne[3] = 1;
        t.nb[0] = sizeof(float);
        t.nb[1] = t.nb[0] * (size_t) t.ne[0];
        t.nb[2] = t.nb[1] * (size_t) t.ne[1];
        t.nb[3] = t.nb[2];
        t.data = data.data() + offset;

        ggml_format_name(&t, "%s", spec.name.c_str());
        gguf_add_tensor(gctx, &t);

        offset += ggml_nbytes(&t);
    }

    const bool ok = gguf_write_to_file(gctx, path.c_str(), false);

    gguf_free(gctx);

    return ok;
}

static const ggml_tensor * find_tensor(const llama_model * model, const char * name) {
    const auto & tensors = llama_internal_get_tensor_map(model);
    for (const auto & kv : tensors) {
        if (kv.first == name) {
            return kv.second;
        }
    }
    return nullptr;
}

int main() {
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / "test-dspark-logsnr-fixture.gguf";
    const std::string path = tmp.string();

    if (!write_fixture(path)) {
        std::fprintf(stderr, "FAIL: could not write %s\n", path.c_str());
        return 1;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = silent_progress;

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "FAIL: the loader rejected the fixture with the log_snr tensors\n");
        std::filesystem::remove(tmp);
        return 1;
    }

    {
        char buf[64];
        int32_t ret = llama_model_meta_val_str(model, "dflash.log_snr_conditioning", buf, sizeof(buf));
        expect_true(ret >= 0, "log_snr_conditioning metadata present");
        expect_true(ret >= 0 && std::strcmp(buf, "true") == 0, "log_snr_conditioning is true");

        ret = llama_model_meta_val_str(model, "dflash.min_log_snr", buf, sizeof(buf));
        expect_true(ret >= 0 && std::fabs(std::strtof(buf, nullptr) + 9.0f) < 1.0e-6f, "min_log_snr is -9");

        ret = llama_model_meta_val_str(model, "dflash.max_log_snr", buf, sizeof(buf));
        expect_true(ret >= 0 && std::fabs(std::strtof(buf, nullptr) - 9.0f) < 1.0e-6f, "max_log_snr is 9");
    }

    {
        const ggml_tensor * t = find_tensor(model, "log_snr_fc1.weight");
        expect_true(t != nullptr, "log_snr_fc1.weight is loaded");
        if (t != nullptr) {
            expect_true(t->type == GGML_TYPE_F32, "log_snr_fc1.weight is F32");
            expect_true(t->ne[0] == N_FREQ && t->ne[1] == N_EMBD, "log_snr_fc1.weight shape is {128, 32}");
        }

        t = find_tensor(model, "log_snr_fc1.bias");
        expect_true(t != nullptr, "log_snr_fc1.bias is loaded");
        if (t != nullptr) {
            expect_true(t->ne[0] == N_EMBD, "log_snr_fc1.bias shape is {32}");
        }

        t = find_tensor(model, "log_snr_fc2.weight");
        expect_true(t != nullptr, "log_snr_fc2.weight is loaded");
        if (t != nullptr) {
            expect_true(t->ne[0] == N_EMBD && t->ne[1] == N_EMBD, "log_snr_fc2.weight shape is {32, 32}");
        }

        t = find_tensor(model, "log_snr_fc2.bias");
        expect_true(t != nullptr, "log_snr_fc2.bias is loaded");
        if (t != nullptr) {
            expect_true(t->ne[0] == N_EMBD, "log_snr_fc2.bias shape is {32}");
        }

        expect_true(find_tensor(model, "fc.weight") != nullptr, "fc.weight is loaded");
        expect_true(find_tensor(model, "markov_w1.weight") != nullptr, "markov_w1.weight is loaded");
    }

    llama_model_free(model);
    std::filesystem::remove(tmp);

    if (failures != 0) {
        std::fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }

    std::printf("OK\n");
    return 0;
}
