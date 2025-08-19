#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>

#include <onnxruntime_cxx_api.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include "xrt_dma.h"
#include "xrt_pynq.h"
#include "xrt_ps.h"
#include "xrt_mmio.hpp"
#include <nlohmann/json.hpp>
#include "base64.h"

namespace spec {
using in_t  = uint8_t;
using out_t = int8_t;
constexpr int N_MAX = 10;
constexpr int IN_H = 7, IN_W = 7, IN_C = 1280;
constexpr int OUT_H = 1, OUT_W = 1, OUT_C = 1280;
constexpr size_t BYTES_PER_IMAGE_IN  = size_t(IN_H) * IN_W * IN_C * sizeof(in_t);
constexpr size_t BYTES_PER_IMAGE_OUT = size_t(OUT_H) * OUT_W * OUT_C * sizeof(out_t);
constexpr off_t MM2S_OFF = 0x0000;
constexpr off_t S2MM_OFF = 0x1000;
} // namespace spec

class FpgaRunner {
public:
  static FpgaRunner& instance() {
    static FpgaRunner inst;
    return inst;
  }

  void load_bitstream(const std::string &bit, const std::string &hwh,
                      float freq_MHz) {
    
    // Create the Overlay directory if it doesn't exist
    std::string overlay_dir = "Overlay";
    if (std::system(("mkdir -p " + overlay_dir).c_str()) != 0) {
      throw std::runtime_error("Failed to create Overlay directory");
    }

    // Write the bitstream and HWH files
    std::ofstream bit_file(overlay_dir + "/design.bit", std::ios::binary);
    if (!bit_file) throw std::runtime_error("Failed to open bitstream file");
    bit_file.write(bit.data(), bit.size());
    bit_file.close();

    std::ofstream hwh_file(overlay_dir + "/design.hwh", std::ios::out);
    if (!hwh_file) throw std::runtime_error("Failed to open HWH file");
    hwh_file.write(hwh.data(), hwh.size());
    hwh_file.close();

    // Program the PL through PYNQ.
    program_with_pynq_cli_or_throw("pynq_program.py",
                                   "Overlay/design.bit");

    // Map AXI-Lite window
    mmio_ = map_axil_window(0xA0000000, 0x10000);

    // Set PL clock
    float freq_achieved = set_pl_from_iopll(ZynqPllIndex::PL0, freq_MHz, 1000.0);

    // Create DMA engines
    tx_.emplace(mmio_.regs, spec::MM2S_OFF, in_bo_);
    rx_.emplace(mmio_.regs, spec::S2MM_OFF, dev_, out_bo_,
                spec::BYTES_PER_IMAGE_OUT, spec::N_MAX);
  }

  void run(const void* in_host, void* out_host, size_t batch) {

    const size_t in_bytes  = batch * spec::BYTES_PER_IMAGE_IN;
    const size_t out_bytes = batch * spec::BYTES_PER_IMAGE_OUT;

    std::lock_guard<std::mutex> lock(mtx_);

    // Copy the input buffer inside the CMA.
    std::memcpy(in_ptr_, in_host, in_bytes);

    // Sync the input buffer to the device memory.
    in_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE, in_bytes, 0);

    // Initialize batch S2MM DMA's block descriptors and start the transfer.
    rx_->transfer(int(batch));

    // Start the MM2S transfer.
    tx_->transfer(in_bytes, 0);

    // Wait for the MM2S transfer to complete.
    if (!tx_->wait_done(20)) {
      throw std::runtime_error("MM2S timeout");
    }

    // Wait for the S2MM transfer to complete.
    if (!rx_->wait_done(20, batch)) {
      throw std::runtime_error("S2MM timeout");
    }

    // Sync the output buffer to the host memory.
    out_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE, out_bytes, 0);

    // Copy the output buffer from the CMA to the host memory.
    std::memcpy(out_host, out_ptr_, out_bytes);
  }

private:
  FpgaRunner()
      : dev_(0), in_bo_(dev_, spec::N_MAX * spec::BYTES_PER_IMAGE_IN, 0, 0),
        out_bo_(dev_, spec::N_MAX * spec::BYTES_PER_IMAGE_OUT, 0, 0) {

    // Map buffers once
    in_ptr_ = static_cast<spec::in_t *>(in_bo_.map());
    out_ptr_ = static_cast<spec::out_t *>(out_bo_.map());
  }

  ~FpgaRunner() = default;

  xrt::device dev_;
  xrt::bo in_bo_;
  xrt::bo out_bo_;
  spec::in_t* in_ptr_  = nullptr;
  spec::out_t* out_ptr_ = nullptr;

  std::optional<Mm2sSimple> tx_;
  std::optional<S2mmSG> rx_;

  Mmio mmio_;
  std::mutex mtx_;
};

struct Nn2FpgaPartitionKernel {
  Nn2FpgaPartitionKernel(const OrtApi &api, const OrtKernelInfo *info) {
    Ort::ConstKernelInfo kinfo(info);
    const std::string pkg_json =
        kinfo.GetAttribute<std::string>("accelerator_package");

    nlohmann::json pkg = nlohmann::json::parse(pkg_json);
    const std::string bit_b64 = pkg.at("bitstream_b64").get<std::string>();
    const std::string hwh_b64 = pkg.at("hwh_b64").get<std::string>();
    const float freq_MHz = std::stof(pkg.at("frequency").get<std::string>());

    const std::string bit = base64_decode(bit_b64);
    const std::string hwh = base64_decode(hwh_b64);

    FpgaRunner::instance().load_bitstream(bit, hwh, freq_MHz);
  }

  void Compute(OrtKernelContext *ctx) {
    Ort::KernelContext kctx{ctx};

    // Input
    Ort::ConstValue in{kctx.GetInput(0)};
    Ort::TensorTypeAndShapeInfo info = in.GetTensorTypeAndShapeInfo();
    const auto in_type = info.GetElementType();
    std::vector<int64_t> in_shape = info.GetShape();

    if (in_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
      ORT_CXX_API_THROW("nn2fpgaPartition expects uint8 input",
                        ORT_INVALID_ARGUMENT);
    if (in_shape.empty())
      ORT_CXX_API_THROW("Input must be at least 1D (batch).",
                        ORT_INVALID_ARGUMENT);

    const int64_t batch64 = in_shape[0];
    if (batch64 <= 0 || batch64 > spec::N_MAX)
      ORT_CXX_API_THROW("Batch exceeds compiled N_MAX.", ORT_INVALID_ARGUMENT);

    const size_t batch = static_cast<size_t>(batch64);
    const int64_t batch_i64 = static_cast<int64_t>(batch);

    std::vector<int64_t> out_shape{batch_i64, spec::OUT_H, spec::OUT_W,
                                   spec::OUT_C};

    // Output (UnownedValue in newer ORT)
    auto out = kctx.GetOutput(0, out_shape.data(), out_shape.size());

    // Raw buffers
    const auto *in_ptr_host = in.GetTensorData<spec::in_t>();     // uint8_t
    auto *out_ptr_host = out.GetTensorMutableData<spec::out_t>(); // int8_t

    FpgaRunner::instance().run(in_ptr_host, out_ptr_host, batch);
  }
};

// ====== Op wrapper ======
struct Nn2FpgaPartitionOp
    : Ort::CustomOpBase<Nn2FpgaPartitionOp, Nn2FpgaPartitionKernel> {

  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return new Nn2FpgaPartitionKernel(api, info);
  }
  const char *GetName() const { return "nn2fpgaPartition"; }
  const char *GetExecutionProviderType() const {
    return "CPUExecutionProvider";
  }
  size_t GetInputTypeCount() const { return 1; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  }
};

// ====== Register ======
extern "C" OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
  const OrtApi *api = api_base->GetApi(ORT_API_VERSION);

#ifdef ORT_API_MANUAL_INIT
  // Initialize the C++ wrappers before any Ort:: usage
  Ort::InitApi(api);
#endif

  try {
    Ort::CustomOpDomain domain{"ai.onnx.contrib"};
    static Nn2FpgaPartitionOp c_op;
    domain.Add(&c_op);

    // Hand ownership of the raw handle to ORT; prevent RAII from freeing it.
    Ort::ThrowOnError(api->AddCustomOpDomain(options, domain));
    domain.release();

    return nullptr;
  } catch (const Ort::Exception &e) {
    return api->CreateStatus(e.GetOrtErrorCode(), e.what());
  } catch (const std::exception &e) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
  }
}
