#include <ATen/ATen.h>

#include "torch_npu/csrc/core/npu/NPUException.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::prefetch_memory_(at::Tensor& self, bool non_blocking)
{
    AT_ASSERT(
        torch_npu::utils::is_npu(self),
        "prefetch_memory_ only support npu tensor", OPS_ERROR(ErrCode::PARAM));

    c10_npu::NPUGuard guard(self.device());

    auto desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;

    int mem_size = 0;
    if (FormatHelper::IsPadded(&self)) {
        AT_ASSERT(self.storage_offset() == 0, OPS_ERROR(ErrCode::VALUE));
        mem_size = c10::multiply_integers(desc.storage_sizes_);
    } else {
        auto element_num = c10::multiply_integers(self.sizes());
        auto storage_num = c10::multiply_integers(desc.storage_sizes_);
        mem_size = (element_num > storage_num) ? storage_num : element_num;
    }

    int64_t nbytes = mem_size * self.itemsize();
    int device_id = self.device().index();

    NPU_CHECK_ERROR(CalcuOpUtil::LaunchAsyncPrefetchTaskWithModeSwitch(
        self.data_ptr(), nbytes, device_id, 0));

    if (!non_blocking) {
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
    return self;
}

} // namespace native
} // namespace at_npu
