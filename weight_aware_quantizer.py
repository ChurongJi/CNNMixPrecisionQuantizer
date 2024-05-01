import copy
import torch
import torch.nn as nn
from model import QuantizedResNet18
import numpy as np
from resnet18 import BasicBlock, ResNet
from tqdm import tqdm
from utils import calibrate_model


class WeightAwareQuantizer():
    def __init__(self, cutoff_low_end = 2, cutoff_high_start = 98, enable_cutoff_search=True):
        """
        {module_name:
        [ # out channel
            [ # in channel
            (kernel_idx - np.uint16, idx_x - np.uint8, idx_y - np.uint8, fp_32_val - float32),
            (xxx), ...
            ],
            [...], ...
        ],
        module_name_2:
        [...],
        ...
        }

        for linear channel, module_name: [(x_idx - np.uint16, y_idx - np.uint16, fp32_val), (...), ...]
        """
        print("-----version 2--------")
        self.fp32_dict = {}
        # [0% - cutoff_low_end%] and [cutoff_high_start% - 100%] will be considered as candidate quantiles
        self.cutoff_low_end = cutoff_low_end
        self.cutoff_high_start = cutoff_high_start
        self.enable_cutoff_search = enable_cutoff_search

    def __find_quantize_range(self, weight):
        out_channel_num = weight.shape[0]
        cutoffs = torch.zeros((out_channel_num, 2))
        for i in range(out_channel_num):
            params = weight[i]
            percentiles = np.percentile(params, range(101))
            best_inclusive_score = 0
            best_start = 0
            best_end = 100
            for start in range(self.cutoff_low_end):
                for end in range(self.cutoff_high_start + 1, 101):
                    delta_count = end - self.cutoff_high_start + self.cutoff_low_end - start
                    delta_range = percentiles[end] - percentiles[self.cutoff_high_start] + percentiles[self.cutoff_low_end] - percentiles[start]
                    score = delta_count / delta_range
                    if score > best_inclusive_score:
                        best_inclusive_score = score
                        best_start = start
                        best_end = end
            cutoffs[i][0] = best_start
            cutoffs[i][1] = best_end
        return cutoffs / 100.0

    def __preprocess_conv_module(self, module_name, module):
        if "downsample" in module_name:
            return

        # strip tail .0 to stay consistent with quantized model's module names
        if module_name[-2:] == ".0":
            module_name = module_name[:-2]

        param = module.weight
        x_dim, y_dim = param.shape[-2:]
        kernel_size = x_dim * y_dim
        self.fp32_dict[module_name] = []
        channel_quantiles = []
        if self.enable_cutoff_search:
          channel_quantiles = self.__find_quantize_range(param.data)

        for i in tqdm(range(param.shape[0])):
            in_channel_change_list = []

            if self.enable_cutoff_search:
              quantiles = channel_quantiles[i]
            else:
              quantiles = torch.Tensor([self.cutoff_low_end, self.cutoff_high_start]) / 100.0
            
            cur_in_channel_vals = param[i].flatten()
            # interpolation for quantile is 'linear' by linear
            # use 'higher'/'lower' if want to get quantile with exact number of tensor
            quantile_vals = torch.quantile(cur_in_channel_vals, quantiles, dim=0)
            small_val_idxes = torch.nonzero(cur_in_channel_vals < quantile_vals[0]).squeeze()
            large_val_idxes = torch.nonzero(cur_in_channel_vals > quantile_vals[1]).squeeze()
            mean_val = torch.mean(cur_in_channel_vals)
            extreme_idxes = torch.cat((small_val_idxes, large_val_idxes), 0)
            for idx in extreme_idxes:
                kernel_idx = idx // kernel_size
                x_idx = (idx % kernel_size) // y_dim
                y_idx = idx % y_dim
                # use numpy type for uint8 and uint16 to save storage, python does not have those dtypes
                in_channel_change_list.append((np.uint16(kernel_idx), np.uint8(x_idx), np.uint8(y_idx), cur_in_channel_vals[idx].data.item()))
                # change param data value in original model to mean val in in_channel
                # module is ref - change here directly reflects in model modules
                with torch.no_grad():
                    module.weight[i, kernel_idx, x_idx, y_idx] = mean_val
            self.fp32_dict[module_name].append(in_channel_change_list)
        print("successfully extract extreme value out from ",module_name, " and replace with per in channel mean vals")


    def __preprocess_linear_module(self, module_name, module):
        param = module.weight
        in_dim = param.shape[1]
        self.fp32_dict[module_name] = []
        if self.enable_cutoff_search:
          quantiles = self.__find_quantize_range(param.data.unsqueeze(0))[0]
        else:
          quantiles = torch.Tensor([self.cutoff_low_end, self.cutoff_high_start]) / 100.0
        
        cur_in_channel_vals = param.flatten()
        quantile_vals = torch.quantile(cur_in_channel_vals, quantiles, dim=0)
        small_val_idxes = torch.nonzero(cur_in_channel_vals < quantile_vals[0]).squeeze()
        large_val_idxes = torch.nonzero(cur_in_channel_vals > quantile_vals[1]).squeeze()
        mean_val = torch.mean(cur_in_channel_vals)
        for idx in torch.cat((small_val_idxes, large_val_idxes), 0):
            x_idx = idx // in_dim
            y_idx = idx % in_dim
            # use numpy type for uint16 to save storage, python does not have those dtypes
            self.fp32_dict[module_name].append((np.uint16(x_idx), np.uint16(y_idx), cur_in_channel_vals[idx].data.item()))
            with torch.no_grad():
                module.weight[x_idx, y_idx] = mean_val
        print("successfully extract extreme value out from ",module_name, " and replace with per in channel mean vals")

    def process_fuse_model(self, fused_model):
        fused_model_to_process = copy.deepcopy(fused_model)
        for module_name, module in fused_model_to_process.named_modules():
            # print(module_name)
            if isinstance(module, torch.nn.Conv2d):
                print("Conv2d with name: ", module_name)
                self.__preprocess_conv_module(module_name, module)
            if isinstance(module, torch.nn.Linear):
                print("Linear with name: ", module_name)
                self.__preprocess_linear_module(module_name, module)
        print("Finish process fused model")
        return fused_model_to_process
    
    def optimzied_quantize_with_x86_default(self, processed_fused_model, train_loader, device):
        processed_quantized_model = QuantizedResNet18(copy.deepcopy(processed_fused_model))
        quantization_config = torch.ao.quantization.get_default_qconfig("x86")
        processed_quantized_model.qconfig = quantization_config
        torch.ao.quantization.prepare(processed_quantized_model, inplace=True)
        calibrate_model(model=processed_quantized_model, loader=train_loader, device=device)
        processed_quantized_model = torch.ao.quantization.convert(processed_quantized_model, inplace=True)
        return processed_quantized_model
    
    def optimzied_quantize(self, processed_fused_model, train_loader, device):
        processed_quantized_model = QuantizedResNet18(copy.deepcopy(processed_fused_model))
        is_reduce_range = not torch.cpu._is_cpu_support_vnni()
        quantization_config = torch.ao.quantization.QConfig(
            activation=torch.quantization.HistogramObserver.with_args(dtype=torch.quint8, reduce_range=is_reduce_range),
            weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range=is_reduce_range))
        processed_quantized_model.qconfig = quantization_config
        torch.ao.quantization.prepare(processed_quantized_model, inplace=True)
        calibrate_model(model=processed_quantized_model, loader=train_loader, device=device)
        processed_quantized_model = torch.ao.quantization.convert(processed_quantized_model, inplace=True)
        return processed_quantized_model

    def __recover_conv_from_fp32_dict(self, conv_module_name, conv_weight):
        out_channel_change_list = self.fp32_dict[conv_module_name]
        for out_idx in range(len(out_channel_change_list)):
            in_channel_change_list = out_channel_change_list[out_idx]
            with torch.no_grad():
                for kernel_idx, x_idx, y_idx, ori_val in in_channel_change_list:
                    conv_weight[out_idx, kernel_idx, x_idx, y_idx] = ori_val
        return conv_weight

    def __recover_linear_from_fp32_dict(self, linear_module_name, linear_weight):
        linear_change_list = self.fp32_dict[linear_module_name]
        for x_idx, y_idx, ori_val in linear_change_list:
            with torch.no_grad():
                linear_weight[x_idx, y_idx] = ori_val
        return linear_weight
    
    def mixed_precision_inference(self, processed_quantized_model, x):
        out = processed_quantized_model.quant(x)
        skip_add_x = x
        for module_name, module in processed_quantized_model.model_fp32.named_modules():
            # print(module_name)
            if isinstance(module, torch.ao.nn.intrinsic.ConvReLU2d) or isinstance(module, ResNet):
                continue
            elif isinstance(module, BasicBlock):
                skip_add_x = out
            elif isinstance(module, torch.nn.Sequential):
                if "downsample" in module_name:
                    skip_add_x = module(skip_add_x)
            elif isinstance(module, torch.ao.nn.quantized.QFunctional):
                out = module.add(skip_add_x, out)
            elif isinstance(module, torch.ao.nn.quantized.Linear):
                out = torch.flatten(out, 1)
                out = processed_quantized_model.dequant(out)
                tmp_linear = torch.nn.Linear(in_features = module.in_features,
                                            out_features = module.out_features,
                                            bias = module.bias)
                with torch.no_grad():
                    dequantized_weight = processed_quantized_model.dequant(module.weight().data)
                    recovered_weight = self.__recover_linear_from_fp32_dict(module_name, dequantized_weight)
                    tmp_linear.weight = nn.Parameter(recovered_weight)
                    dequantized_bias = processed_quantized_model.dequant(module.bias().data)
                    tmp_linear.bias = nn.Parameter(dequantized_bias)
                out = tmp_linear(out)
                out = processed_quantized_model.quant(out)
            elif isinstance(module, torch.ao.nn.intrinsic.quantized.ConvReLU2d) or isinstance(module, torch.ao.nn.quantized.Conv2d):
                if "downsample" in module_name:
                    continue
                out = processed_quantized_model.dequant(out)
                tmp_conv = torch.nn.Conv2d(in_channels = module.in_channels,
                                        out_channels = module.out_channels,
                                        kernel_size = module.kernel_size,
                                        stride = module.stride,
                                        padding = module.padding,
                                        dilation = module.dilation,
                                        bias = module.bias,
                                        groups = module.groups)
                # print(module.in_channels, module.out_channels, module.kernel_size, module.stride)
                if isinstance(module, torch.ao.nn.intrinsic.quantized.ConvReLU2d):
                    tmp_relu = torch.nn.ReLU(inplace=True)
                with torch.no_grad():
                    dequantized_weight = processed_quantized_model.dequant(module.weight().data)
                    recovered_weight = self.__recover_conv_from_fp32_dict(module_name, dequantized_weight)
                    tmp_conv.weight = nn.Parameter(recovered_weight)
                    dequantized_bias = processed_quantized_model.dequant(module.bias().data)
                    tmp_conv.bias = nn.Parameter(dequantized_bias)
                out = tmp_conv(out)
                if isinstance(module, torch.ao.nn.intrinsic.quantized.ConvReLU2d):
                    out = tmp_relu(out)
                out = processed_quantized_model.quant(out)
            else:
                if "downsample" in module_name:
                    continue
                out = module(out)
        out = processed_quantized_model.dequant(out)
        return out