from fastai.callbacks import model_sizes
from fastai.vision import Callable, Optional, NormType, Union, Tuple, SplitFuncOrIdxList, Any, SequentialEx, \
    SigmoidRange, relu, Tensor
from fastai.vision.models.unet import _get_sfs_idxs, dummy_eval, conv_layer, batchnorm_2d, \
    PixelShuffle_ICNR
from torch import nn


class DynamicUnetWithoutSkipConnections(SequentialEx):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int,
                 y_range:Optional[Tuple[float,float]]=None,
                 **kwargs):
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni*2, **kwargs),
                                    conv_layer(ni*2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            unet_block = UnetBlockWithoutSkipConnection(up_in_c, final_div=not_final, blur=False, self_attention=False,
                                   **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        layers += [conv_layer(ni, n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()


class UnetBlockWithoutSkipConnection(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, final_div:bool=True, blur:bool=False, leaky:float=None,
                 self_attention:bool=False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky, **kwargs)
        ni = up_in_c//2
        nf = ni if final_div else ni//2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in:Tensor) -> Tensor:
        up_out = self.shuf(up_in)
        cat_x = self.relu(up_out)
        return self.conv2(self.conv1(cat_x))


def unet_learner_without_skip_connections(n_classes, device, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 y_range:Optional[Tuple[float,float]]=None,
                 cut:Union[int,Callable]=None, **learn_kwargs:Any):
    "Build Unet learner from `data` and `arch`."
    from fastai.vision import create_body
    from fastai.torch_core import to_device
    from fastai.torch_core import apply_init

    # meta = cnn_config(arch)

    body = create_body(arch, pretrained, cut)
    # noinspection PyTypeChecker
    model = to_device(DynamicUnetWithoutSkipConnections(body, n_classes=n_classes, y_range=y_range, norm_type=norm_type),
                      device)

    # learn = Learner(data, model, **learn_kwargs)
    # learn.split(ifnone(split_on, meta['split']))
    # if pretrained: learn.freeze()

    apply_init(model[2], nn.init.kaiming_normal_)
    return model
