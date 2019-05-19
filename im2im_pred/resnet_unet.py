import torch
from fastai.callbacks import model_sizes
from fastai.vision import Callable, Optional, NormType, Union, Tuple, SplitFuncOrIdxList, Any, SequentialEx, \
    SigmoidRange, relu, Tensor
from fastai.vision.models.unet import _get_sfs_idxs, dummy_eval, conv_layer, batchnorm_2d, \
    PixelShuffle_ICNR, hook_outputs, Hook, F
from torch import nn


class DynamicUnetWithoutSkipConnections(SequentialEx):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int,
                 y_range:Optional[Tuple[float,float]]=None, skip_connections=True,
                 **kwargs):

        encoder[2] = encoder[0:3]
        encoder = nn.Sequential(*list(encoder.children())[2:])

        attented_layers = []
        filter = []

        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        sfs_idxs = sfs_idxs[:-1]

        attented_layers.extend([encoder[ind] for ind in sfs_idxs[::-1]])
        attented_layers.append(encoder[-1])

        filter.extend([sfs_szs[ind][1] for ind in sfs_idxs[::-1]])

        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        filter.append(ni)

        middle_conv_enc = conv_layer(ni, ni*2, **kwargs).eval()
        middle_conv_dec = conv_layer(ni*2, ni, **kwargs).eval()

        x = middle_conv_enc(x)
        x = middle_conv_dec(x)

        layers = list(encoder)
        layers = layers + [batchnorm_2d(ni), nn.ReLU(), middle_conv_enc, middle_conv_dec]

        attented_layers.append(middle_conv_enc)
        attented_layers.append(middle_conv_dec)

        filter.extend([ni*2,ni*2,ni])

        # sfs_idxs = sfs_idxs[:-2]
        for i,idx in enumerate(sfs_idxs):
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            if skip_connections:
                not_final = not (i!=len(sfs_idxs)-1)
                unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
                                       **kwargs).eval()
            else:
                unet_block = UnetBlockWithoutSkipConnection(up_in_c, **kwargs).eval()


            layers.append(unet_block)
            x = unet_block(x)

            attented_layers.append(layers[-1])
            filter.append(x_in_c)   # in for first filter param for attention block

        filter = filter[:-1]

        ni = x.shape[1]

        unet_block_last = UnetBlockWithoutSkipConnection(10,
                                                        # final_div=not_final,
                                                        blur=False, self_attention=False,
                                   **kwargs)

        if imsize != sfs_szs[0][-2:]:
            unet_block_last.shuf = PixelShuffle_ICNR(ni, **kwargs)
        else:
            unet_block_last.shuf = nn.Identity()

        unet_block_last.conv1 = conv_layer(ni, n_classes, ks=1, use_activ=False, **kwargs)
        unet_block_last.conv2 = nn.Identity()
        unet_block_last.relu = nn.Identity()

        layers.append(unet_block_last)
        attented_layers.append(unet_block_last)
        # if skip_connections:
        #     ni = 32
        filter.extend([ni, n_classes])

        if y_range is not None: layers.append(SigmoidRange(*y_range))

        super().__init__(*layers)
        self.attended_layers = attented_layers
        self.filter = filter

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

    def forward(self, up_in:Tensor, index_intermediate=None):
        assert index_intermediate is None or index_intermediate <= 2, 'index_intermediate too large'

        return_intermediate = None

        up_out = self.shuf(up_in)
        if index_intermediate == 0:
            return_intermediate = up_out

        cat_x = self.relu(up_out)
        if index_intermediate == 1:
            return_intermediate = cat_x

        cat_x = self.conv1(cat_x)
        if index_intermediate == 2:
            return_intermediate = cat_x

        if return_intermediate is None:
            return self.conv2(cat_x)
        else:
            return self.conv2(cat_x), return_intermediate


class UnetBlock(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, blur:bool=False, leaky:float=None,
                 self_attention:bool=False, **kwargs):
        super().__init__()
        self.hook = hook
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in:Tensor, index_intermediate=None):
        assert index_intermediate is None or index_intermediate == 0, \
            'only index_intermediate = 0 supported bc of concatenating'

        return_intermediate = None

        s = self.hook.stored
        up_out = self.shuf(up_in)
        if index_intermediate == 0:
            return_intermediate = up_out

        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))

        if return_intermediate is None:
            return self.conv2(self.conv1(cat_x))
        else:
            return self.conv2(self.conv1(cat_x)), return_intermediate


def unet_learner_without_skip_connections(n_classes, device, arch:Callable, pretrained:bool=True, blur_final:bool=True,
                 norm_type:Optional[NormType]=NormType, split_on:Optional[SplitFuncOrIdxList]=None, blur:bool=False,
                 y_range:Optional[Tuple[float,float]]=None, skip_connections=True,
                 cut:Union[int,Callable]=None, **learn_kwargs:Any):
    "Build Unet learner from `data` and `arch`."
    from fastai.vision import create_body
    from fastai.torch_core import to_device
    from fastai.torch_core import apply_init

    # meta = cnn_config(arch)

    body = create_body(arch, pretrained, cut)
    # noinspection PyTypeChecker
    model = to_device(DynamicUnetWithoutSkipConnections(body, n_classes=n_classes, y_range=y_range, norm_type=norm_type,
                                                        skip_connections=skip_connections),
                      device)

    # learn = Learner(data, model, **learn_kwargs)
    # learn.split(ifnone(split_on, meta['split']))
    # if pretrained: learn.freeze()

    apply_init(model[2], nn.init.kaiming_normal_)
    return model
