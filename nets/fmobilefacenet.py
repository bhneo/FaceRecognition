from tensorflow import keras

from common import block
from config import config


''''
def Act(data, act_type, name):
    #ignore param act_type, set it in this function
    if act_type=='prelu':
      body = mx.sym.LeakyReLU(data = data, act_type='prelu', name = name)
    else:
      body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    act = Act(data=bn, act_type=config.net_act, name='%s%s_relu' %(name, suffix))
    return act

def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=config.bn_mom)
    return bn

def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    return conv


def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_sep' %(name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name='%s%s_conv_dw' %(name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name='%s%s_conv_proj' %(name, suffix))
    return proj

def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity=data
    for i in range(num_block):
        shortcut=identity
        conv=DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_block' %(name, suffix), suffix='%d'%i)
        identity=conv+shortcut
    return identity
'''


def res_depth_wise(data, training, num_block=1, num_out=1, kernel_size=3, stride=1, padding=1, num_group=1, wd=0.0005,
                   act_type='prelu', name='res'):
    for i in range(num_block):
        data = block.ResDWBlock(num_out=num_out,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                num_group=num_group,
                                act_type=act_type,
                                wd=wd,
                                bn_mom=config.bn_mom,
                                name=name,
                                suffix=str(i+1))(data, training=training)
    return data
        

def get_symbol(inputs, embedding_size, training=None, net_act='prelu', wd=0.0005):
    print('in_network', config)
    data = block.ImageStandardization()(inputs)
    blocks = config.net_blocks
    conv_1 = block.ConvBnAct(filters=64, kernel_size=3, stride=2, padding=1, wd=wd, bn_mom=config.bn_mom,
                             act_type=net_act, name='conv_1')(data, training=training)
    if blocks[0] == 1:
        conv_2_dw = block.DWConvBnAct(kernel_size=3, stride=1, padding=1, wd=wd, bn_mom=config.bn_mom,
                                      act_type=net_act, name='conv_2_dw')(conv_1, training=training)
    else:
        conv_2_dw = res_depth_wise(conv_1, training, num_block=blocks[0], num_out=64, kernel_size=3, stride=1,
                                   padding=1, num_group=64, wd=wd, act_type=net_act, name='res_2')
    conv_23 = block.DWBlock(num_out=64, kernel_size=3, stride=2, padding=1, num_group=128, act_type=net_act, wd=wd,
                            bn_mom=config.bn_mom, name='dconv_23')(conv_2_dw, training=training)
    conv_3 = res_depth_wise(conv_23, training, num_block=blocks[1], num_out=64, kernel_size=3, stride=1, padding=1,
                            num_group=128, wd=wd, name='res_3')
    conv_34 = block.DWBlock(num_out=128, kernel_size=3, stride=2, padding=1, num_group=256, act_type=net_act, wd=wd,
                            bn_mom=config.bn_mom, name='dconv_34')(conv_3, training=training)
    conv_4 = res_depth_wise(conv_34, training, num_block=blocks[2], num_out=128, kernel_size=3, stride=1, padding=1,
                            num_group=256, wd=wd, name='res_4')
    conv_45 = block.DWBlock(num_out=128, kernel_size=3, stride=2, padding=1, num_group=512, act_type=net_act, wd=wd,
                            bn_mom=config.bn_mom, name='dconv_45')(conv_4, training=training)
    conv_5 = res_depth_wise(conv_45, training, num_block=blocks[3], num_out=128, kernel_size=3, stride=1, padding=1,
                            num_group=256, wd=wd, name='res_5')
    conv_6_sep = block.ConvBnAct(filters=512, kernel_size=1, stride=1, padding=0, wd=wd, bn_mom=config.bn_mom,
                                 act_type=net_act, name='conv_6sep')(conv_5, training=training)

    fc1 = block.get_fc1(conv_6_sep, training, embedding_size, fc_type=config.net_output, bn_mom=config.bn_mom)

    return fc1

