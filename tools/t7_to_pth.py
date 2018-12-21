# load_lua is supported in pytorch 0.4.1 but not 1.0.0
import torch
import collections
from torch.utils.serialization import load_lua

multiGPU = False
prefix = 'module.' if multiGPU else ''

model1 = load_lua('vgg_SCNN_DULR_w9.t7', unknown_classes=True)
model2 = collections.OrderedDict()

model2[prefix+'conv1_1.weight'] = model1.modules[0].weight
model2[prefix+'bn1_1.weight'] = model1.modules[1].weight
model2[prefix+'bn1_1.bias'] = model1.modules[1].bias
model2[prefix+'bn1_1.running_mean'] = model1.modules[1].running_mean
model2[prefix+'bn1_1.running_var'] = model1.modules[1].running_var
model2[prefix+'conv1_2.weight'] = model1.modules[3].weight
model2[prefix+'bn1_2.weight'] = model1.modules[4].weight
model2[prefix+'bn1_2.bias'] = model1.modules[4].bias
model2[prefix+'bn1_2.running_mean'] = model1.modules[4].running_mean
model2[prefix+'bn1_2.running_var'] = model1.modules[4].running_var

model2[prefix+'conv2_1.weight'] = model1.modules[7].weight
model2[prefix+'bn2_1.weight'] = model1.modules[8].weight
model2[prefix+'bn2_1.bias'] = model1.modules[8].bias
model2[prefix+'bn2_1.running_mean'] = model1.modules[8].running_mean
model2[prefix+'bn2_1.running_var'] = model1.modules[8].running_var
model2[prefix+'conv2_2.weight'] = model1.modules[10].weight
model2[prefix+'bn2_2.weight'] = model1.modules[11].weight
model2[prefix+'bn2_2.bias'] = model1.modules[11].bias
model2[prefix+'bn2_2.running_mean'] = model1.modules[11].running_mean
model2[prefix+'bn2_2.running_var'] = model1.modules[11].running_var

model2[prefix+'conv3_1.weight'] = model1.modules[14].weight
model2[prefix+'bn3_1.weight'] = model1.modules[15].weight
model2[prefix+'bn3_1.bias'] = model1.modules[15].bias
model2[prefix+'bn3_1.running_mean'] = model1.modules[15].running_mean
model2[prefix+'bn3_1.running_var'] = model1.modules[15].running_var
model2[prefix+'conv3_2.weight'] = model1.modules[17].weight
model2[prefix+'bn3_2.weight'] = model1.modules[18].weight
model2[prefix+'bn3_2.bias'] = model1.modules[18].bias
model2[prefix+'bn3_2.running_mean'] = model1.modules[18].running_mean
model2[prefix+'bn3_2.running_var'] = model1.modules[18].running_var
model2[prefix+'conv3_3.weight'] = model1.modules[20].weight
model2[prefix+'bn3_3.weight'] = model1.modules[21].weight
model2[prefix+'bn3_3.bias'] = model1.modules[21].bias
model2[prefix+'bn3_3.running_mean'] = model1.modules[21].running_mean
model2[prefix+'bn3_3.running_var'] = model1.modules[21].running_var

model2[prefix+'conv4_1.weight'] = model1.modules[24].weight
model2[prefix+'bn4_1.weight'] = model1.modules[25].weight
model2[prefix+'bn4_1.bias'] = model1.modules[25].bias
model2[prefix+'bn4_1.running_mean'] = model1.modules[25].running_mean
model2[prefix+'bn4_1.running_var'] = model1.modules[25].running_var
model2[prefix+'conv4_2.weight'] = model1.modules[27].weight
model2[prefix+'bn4_2.weight'] = model1.modules[28].weight
model2[prefix+'bn4_2.bias'] = model1.modules[28].bias
model2[prefix+'bn4_2.running_mean'] = model1.modules[28].running_mean
model2[prefix+'bn4_2.running_var'] = model1.modules[28].running_var
model2[prefix+'conv4_3.weight'] = model1.modules[30].weight
model2[prefix+'bn4_3.weight'] = model1.modules[31].weight
model2[prefix+'bn4_3.bias'] = model1.modules[31].bias
model2[prefix+'bn4_3.running_mean'] = model1.modules[31].running_mean
model2[prefix+'bn4_3.running_var'] = model1.modules[31].running_var

model2[prefix+'conv5_1.weight'] = model1.modules[33].weight
model2[prefix+'bn5_1.weight'] = model1.modules[34].weight
model2[prefix+'bn5_1.bias'] = model1.modules[34].bias
model2[prefix+'bn5_1.running_mean'] = model1.modules[34].running_mean
model2[prefix+'bn5_1.running_var'] = model1.modules[34].running_var
model2[prefix+'conv5_2.weight'] = model1.modules[36].weight
model2[prefix+'bn5_2.weight'] = model1.modules[37].weight
model2[prefix+'bn5_2.bias'] = model1.modules[37].bias
model2[prefix+'bn5_2.running_mean'] = model1.modules[37].running_mean
model2[prefix+'bn5_2.running_var'] = model1.modules[37].running_var
model2[prefix+'conv5_3.weight'] = model1.modules[39].weight
model2[prefix+'bn5_3.weight'] = model1.modules[40].weight
model2[prefix+'bn5_3.bias'] = model1.modules[40].bias
model2[prefix+'bn5_3.running_mean'] = model1.modules[40].running_mean
model2[prefix+'bn5_3.running_var'] = model1.modules[40].running_var

model2[prefix+'conv6.weight'] = model1.modules[42].modules[0].weight
model2[prefix+'bn6.weight'] = model1.modules[42].modules[1].weight
model2[prefix+'bn6.bias'] = model1.modules[42].modules[1].bias
model2[prefix+'bn6.running_mean'] = model1.modules[42].modules[1].running_mean
model2[prefix+'bn6.running_var'] = model1.modules[42].modules[1].running_var
model2[prefix+'conv7.weight'] = model1.modules[42].modules[3].weight
model2[prefix+'bn7.weight'] = model1.modules[42].modules[4].weight
model2[prefix+'bn7.bias'] = model1.modules[42].modules[4].bias
model2[prefix+'bn7.running_mean'] = model1.modules[42].modules[4].running_mean
model2[prefix+'bn7.running_var'] = model1.modules[42].modules[4].running_var

model2[prefix+'conv_d.weight'] = model1.modules[42].modules[6].modules[0].modules[0].modules[2].modules[0].modules[1].modules[1].weight
model2[prefix+'conv_u.weight'] = model1.modules[42].modules[6].modules[0].modules[0].modules[140].modules[1].modules[2].modules[0].weight
model2[prefix+'conv_r.weight'] = model1.modules[42].modules[6].modules[1].modules[0].modules[2].modules[0].modules[1].modules[1].weight
model2[prefix+'conv_l.weight'] = model1.modules[42].modules[6].modules[1].modules[0].modules[396].modules[1].modules[2].modules[0].weight

model2[prefix+'conv8.weight'] = model1.modules[42].modules[8].weight
model2[prefix+'conv8.bias'] = model1.modules[42].modules[8].bias
model2[prefix+'fc9.weight'] = model1.modules[43].modules[1].modules[3].weight
model2[prefix+'fc9.bias'] = model1.modules[43].modules[1].modules[3].bias
model2[prefix+'fc10.weight'] = model1.modules[43].modules[1].modules[5].weight
model2[prefix+'fc10.bias'] = model1.modules[43].modules[1].modules[5].bias

torch.save(model2, 'vgg_SCNN_DULR_w9.pth')