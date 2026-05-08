import torch
from net.DRNet_arch import DRNet, DRMLP

net_denoise = DRNet(num_experts=4, input_size=128, expert_dim=3, num_blocks=[4,6,6,8])
state_dict = torch.load("/data2/liao/projects/All-in-One/PromptIR-main/train_ckpt/Ours_onehot_3tasks/epoch=149-step=665000.ckpt")['state_dict']

# 创建新的 state_dict，去除 'net.' 前缀
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('net.'):
        # 去除 'net.' 前缀
        new_key = key[4:]  # 移除前4个字符 'net.'
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value


# 加载修改后的 state_dict
net_denoise.load_state_dict(new_state_dict, strict=True)

torch.save(net_denoise.state_dict(), 'ckpt/3tasks/DRNet.pth')