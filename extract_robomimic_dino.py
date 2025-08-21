import h5py
import os
import timm
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# ==== 设置路径 ====
input_hdf5 = "/mnt/data1/minghao/robomimic/can/ph/image.hdf5"        # 输入图像观测
output_hdf5 = "/mnt/data1/minghao/robomimic/can/ph/image_dino.hdf5"  # 输出 DINO 特征
camera_key = "agentview_image"                                       # 相机视角 key

# ==== 加载 DINOv2 ViT-B (518×518) ====
print("Loading DINOv2 model...")
model = timm.create_model("vit_base_patch14_dinov2", pretrained=True)
model.eval().cuda()

transform = T.Compose([
    T.Resize((518, 518), interpolation=Image.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# ==== 读取输入文件并写入输出特征 ====
with h5py.File(input_hdf5, "r") as f_in, h5py.File(output_hdf5, "w") as f_out:
    demo_names = list(f_in["data"].keys())
    print(f"Found {len(demo_names)} demos.")

    for demo in tqdm(demo_names, desc="Extracting DINO features"):
        # 读取图像序列
        try:
            images = f_in[f"data/{demo}/obs/{camera_key}"][:]  # shape: [T, H, W, 3]
        except KeyError:
            print(f"[Warning] {demo} missing {camera_key}, skipping.")
            continue

        T_len = images.shape[0]
        dino_feats = []

        for i in range(T_len):
            img = Image.fromarray(images[i].astype("uint8"))
            inp = transform(img).unsqueeze(0).cuda()  # [1, 3, 518, 518]

            with torch.no_grad():
                feat_all = model.forward_features(inp)  # shape: [1, num_tokens, 768]
                cls_feat = feat_all[:, 0, :]            # 取 CLS token 作为图像整体特征
                dino_feats.append(cls_feat.cpu().numpy())

        dino_feats = np.concatenate(dino_feats, axis=0)  # shape: [T, 768]
        print(dino_feats.shape)

        # === 写入 HDF5：demo-wise 分开存储 ===
        f_out.create_dataset(demo, data=dino_feats, compression="gzip")

print(f"\n✅ Done. DINO features saved to: {output_hdf5}")