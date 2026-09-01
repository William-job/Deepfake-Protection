"""为 HuggingFace 缓存创建 snapshot 符号链接（Windows 修复）。

问题: hf_hub_download 成功下载 blob 文件, 但 snapshot 目录中缺少 model.safetensors 链接,
     导致 timm.create_model('xception41', pretrained=True) 报 FileNotFoundError。
原因: Windows 上 hf_hub_download 创建符号链接需要管理员权限, 普通用户失败后未回退到 copy。
修复: 直接 copy blob 到 snapshot/model.safetensors。
"""
import os
import shutil
import sys

CACHE_ROOT = r"C:\Users\86188\.cache\huggingface\hub\models--timm--xception41.tf_in1k"
SNAPSHOT_HASH = "8a17189361e63c972815ef62f2a30dd5b9f393b1"
BLOB_NAME = "c208ded589c3f45080f3c50b9c84fa4f1cb51aea1fc5308e5a58efc7f4f2892d"


def main():
    blob_path = os.path.join(CACHE_ROOT, "blobs", BLOB_NAME)
    snapshot_dir = os.path.join(CACHE_ROOT, "snapshots", SNAPSHOT_HASH)
    target_path = os.path.join(snapshot_dir, "model.safetensors")

    if not os.path.exists(blob_path):
        print(f"[ERROR] blob 不存在: {blob_path}")
        sys.exit(1)

    blob_size = os.path.getsize(blob_path)
    print(f"[INFO] blob 大小: {blob_size / 1024 / 1024:.1f} MB")

    if os.path.exists(target_path) and os.path.getsize(target_path) == blob_size:
        print(f"[SKIP] snapshot link 已存在且完整: {target_path}")
        return

    os.makedirs(snapshot_dir, exist_ok=True)
    print(f"[COPY] {blob_path} -> {target_path}")
    shutil.copy(blob_path, target_path)

    target_size = os.path.getsize(target_path)
    if target_size != blob_size:
        print(f"[ERROR] 复制后大小不匹配: {target_size} != {blob_size}")
        sys.exit(1)
    print(f"[OK] snapshot link 已创建: {target_path} (size={target_size / 1024 / 1024:.1f} MB)")

    # 验证可加载
    try:
        from safetensors.torch import load_file
        sd = load_file(target_path)
        print(f"[VERIFY] safetensors 加载成功, keys 数: {len(sd)}")
        print(f"[VERIFY] first 3 keys: {list(sd.keys())[:3]}")
    except Exception as e:
        print(f"[ERROR] safetensors 加载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
