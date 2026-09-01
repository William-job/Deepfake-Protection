"""下载 xception 预训练权重到本地缓存（绕过 SSL 证书过期问题）。

来源: https://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth
MD5: 43020ad28 (timm 旧版 xception 默认权重)
"""
import os
import sys
import urllib.request
import ssl

URL = "https://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth"
DEST = r"C:\Users\86188\.cache\torch\hub\checkpoints\xception-43020ad28.pth"
EXPECTED_SIZE = 87783392  # 约 84 MB（完整文件）


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)

    # 若已存在完整文件, 跳过下载
    if os.path.exists(DEST) and os.path.getsize(DEST) > 80_000_000:
        print(f"[SKIP] 已存在完整文件: {DEST} (size={os.path.getsize(DEST)})")
        return

    # 删除损坏的旧文件
    if os.path.exists(DEST):
        os.remove(DEST)
        print(f"[DEL] 删除损坏的旧文件: {DEST}")

    # 关闭 SSL 证书验证（data.lip6.fr 证书已过期）
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f"[GET] {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        with open(DEST, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)

    size = os.path.getsize(DEST)
    print(f"[OK] 下载完成: {DEST} (size={size})")
    if size < 80_000_000:
        print(f"[WARN] 文件大小 {size} < 80MB, 可能不完整")
        sys.exit(1)
    print(f"[INFO] 文件大小 {size / 1024 / 1024:.1f} MB, 预期 ~84 MB")

    # 验证可加载
    try:
        import torch
        ckpt = torch.load(DEST, map_location="cpu", weights_only=True)
        print(f"[VERIFY] 加载成功, keys 数: {len(ckpt)}")
    except Exception as e:
        print(f"[ERROR] 加载验证失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
