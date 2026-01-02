#!/usr/bin/env python3
"""
从 Backblaze B2 下载数据文件的 Python 脚本
用于在 Render 部署时自动下载数据文件
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

# 配置
B2_KEY_ID = os.environ.get('B2_KEY_ID', '005f3bca11c7bdf0000000001')
B2_APPLICATION_KEY = os.environ.get('B2_APPLICATION_KEY', '')
B2_BUCKET_NAME = os.environ.get('B2_BUCKET_NAME', 'cdss-data')
FILE_NAME = 'eicu_mimic_lab_time.csv'

# 如果bucket是公开的，可以使用这个URL（需要根据实际情况修改）
PUBLIC_URL = f"https://f000.backblazeb2.com/file/{B2_BUCKET_NAME}/{FILE_NAME}"

def download_with_requests(url, output_file):
    """使用 requests 下载文件"""
    print(f"正在从 {url} 下载文件...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print("\n✅ 下载完成")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def download_with_b2sdk():
    """使用 b2sdk Python库下载（推荐，更可靠）"""
    try:
        # 安装 b2sdk
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'b2sdk', '--quiet'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        from b2sdk.v1 import InMemoryAccountInfo, B2Api
        from b2sdk.v1.exception import B2Error
        
        # 初始化 B2 API
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", B2_KEY_ID, B2_APPLICATION_KEY)
        
        # 获取bucket并下载文件
        bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        downloaded_file = bucket.download_file_by_name(FILE_NAME)
        
        # 保存文件
        downloaded_file.save_to(FILE_NAME)
        
        print("✅ 下载完成（使用 b2sdk）")
        return True
    except ImportError:
        # b2sdk未安装，返回False让其他方法尝试
        return False
    except Exception as e:
        print(f"⚠️  b2sdk 下载失败: {e}")
        return False

def download_with_b2():
    """使用 b2 命令行工具下载（备用方案）"""
    try:
        # 安装 b2
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'b2', '--quiet'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 授权
        auth_cmd = ['b2', 'authorize-account', B2_KEY_ID, B2_APPLICATION_KEY]
        subprocess.check_call(auth_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 下载
        download_cmd = ['b2', 'download-file-by-name', B2_BUCKET_NAME, FILE_NAME, FILE_NAME]
        subprocess.check_call(download_cmd)
        print("✅ 下载完成（使用 b2 工具）")
        return True
    except Exception as e:
        print(f"⚠️  b2 工具下载失败: {e}")
        return False

def main():
    print("=" * 60)
    print("从 Backblaze B2 下载数据文件")
    print("=" * 60)
    print(f"文件: {FILE_NAME}")
    print(f"Bucket: {B2_BUCKET_NAME}")
    print()
    
    # 如果文件已存在，跳过
    if Path(FILE_NAME).exists():
        file_size = Path(FILE_NAME).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 文件已存在 ({file_size:.1f} MB)，跳过下载")
        return 0
    
    # 如果应用密钥未设置，尝试公开URL
    if not B2_APPLICATION_KEY:
        print("⚠️  B2_APPLICATION_KEY 未设置，尝试使用公开URL下载...")
        if download_with_requests(PUBLIC_URL, FILE_NAME):
            return 0
        else:
            print("❌ 公开URL下载失败")
            print("提示: 请设置 B2_APPLICATION_KEY 环境变量，或将bucket设置为公开访问")
            return 1
    
    # 尝试使用 b2 工具下载
    if download_with_b2():
        return 0
    
    # 首先尝试使用 b2sdk（推荐）
    if download_with_b2sdk():
        return 0
    
    # 如果 b2sdk 失败，尝试使用 b2 命令行工具（备用）
    if download_with_b2():
        return 0
    
    # 如果 b2 工具失败，尝试公开URL
    print("尝试使用公开URL下载...")
    if download_with_requests(PUBLIC_URL, FILE_NAME):
        return 0
    
    print("❌ 所有下载方式均失败")
    print("提示: 部分功能将不可用，但服务仍可启动")
    return 1

if __name__ == '__main__':
    sys.exit(main())

