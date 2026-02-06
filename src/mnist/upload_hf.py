from huggingface_hub import HfApi, hf_hub_download, upload_file

from mnist.train import MnisModel


def upload_model(model: MnisModel):
    """
    upload_model 保存模型参数到 hugging face hub
    """
    api = HfApi()  # 创建 Hugging Face API 实例
    upload_file(  # 上传文件到 Hugging Face
        path_or_fileobj="simple.bin",  # 本地文件路径
        path_in_repo="simple.bin",  # 仓库中的文件名
        repo_id="miniocean404/simple-nn",  # 仓库 ID
        repo_type="model",  # 仓库类型为模型
    )


def download_model(model_path: str):
    """
    download_model 从 hugging face 下载模型权重文件
    """
    hf_hub_download(repo_id="miniocean404/simple-nn", filename="simple.bin")
