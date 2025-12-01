import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("cuda device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")



# print("hello world!")