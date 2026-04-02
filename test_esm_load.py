# test_esm_load.py（改用 transformers 格式加载本地模型）
import torch
from transformers import EsmModel, EsmTokenizer

# 本地模型路径（就是你截图里的那个文件夹）
MODEL_PATH = "./esm2_t33_650M_UR50D"

print(f"从本地加载 ESM-2：{MODEL_PATH}")

try:
    tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
    model = EsmModel.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print("加载成功！")
    print(f"参数量约：{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"运行设备：{device}")
    
    # 测试单条序列推理
    test_seq = "LRGIKNYRVAVL"
    inputs = tokenizer(test_seq, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden = outputs.last_hidden_state   # (1, L+2, 1280)
    print(f"\n测试序列：{test_seq}")
    print(f"输出 hidden state shape：{last_hidden.shape}")
    print(f"去掉 <cls> 和 <eos> 后：{last_hidden[0, 1:len(test_seq)+1].shape}")
    print(f"mean pooling 结果：{last_hidden[0, 1:len(test_seq)+1].mean(dim=0).shape}")

except Exception as e:
    import traceback
    print("加载失败：")
    traceback.print_exc()