# test_dataset.py
from src.data.aop_dataset import AOPDataset

DATA_PATH = "data/combined/combined_data.csv"   # ← 如果你移动了文件，记得改路径

dataset = AOPDataset(DATA_PATH)

print(f"总长度：{len(dataset)}")
print("\n前 3 个样本示例：")
for i in range(3):
    item = dataset[i]
    print(f"[{i}] sequence: {item['sequence'][:20]}... | label: {item['label']}")
