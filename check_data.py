# check_data.py
import pandas as pd

FILE_PATH = "data/combined/combined_data.csv"

try:
    df = pd.read_csv(FILE_PATH)
    print(f"成功读取文件：{FILE_PATH}")
    print("=" * 70)
    
    print(f"总行数（含表头）：{len(df) + 1}，实际样本数：{len(df)}")
    print("\n列名：", df.columns.tolist())
    
    # label 分布
    if 'label' in df.columns:
        pos = df['label'].sum()
        print(f"\n正样本 (label=1)：{pos} 条 ({pos/len(df):.2%})")
        print(f"负样本 (label=0)：{len(df) - pos} 条")
        print("label 详细分布：\n", df['label'].value_counts(dropna=False))
    else:
        print("⚠️ 没有 'label' 列！")
    
    # 序列信息（用 SEQUENCE 列）
    if 'SEQUENCE' in df.columns:
        df['seq_len'] = df['SEQUENCE'].astype(str).str.len()
        print("\n序列长度统计（SEQUENCE 列）：")
        print(df['seq_len'].describe())
        print("\n长度分桶（前10个常见长度）：")
        print(df['seq_len'].value_counts().head(10))
        print(f"最短序列长度：{df['seq_len'].min()}，最长：{df['seq_len'].max()}")
        
        # 非法字符粗查
        illegal = df['SEQUENCE'].str.contains(r'[^ACDEFGHIKLMNPQRSTVWY]', regex=True, na=False)
        print(f"\n包含非标准氨基酸的序列数量：{illegal.sum()}")
        if illegal.sum() > 0:
            print("含非法字符的前3个示例：")
            print(df[illegal][['SEQUENCE']].head(3).to_string(index=False))
    else:
        print("⚠️ 没有 'SEQUENCE' 列！请检查数据")
    
    print("\n前 6 行完整预览：")
    print(df.head(6).to_string(index=False))
    
    print("\n缺失值统计：")
    print(df.isnull().sum())

except Exception as e:
    print("读取失败：", str(e))