import os
import yaml
import argparse

def generate_config(split, random_id, data_root):
    """生成单个配置文件"""
    template_path = os.path.join(os.path.dirname(__file__), 'template.yaml')
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # 根据split设置不同参数
    if split == '5':
        num_epoch = 50
        dataloader = 'ntu_60'
    elif split == '10':
        num_epoch = 100
        dataloader = 'ntu_120'
    elif split == '12':
        num_epoch = 92
        dataloader = 'ntu_60'
    elif split == '24':
        num_epoch = 75
        dataloader = 'ntu_120'
    else:
        raise ValueError(f"Unsupported split: {split}")
    
    # 设置工作目录配置
    if random_id == 'r':
        split_config = f"{split}r"
        random_suffix = ""
    else:
        split_config = f"{split}{random_id}"
        random_suffix = random_id.replace('r', '')
    
    # 替换模板中的占位符
    config_content = template_content.format(
        split_config=split_config,
        num_epoch=num_epoch,
        data_root=data_root,
        split=split,
        random_suffix=random_suffix,
        dataloader=dataloader
    )
    
    # 保存配置文件
    if random_id == 'r':
        filename = os.path.join(os.path.dirname(__file__), f"{split}r.yaml")
    else:
        filename = os.path.join(os.path.dirname(__file__), f"{split}{random_id}.yaml")
    
    with open(filename, 'w') as f:
        f.write(config_content)
    
    return filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, help='Split number (5, 10, 12, or 24)')
    parser.add_argument('--random_ids', type=str, nargs='+', default=['r'], 
                      help='Random IDs to generate (e.g., r r2 r3 r4)')
    parser.add_argument('--data_root', type=str, default='skeleton_feature',
                      help='Data root directory (skeleton_feature or random)')
    args = parser.parse_args()
    
    # 验证split
    valid_splits = ['5', '10', '12', '24']
    if args.split not in valid_splits:
        raise ValueError(f"Split must be one of {valid_splits}")
    
    # 生成配置
    generated_files = []
    for random_id in args.random_ids:
        filename = generate_config(args.split, random_id, args.data_root)
        generated_files.append(filename)
    
    print(f"Generated config files: {', '.join(generated_files)}")

if __name__ == '__main__':
    main() 