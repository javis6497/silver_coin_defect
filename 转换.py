import os
import xml.etree.ElementTree as ET
import json
import random
import shutil

def get_global_categories(xml_dir, xml_list):
    """全局扫描，确保所有 json 文件的类别 ID 保持绝对一致"""
    category_map = {}
    category_id_counter = 1
    
    for xml_file in xml_list:
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in category_map:
                category_map[name] = category_id_counter
                category_id_counter += 1
                
    return category_map

def process_split(split_name, xml_dir, img_dir, xml_list, category_map, output_json_path, output_img_dir):
    """处理单个数据集划分：生成 JSON 并复制对应的图片"""
    # 确保图片输出目录存在
    os.makedirs(output_img_dir, exist_ok=True)
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": id, "name": name, "supercategory": "none"} for name, id in category_map.items()]
    }
    
    image_id_counter = 1
    annotation_id_counter = 1
    
    for xml_file in xml_list:
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 1. 提取图片文件名并处理图片复制
        filename_node = root.find('filename')
        if filename_node is not None and filename_node.text.endswith('.jpg'):
            filename = filename_node.text
        else:
            filename = os.path.splitext(xml_file)[0] + '.jpg'
            
        src_img_path = os.path.join(img_dir, filename)
        dst_img_path = os.path.join(output_img_dir, filename)
        
        # 检查原图片是否存在，存在则复制
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"⚠️ 警告: 找不到图片 {src_img_path}，已跳过该样本。")
            continue
            
        # 2. 提取图片尺寸信息
        size_node = root.find('size')
        width = int(size_node.find('width').text)
        height = int(size_node.find('height').text)
        
        coco_data["images"].append({
            "file_name": filename,
            "id": image_id_counter,
            "width": width,
            "height": height
        })
        
        # 3. 提取并转换标注信息
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            w = xmax - xmin
            h = ymax - ymin
            
            coco_data["annotations"].append({
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": category_map[name],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id_counter += 1
            
        image_id_counter += 1

    # 4. 写入 JSON 文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=4)
        
    print(f"✅ [{split_name.upper()}] 集处理完成: 包含 {image_id_counter - 1} 张图片，已复制到 {output_img_dir}")

def split_and_convert_full(root_dir, output_root, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """主控函数：协调整个流程"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "划分比例总和必须为 1"
    
    xml_dir = os.path.join(root_dir, 'ANNOTATIONS')
    img_dir = os.path.join(root_dir, 'IMAGES')
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    total_files = len(xml_files)
    
    print(f"总共找到 {total_files} 个 XML 标注文件。开始处理...")
    
    # 扫描全局类别
    category_map = get_global_categories(xml_dir, xml_files)
    print(f"📊 全局类别映射: {category_map}\n")
    
    # 打乱文件
    random.seed(42)
    random.shuffle(xml_files)
    
    # 计算切分点
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    splits = {
        "train": xml_files[:train_end],
        "val": xml_files[train_end:val_end],
        "test": xml_files[val_end:]
    }
    
    # 依次处理三个切分集
    for split_name, split_list in splits.items():
        if len(split_list) == 0:
            continue
            
        out_json = os.path.join(output_root, 'annotations', f'{split_name}.json')
        out_img = os.path.join(output_root, 'images', split_name)
        
        process_split(split_name, xml_dir, img_dir, split_list, category_map, out_json, out_img)
        
    print("\n🎉 全部转换与文件复制完成！")

if __name__ == "__main__":
    # --- 必改配置区域 ---
    # 你的原始大文件夹路径（必须包含 ANNOTATIONS 和 IMAGES 两个子文件夹）
    DATASET_ROOT_DIR = r"F:\.Javis\silver_coin_detection\NEU‑DET 钢材表面缺陷数据集\NEU-DET"
    
    # 转换后，标准 COCO 数据集的保存路径
    OUTPUT_COCO_DIR = r"F:\.Javis\silver_coin_detection\dataset"
    
    # 设置切分比例 (默认 8:1:1)
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    split_and_convert_full(DATASET_ROOT_DIR, OUTPUT_COCO_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)