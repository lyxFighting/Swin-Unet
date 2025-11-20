import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import argparse

# 添加项目路径
sys.path.append('/home/liuyuxiu/models/Swin-Unet')

from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config

def create_config():
    """创建与训练时相同的配置"""
    # 创建一个包含所有必要属性的配置对象
    class Args:
        def __init__(self):
            self.cfg = '/home/liuyuxiu/models/Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml'
            self.opts = None
            self.zip = False
            self.cache_mode = 'part'
            self.resume = None
            self.accumulation_steps = None
            self.use_checkpoint = False
            self.amp_opt_level = 'O1'
            self.tag = None
            self.eval = False
            self.throughput = False
            
            # 添加训练脚本中的必要参数
            self.batch_size = 24  # 与训练时相同
            self.n_gpu = 1
            self.deterministic = 1
            self.base_lr = 0.01
            self.img_size = 224
            self.seed = 1234
            self.dataset = 'Synapse'
            self.root_path = './data/project_TransUNet/project_TransUNet/data/Synapse/train_npz'
            self.list_dir = './lists/lists_Synapse'
            self.num_classes = 9
            self.output_dir = './model_output'
            self.max_iterations = 30000
            self.max_epochs = 150
    
    return get_config(Args())

def load_model(config, model_path, num_classes=9):
    """加载训练好的模型"""
    # 创建模型实例，与训练时一致
    model = ViT_seg(config, img_size=224, num_classes=num_classes)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"检查点类型: {type(checkpoint)}")
    
    # 根据训练保存方式加载权重
    if isinstance(checkpoint, dict):
        print(f"检查点键: {list(checkpoint.keys())}")
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
            print("从 'net' 键加载权重")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("从 'model_state_dict' 键加载权重")
        else:
            # 尝试直接加载
            model.load_state_dict(checkpoint)
            print("直接加载权重")
    else:
        model.load_state_dict(checkpoint)
        print("直接加载权重（非字典格式）")
    
    model.eval()
    print("✅ 模型加载成功!")
    return model

def preprocess_image(image_path, img_size=224):
    """预处理图像"""
    # 打开图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    
    # 预处理变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # 应用变换
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度 (1, 1, H, W)
    
    return image_tensor, image

def predict(model, image_tensor):
    """进行预测"""
    with torch.no_grad():
        # 将输入移动到GPU（如果可用）
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            model = model.cuda()
            print("使用GPU进行预测")
        else:
            print("使用CPU进行预测")
        
        # 前向传播
        output = model(image_tensor)
        
        # 获取预测结果（多类别分割取argmax）
        prediction = torch.argmax(output, dim=1)
        
        return output, prediction

def save_results(original_image, prediction, output_dir, input_filename):
    """保存预测结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取预测结果的numpy数组
    pred_np = prediction.squeeze().cpu().numpy()
    
    # 生成输出文件名
    base_name = os.path.splitext(input_filename)[0]
    
    # 1. 保存对比图
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    # 预测结果
    plt.subplot(1, 3, 2)
    plt.imshow(pred_np, cmap='jet')
    plt.title('预测分割')
    plt.axis('off')
    
    # 叠加显示
    plt.subplot(1, 3, 3)
    plt.imshow(original_image, cmap='gray')
    plt.imshow(pred_np, cmap='jet', alpha=0.5)
    plt.title('叠加显示')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    # 2. 保存纯预测掩码
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(pred_np, cmap='jet')
    plt.axis('off')
    plt.savefig(mask_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"✅ 预测结果已保存:")
    print(f"   - 对比图: {comparison_path}")
    print(f"   - 预测掩码: {mask_path}")
    
    return comparison_path, mask_path

def main():
    """主函数"""
    # 文件路径
    model_path = '/home/liuyuxiu/models/Swin-Unet/model_output/best_model.pth'
    input_image = '/home/liuyuxiu/models/Swin-Unet/data/project_TransUNet/project_TransUNet/data/Synapse/image/train_npz/case0040_slice061_image.png'
    output_dir = '/home/liuyuxiu/models/Swin-Unet/predict_lyx'
    
    print("🚀 开始预测...")
    
    # 1. 创建配置
    print("📋 创建配置...")
    config = create_config()
    
    # 2. 加载模型
    print("🤖 加载模型...")
    model = load_model(config, model_path, num_classes=9)
    
    # 3. 预处理图像
    print("🖼️  预处理图像...")
    image_tensor, original_image = preprocess_image(input_image)
    
    # 4. 进行预测
    print("🔮 进行预测...")
    output, prediction = predict(model, image_tensor)
    
    # 5. 保存结果
    print("💾 保存结果...")
    input_filename = os.path.basename(input_image)
    comparison_path, mask_path = save_results(
        np.array(original_image), prediction, output_dir, input_filename
    )
    
    print("\n🎉 预测完成!")
    print(f"📁 输入图像: {input_image}")
    print(f"📁 输出目录: {output_dir}")

if __name__ == "__main__":
    main()