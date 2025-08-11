# ComfyUI 阿里云视频生成插件

这是一个为ComfyUI开发的阿里云模型工作室视频生成插件，支持多种视频生成功能。

## 功能特性

### 1. 文生视频 (Text-to-Video)
- 支持模型：wan2.2-t2v-plus, wanx2.1-t2v-turbo, wanx2.1-t2v-plus
- 支持分辨率：480*480, 720*720, 1920*1080
- 支持负面提示词
- 可调节视频时长

### 2. 图生视频 (Image-to-Video)
- 支持模型：wan2.2-i2v-plus, wan2.2-i2v-flash, wanx2.1-i2v-turbo, wanx2.1-i2v-plus
- 支持分辨率：480P, 720P, 1080P
- 支持提示词扩展
- 基于输入图像生成动态视频

### 3. 首尾帧生视频 (First-Last Frame to Video)
- 支持模型：wanx2.1-kf2v-plus
- 支持分辨率：720P
- 基于首帧和尾帧生成平滑过渡视频

### 4. 视频特效 (Video Effects)
- 支持模型：wanx2.1-i2v-turbo, wanx2.1-i2v-plus
- 支持分辨率：480P, 720P
- 丰富的特效模板：
  - **通用特效**：解压捏捏(squish)、转圈圈(rotation)、戳戳乐(poke)、气球膨胀(inflate)、分子扩散(dissolve)
  - **单人特效**：时光木马(carousel)、爱你哟(singleheart)、摇摆时刻(dance1)、头号甩舞(dance2)、星摇时刻(dance3)、人鱼觉醒(mermaid)、学术加冕(graduation)、巨兽追袭(dragon)、财从天降(money)
  - **单人或动物特效**：魔法悬浮(flying)、赠人玫瑰(rose)、闪亮玫瑰(crystalrose)
  - **双人特效**：爱的抱抱(hug)、唇齿相依(frenchkiss)、双倍心动(coupleheart)

## 安装方法

### 1. 环境准备
确保已安装ComfyUI和必要的Python依赖：
```bash
pip install requests pillow numpy torch
```

### 2. 获取API密钥
1. 访问[阿里云百炼平台](https://bailian.console.aliyun.com/)
2. 创建应用并获取API Key
3. 设置环境变量：
   ```bash
   export DASHSCOPE_API_KEY="your_api_key_here"
   ```
   或在Windows中：
   ```cmd
   set DASHSCOPE_API_KEY=your_api_key_here
   ```

### 3. 安装插件
1. 将插件文件复制到ComfyUI的`custom_nodes`目录下：
   ```
   ComfyUI/custom_nodes/comfyui-aliyun-video/
   ├── __init__.py
   ├── nodes.py
   └── README.md
   ```

2. 重启ComfyUI

## 使用方法

### 1. 文生视频节点
- 在节点菜单中找到"Aliyun Video" -> "阿里云文生视频"
- 输入文本提示词
- 选择模型和分辨率
- 可选择添加负面提示词和调整时长
- 连接到输出节点查看生成的视频

### 2. 图生视频节点
- 添加"阿里云图生视频"节点
- 连接图像输入
- 输入描述图像动作的提示词
- 选择模型和分辨率
- 启用提示词扩展以获得更好效果

### 3. 首尾帧生视频节点
- 添加"阿里云首尾帧生视频"节点
- 连接首帧和尾帧图像
- 输入过渡描述
- 生成平滑的过渡视频

### 4. 视频特效节点
- 添加"阿里云视频特效"节点
- 连接输入图像
- 选择特效模板
- 根据特效类型选择合适的输入图像：
  - 通用特效：支持任意主体
  - 单人特效：需要单人照片
  - 双人特效：需要双人照片

## 注意事项

1. **API密钥**：必须设置有效的DASHSCOPE_API_KEY环境变量
2. **网络连接**：需要稳定的网络连接访问阿里云API
3. **生成时间**：视频生成通常需要1-5分钟，请耐心等待
4. **计费**：使用API会产生费用，请查看阿里云计费说明
5. **图像格式**：输入图像会自动转换为PNG格式并编码为base64
6. **视频下载**：生成的视频会自动下载到ComfyUI的输出目录

## 错误处理

常见错误及解决方法：

1. **API密钥错误**：检查环境变量DASHSCOPE_API_KEY是否正确设置
2. **网络超时**：检查网络连接，必要时增加超时时间
3. **模型不支持**：确认选择的模型支持所选的分辨率
4. **图像格式错误**：确保输入的是有效的图像张量
5. **任务失败**：检查输入参数是否符合API要求

## 技术支持

如遇到问题，请：
1. 检查ComfyUI控制台的错误信息
2. 确认API密钥和网络连接正常
3. 参考阿里云官方文档：[模型工作室文档](https://help.aliyun.com/zh/model-studio/)

## 更新日志

### v1.0.0
- 初始版本发布
- 支持文生视频、图生视频、首尾帧生视频和视频特效
- 支持多种模型和分辨率选择
- 自动处理异步任务和视频下载

## 许可证

本插件遵循MIT许可证。使用阿里云API需要遵循阿里云的服务条款。