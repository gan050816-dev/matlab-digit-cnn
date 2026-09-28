# MATLAB 手写数字 CNN 分类

## 项目简介

使用 MATLAB 内置手写数字数据集训练一个轻量卷积神经网络，并提供 CPU 与 GPU 两种运行版本。每个类别最多选取 100 张图像，按 80%/20% 划分训练集和测试集。

## 主要内容

```text
28×28 输入
→ 3×3 卷积（8 通道）+ 批归一化 + ReLU
→ 2×2 最大池化
→ 3×3 卷积（16 通道）+ 批归一化 + ReLU
→ 全连接（10 类）+ Softmax
```

训练使用 SGDM，默认训练 2 个 epoch。运行后会显示训练过程、混淆矩阵和部分测试图像的预测结果。

## 环境依赖

- MATLAB
- Deep Learning Toolbox
- Image Processing Toolbox
- GPU 版本还需要 Parallel Computing Toolbox 与兼容的 CUDA GPU

## 使用方法

CPU 版本：

```matlab
run("src/cnn_cpu.m")
```

GPU 版本：

```matlab
run("src/cnn_gpu.m")
```

两个脚本均读取 MATLAB 安装目录中的 `DigitDataset`，仓库不重复分发训练图片。原始脚本的中文注释已从 GBK 转换为 UTF-8，网络结构和训练参数未改动。

## 目录结构

```text
src/cnn_cpu.m    CPU 训练与评估脚本
src/cnn_gpu.m    GPU 训练与评估脚本
```

## 验证状态

CPU 版本已完整运行，本次随机划分测试准确率为 88.50%；GPU 版本完成静态检查，尚未在 CUDA GPU 上运行。

## 已知限制

- 随机划分数据集会导致每次准确率略有变化；需要严格复现时应在脚本开头设置随机种子。
- 当前脚本使用较早版本的 `trainNetwork`、`classificationLayer` 接口；新版本 MATLAB 可能提示迁移到 `trainnet`。
- 两个版本的输入通道设置不同：CPU 版本使用单通道，GPU 版本把灰度图转换为三通道。

## 隐私与公开范围

公开副本只包含 MATLAB 脚本，不重复分发 MATLAB 自带数据集，也不包含个人信息或本地绝对路径。

## 许可证

当前未附加开源许可证。公开仓库可用于作品展示，第三方复用权限需由仓库所有者另行确定。
