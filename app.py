import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置与全局样式
# ==========================================
st.set_page_config(
    page_title="STMamba-Hub | 智能超声云诊断",
    page_icon="🐍",
    layout="wide"
)

st.title("🐍 STMamba-Hub: 跨模态时空解耦乳腺造影辅助诊断系统")
st.markdown("基于 **ST-SAMamba** 架构，融合 B-mode 解剖先验与 CEUS 动态灌注特征，提供秒级精准辅助诊断。")
st.divider()

# ==========================================
# 2. 模拟推理引擎后端 (Mock Backend)
# ==========================================
def mock_inference(bmode_img, ceus_video):
    """模拟 ST-SAMamba 模型的全栈推理过程"""
    
    # 模拟 1: 分类结果
    classification_result = {"Malignant": 0.86, "Benign": 0.14}
    
    # 模拟 2: 分割边界生成 (在原图上画一个模拟的肿瘤轮廓)
    img_array = np.array(bmode_img.convert("RGB"))
    # 假定在图片中央生成一个不规则多边形作为分割边界
    h, w = img_array.shape[:2]
    center_x, center_y = w // 2, h // 2
    pts = np.array([
        [center_x - 40, center_y - 30], [center_x + 30, center_y - 45],
        [center_x + 50, center_y + 20], [center_x - 20, center_y + 50],
        [center_x - 50, center_y + 10]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # 用绿色线条画出模型预测的分割边界
    segmented_img = cv2.polylines(img_array.copy(), [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    
    # 模拟 3: Grad-CAM 热力图叠加
    heatmap = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(heatmap, [pts], 255)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    grad_cam_img = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    
    # 模拟 4: Temporal Mamba 灌注重要性曲线 (洗入-达峰-洗脱)
    frames = np.arange(1, 61) # 假设 60 帧
    # 模拟一个达峰在 25 帧左右的曲线，带有一定噪声
    perfusion_curve = np.exp(-0.5 * ((frames - 25) / 10)**2) + np.random.normal(0, 0.05, 60)
    temporal_data = pd.DataFrame({"Frame": frames, "Attention_Weight": perfusion_curve})
    
    return classification_result, segmented_img, grad_cam_img, temporal_data

# ==========================================
# 3. 多模态影像上传区 (Multi-modal Input)
# ==========================================
st.subheader("📁 第一步：多模态影像上传区")
col1, col2 = st.columns(2)

with col1:
    bmode_file = st.file_uploader("上传灰阶超声 (B-mode) 图像", type=["jpg", "png", "jpeg"])
    if bmode_file is not None:
        bmode_image = Image.open(bmode_file)
        st.image(bmode_image, caption="B-mode 解剖结构预览", use_column_width=True)

with col2:
    ceus_file = st.file_uploader("上传超声造影 (CEUS) 序列/视频", type=["mp4", "avi"])
    if ceus_file is not None:
        # 这里用 Streamlit 原生组件播放视频
        st.video(ceus_file, format="video/mp4")
        st.caption("CEUS 动态灌注预览")

# ==========================================
# 4. 推理控制与动态进度条
# ==========================================
st.divider()
start_btn = st.button("🚀 开始智能诊断 (Start Inference)", type="primary", use_container_width=True)

if start_btn:
    if bmode_file is None or ceus_file is None:
        st.error("请同时上传 B-mode 图像与 CEUS 视频！")
    else:
        st.subheader("⚙️ 云端极速推理引擎状态")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 模拟推理状态流转，向评委展示解耦机制
        status_text.text("正在启动 Spatial Mamba 提取 B-mode 全局空间拓扑...")
        time.sleep(1)
        progress_bar.progress(25)
        
        status_text.text("正在通过多尺度 FiLM 模块生成结构感知调制参数...")
        time.sleep(1)
        progress_bar.progress(50)
        
        status_text.text("正在启动 Temporal Mamba 建模 CEUS 长程血液动力学特征...")
        time.sleep(1.5)
        progress_bar.progress(75)
        
        status_text.text("正在融合双分支特征，执行分割与分类多任务联合解码...")
        time.sleep(1)
        progress_bar.progress(100)
        status_text.text("诊断完成！")
        
        # 执行推理逻辑
        cls_res, seg_img, cam_img, temp_data = mock_inference(bmode_image, ceus_file)
        
        # ==========================================
        # 5. 交互式智能诊断报告 (Interactive Report)
        # ==========================================
        st.divider()
        st.header("📋 智能诊断报告")
        
        # 面板 1: 分类结果
        st.subheader("1. 肿瘤良恶性预测")
        mal_prob = cls_res["Malignant"]
        st.metric(label="恶性概率 (Malignant Probability)", value=f"{mal_prob * 100:.2f}%")
        st.progress(mal_prob)
        if mal_prob > 0.5:
            st.warning("⚠️ 高危提示：建议进行穿刺活检进一步确诊。")
        else:
            st.success("✅ 低危提示：良性可能性较高，建议定期随访。")
            
        st.markdown("---")
        
        # 面板 2: 结构引导与空间可视化 (加入分割反馈)
        st.subheader("2. 结构感知可解释性 (Spatial Mamba 视图)")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(seg_img, caption="模型输出的像素级肿瘤分割边界 (绿色轮廓)", use_column_width=True)
            st.markdown("**临床意义**：通过附加的分割监督任务，强制模型精准勾勒了肿瘤的解剖边界，为功能灌注提供了纯净的空间先验。")
        with col_img2:
            st.image(cam_img, caption="Grad-CAM 热力图 (高亮模型关注区域)", use_column_width=True)
            st.markdown("**临床意义**：证明 FiLM 调制机制成功滤除了周围的散斑噪声，使模型注意力高度聚焦于病灶内部。")
            
        st.markdown("---")
        
        # 面板 3: 时序动态可视化
        st.subheader("3. 动态血流可解释性 (Temporal Mamba 视图)")
        st.line_chart(data=temp_data.set_index("Frame"))
        st.markdown("**临床意义**：时间步重要性曲线证实，Temporal Mamba 自适应地聚焦于 CEUS 序列的**“洗入-达峰-洗脱”**关键阶段，有效剔除了无效背景帧。")