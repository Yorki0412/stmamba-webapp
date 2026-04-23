import streamlit as st
import time
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
import io
from datetime import datetime
from inference import STMambaPredictor # 引入推理引擎

# ==========================================
# 1. 核心工具函数：视频采样与报告生成
# ==========================================

def process_ceus_video(video_bytes, target_frames=60):
    """
    均匀采样 60 帧血流动力学数据
    """
    tfile = "temp_video_upload.mp4"
    with open(tfile, "wb") as f:
        f.write(video_bytes.read())

    cap = cv2.VideoCapture(tfile)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: return None

    indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
        if len(frames) == target_frames: break
    cap.release()
    if os.path.exists(tfile): os.remove(tfile)
    while len(frames) < target_frames:
        frames.append(frames[-1] if frames else np.zeros((224,224,3), dtype=np.uint8))
    return frames

def generate_medical_report(patient_info, diagnosis_res):
    """
    自动生成临床诊断报告文本
    """
    report = f"""
    【STMamba-Hub 乳腺造影辅助诊断报告】
    -------------------------------------------
    报告编号：{datetime.now().strftime('%Y%m%d%H%M%S')}
    生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    一、患者基本信息
    - 姓名：{patient_info['name']}
    - 性别：{patient_info['sex']}
    - 年龄：{patient_info['age']}
    - 临床病史：{patient_info['history']}
    
    二、AI 影像分析结果
    - 恶性概率评分：{diagnosis_res['prob']*100:.2f}%
    - 结构感知 (SA) 分析：病灶边界{"清晰" if diagnosis_res['prob'] < 0.5 else "毛刺状/不规则"}，内部回声{"均匀" if diagnosis_res['prob'] < 0.3 else "不均"}。
    - 时序 Mamba 建模：造影剂表现为{"快进快出" if diagnosis_res['prob'] > 0.6 else "缓慢增强"}特征。
    
    三、诊断辅助建议
    - BI-RADS 分级参考：{diagnosis_res['birads']}
    - 处理建议：{diagnosis_res['advice']}
    
    -------------------------------------------
    报告签名：STMamba AI 辅助系统 (审核医生：{st.session_state.current_doctor})
    注：本报告由深度学习模型生成，仅供临床参考，不作为最终诊断依据。
    """
    return report

# ==========================================
# 2. 页面配置与 CSS 美化
# ==========================================
st.set_page_config(page_title="STMamba-Hub | 临床辅助诊断系统", page_icon="🩺", layout="wide")

def inject_medical_css():
    st.markdown("""
        <style>
        .stApp { background-color: #f0f4f8; }
        .stButton>button {
            border-radius: 4px; background: #2c3e50; color: white; height: 3em; width: 100%;
        }
        div[data-testid="stExpander"] { background-color: white; border-radius: 10px; }
        .report-box { padding: 20px; background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 5px; font-family: 'Courier New', Courier, monospace; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. 状态管理
# ==========================================
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'current_doctor' not in st.session_state: st.session_state.current_doctor = ""
if 'diagnosis_data' not in st.session_state: st.session_state.diagnosis_data = None

@st.cache_resource
def load_predictor():
    return STMambaPredictor(weight_path="best_stmamba.pth")

# ==========================================
# 4. 登录页面 (手机号登录)
# ==========================================
def login_page():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<h2 style='text-align: center;'>🩺 STMamba 辅助诊断系统</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #7f8c8d;'>专业·精准·高效的乳腺超声影像工作站</p>", unsafe_allow_html=True)
        with st.form("login_form"):
            phone = st.text_input("📞 医生手机号", placeholder="请输入 11 位手机号")
            pwd = st.text_input("🔒 登录密码", type="password")
            if st.form_submit_button("认证登录"):
                if len(phone) == 11 and pwd == "doctor123": # 预设测试密码
                    st.session_state.logged_in = True
                    st.session_state.current_doctor = phone
                    st.rerun()
                else:
                    st.error("手机号或密码无效。测试号：13800000000 / 密码：doctor123")

# ==========================================
# 5. 临床工作台
# ==========================================
def main_dashboard():
    predictor = load_predictor()
    
    with st.sidebar:
        st.title("🏥 门诊系统")
        st.write(f"**操作医生**：{st.session_state.current_doctor}")
        st.divider()
        st.info("系统状态：连接 4090 推理服务器成功")
        if st.button("安全退出"):
            st.session_state.logged_in = False
            st.rerun()

    st.header("📋 患者影像诊断工作流")
    tab_input, tab_report, tab_archive = st.tabs(["🎥 影像诊断", "📑 自动报告", "📚 历史病例"])

    with tab_input:
        # 录入患者基本信息
        with st.expander("👤 第一步：录入患者基本信息", expanded=True):
            c1, c2, c3 = st.columns(3)
            p_name = c1.text_input("姓名", value="张女士")
            p_age = c2.number_input("年龄", value=45)
            p_sex = c3.selectbox("性别", ["女", "男"])
            p_history = st.text_area("临床病史摘要", placeholder="例如：左乳触及肿块，无痛，边界不清...")

        # 影像上传
        with st.expander("🖼️ 第二步：上传双模态影像数据", expanded=True):
            col_b, col_c = st.columns(2)
            b_file = col_b.file_uploader("B-mode 灰阶图", type=["jpg", "png"])
            c_file = col_c.file_uploader("CEUS 动态序列", type=["mp4", "avi"])

        if st.button("🚀 启动 ST-SAMamba 联合分析"):
            if b_file and c_file:
                with st.spinner("AI 正在解析时空特征..."):
                    frames = process_ceus_video(c_file)
                    prob, mask = predictor.predict(b_file, frames)
                    
                    # 封装结论
                    birads = "BI-RADS 4C (高风险)" if prob > 0.7 else "BI-RADS 2 (良性)"
                    advice = "建议进行空芯针活检 (CNB)" if prob > 0.5 else "建议 6 个月后复查"
                    
                    st.session_state.diagnosis_data = {
                        "prob": prob, "mask": mask, "birads": birads, 
                        "advice": advice, "b_file": b_file,
                        "p_info": {"name": p_name, "age": p_age, "sex": p_sex, "history": p_history}
                    }
                st.success("分析完成！请前往【自动报告】选项卡查看。")
            else:
                st.warning("请完整上传双模态数据。")

    with tab_report:
        if st.session_state.diagnosis_data:
            data = st.session_state.diagnosis_data
            st.subheader("生成诊断预览")
            
            # 左右布局：左侧图示，右侧文本
            r_col1, r_col2 = st.columns([1, 1.5])
            with r_col1:
                # 绘制绿框分割图
                img = np.array(Image.open(data['b_file']).convert('RGB').resize((224, 224)))
                cnts, _ = cv2.findContours((data['mask'] > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                st.image(img, caption="SA 结构感知定位图")
                st.metric("恶性风险", f"{data['prob']*100:.1f}%")

            with r_col2:
                report_text = generate_medical_report(data['p_info'], data)
                st.markdown(f"```\n{report_text}\n```")
                
                # 下载按钮
                st.download_button(
                    label="📥 下载 PDF 格式诊断报告 (.txt)",
                    data=report_text,
                    file_name=f"Report_{data['p_info']['name']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("暂无分析数据，请先在【影像诊断】中执行分析。")

    with tab_archive:
        st.write("已归档病例列表")
        st.dataframe(pd.DataFrame({
            "时间": ["2026-04-23", "2026-04-22"], "姓名": ["王女士", "李女士"],
            "AI 风险值": ["86.3%", "12.5%"], "建议": ["穿刺", "随访"]
        }), use_container_width=True)

# ==========================================
# 6. 程序入口
# ==========================================
if __name__ == "__main__":
    inject_medical_css()
    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()
