import streamlit as st
import time
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# ==========================================
# 1. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="STMamba-Hub | 智能超声云诊断",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. UI 美化：注入自定义 CSS
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* 隐藏 Streamlit 默认的右上角菜单和底部水印 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 美化主背景和字体 */
        .stApp {
            background-color: #f8fbff; /* 淡雅的医疗蓝背景 */
        }
        
        /* 美化按钮：圆角、渐变色、阴影 */
        .stButton>button {
            border-radius: 8px;
            background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
            color: white;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* 美化登录框的容器外观 */
        div[data-testid="stForm"] {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. 初始化会话状态 (记忆盒子)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = ""

# ==========================================
# 4. 登录页面模块
# ==========================================
def login_page():
    # 使用空白列居中对齐登录框
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center; color: #1565c0;'>⚕️ STMamba 云诊断中心</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>跨模态时空解耦乳腺造影辅助系统</p>", unsafe_allow_html=True)
        st.write("")
        
        # 使用表单包起来，支持回车登录
        with st.form("login_form"):
            st.subheader("用户登录")
            
            # 身份证号输入限制为最大 18 位
            id_card = st.text_input("👤 用户账号 (身份证号)", max_chars=18, placeholder="请输入 18 位身份证号")
            password = st.text_input("🔒 登录密码", type="password", placeholder="请输入密码")
            
            submit_button = st.form_submit_button("安全登录", use_container_width=True)
            
            if submit_button:
                # 这里做了一个极简的账号密码验证 (测试账号)
                if len(id_card) != 18:
                    st.error("账号格式不正确，需为 18 位身份证号。")
                elif id_card == "320102199001011234" and password == "123456": # 测试账号
                    st.session_state.logged_in = True
                    st.session_state.current_user = id_card
                    st.success("验证成功！正在进入系统...")
                    time.sleep(0.5)
                    st.rerun() # 刷新页面，跳转到主界面
                else:
                    st.error("账号或密码错误，请重试。（测试账号: 320102199001011234 / 密码: 123456）")

# ==========================================
# 5. 主工作台页面模块
# ==========================================
def main_dashboard():
    # 侧边栏：用户信息与导航
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1077/1077114.png", width=100) # 更换为通用用户头像占位图
        st.markdown(f"**当前用户**：<br>ID: `{st.session_state.current_user[:6]}****{st.session_state.current_user[-4:]}`", unsafe_allow_html=True)
        st.divider()
        st.markdown("🛠️ **系统设置**")
        st.checkbox("开启结构感知 (Spatial)")
        st.checkbox("开启时序追踪 (Temporal)")
        st.divider()
        if st.button("🚪 退出登录"):
            st.session_state.logged_in = False
            st.session_state.current_user = ""
            st.rerun()

    # 主界面顶部
    st.title("🐍 STMamba-Hub: 智能辅助工作台")
    
    # 引入选项卡 (Tabs) 让界面更整洁
    tab_diagnose, tab_history = st.tabs(["🩺 多模态诊断", "📂 历史档案"])
    
    with tab_diagnose:
        st.info("💡 操作指引：请在下方分别拖入待测的灰阶超声图像（定解剖）与超声造影视频（定血流）。")
        
        # 用卡片式布局包裹上传区
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                bmode_file = st.file_uploader("📥 B-mode 图像", type=["jpg", "png"])
            with col2:
                ceus_file = st.file_uploader("📥 CEUS 视频", type=["mp4", "avi"])
        
        st.write("")
        start_btn = st.button("🚀 启动 ST-SAMamba 联合诊断", use_container_width=True)
        
        if start_btn:
            if bmode_file is None or ceus_file is None:
                st.warning("⚠️ 请确保多模态数据已全部上传。")
            else:
                with st.status("正在进行云端 AI 推理...", expanded=True) as status:
                    st.write("1️⃣ 提取 B-mode 空间解剖特征...")
                    time.sleep(1)
                    st.write("2️⃣ 建立 Temporal Mamba 血流动力学模型...")
                    time.sleep(1)
                    st.write("3️⃣ 多尺度特征对齐与联合解码...")
                    time.sleep(1)
                    status.update(label="诊断完成！", state="complete", expanded=False)
                
                # --- 这里放之前的 Mock 推理与报告展示代码 ---
                st.success("良恶性概率：恶性 86% | 良性 14%")
                
    with tab_history:
        st.write("📅 历史检测记录（演示数据）")
        st.dataframe(
            pd.DataFrame({
                "检测时间": ["2026-04-19 10:30", "2026-04-18 14:15"],
                "样本编号": ["S-001", "S-002"],
                "AI 预测结果": ["高危 (86%)", "低危 (12%)"],
                "实际标签": ["等待确认", "良性"]
            }), 
            use_container_width=True
        )

# ==========================================
# 6. 主逻辑控制器
# ==========================================
if __name__ == "__main__":
    inject_custom_css()
    
    # 根据登录状态决定显示哪个页面
    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()
