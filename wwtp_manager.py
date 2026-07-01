import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

# ================= 全局字体配置（解决云端中文乱码方框） =================
plt.rcParams['font.sans-serif'] = ['SimHei','WenQuanYi Zen Hei','AR PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False

# ================= 页面基础配置 =================
st.set_page_config(
    page_title="五段Bardenpho污水厂运维管理系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 密码登录校验（新增部分） =================
# 初始化登录状态
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# 未登录时显示登录页
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align:center; margin-top:15%'>🔒 系统登录</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        input_pwd = st.text_input("请输入访问密码", type="password")
        if st.button("登录系统", type="primary", use_container_width=True):
            # 从平台后台读取正确密码（本地测试可临时写死，部署后走secrets）
            # 兼容本地无secrets文件、云端有secrets的两种场景
            try:
                correct_pwd = st.secrets["access_password"]
            except:
                correct_pwd = "123456"  # 本地默认密码
            if input_pwd == correct_pwd:
                st.session_state.logged_in = True
                st.success("登录成功，正在进入系统...")
                st.rerun()
            else:
                st.error("密码错误，请重试")
    st.stop()  # 密码验证不通过，停止执行后面所有代码

# matplotlib中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ================= 全局参数初始化 =================
if 'base_params' not in st.session_state:
    st.session_state.base_params = {
        # 水量参数
        'Q_design': 20000,      # 设计日水量 m³/d
        'Q_actual': 14000,      # 实际日水量 m³/d
        'Kz': 1.65,             # 总变化系数
        'Q_max': 750,          # 最大时流量 m³/h
        # 池体容积
        'V_ana': 594,          # 厌氧池 m³
        'V_anox1': 1320,        # 第一缺氧池 m³
        'V_aero1': 5200,        # 第一好氧池 m³
        'V_anox2': 1259,        # 第二缺氧池 m³
        'V_aero2': 945,        # 第二好氧池 m³
        'V_total': 9318,       # 生化总容积 m³
        'settler_area': 615,    # 二沉池总面积 m²
        'settler_depth': 4.0,   # 二沉池有效水深 m

        # 动力学系数
        'Y': 0.45,               # 污泥产率系数
        'Kd': 0.05,             # 内源衰减系数 d⁻¹
        'nitr_rate': 0.045,     # 硝化速率 kgNH3/(kgMLSS·d)
        'denitr_rate': 0.06,    # 反硝化速率 kgNO3/(kgMLSS·d)
        'mlvss_mlss': 0.75,     # MLVSS/MLSS
        'carbon_cod_eq': 0.68,  # 乙酸钠COD当量 gCOD/g
        # 经济参数
        'elec_price': 0.75,     # 电价 元/kWh
        # 除磷药剂单价
        'pac_price': 280,      # PAC单价 元/吨（铝盐，Al2O3 28%）
        'pfs_price': 200,      # PFS单价 元/吨（铁盐，Fe2O3 19%）
        # 碳源药剂单价
        'naac_price': 800,     # 乙酸钠单价 元/吨
        'methanol_price': 600, # 甲醇单价 元/吨
        'glucose_price': 1000,  # 葡萄糖单价 元/吨
        'composite_carbon_price': 600, # 复合碳源单价 元/吨
        # 其他药剂
        'naclo_price': 450,    # 次氯酸钠单价 元/吨
        'pam_price': 12000,     # PAM单价 元/吨
        'hcl_price': 200,       # 盐酸单价 元/吨（pH调节）
        'sludge_dispose_price': 200,  # 污泥处置单价 元/吨湿泥
        'staff_num': 12,        # 运维人数
        'staff_salary': 6800,   # 人均月薪 元
        'maintain_cost': 18000, # 月度维修费 元
        'other_cost': 8000      # 其他杂费 元/月
    }

if 'bio_result' not in st.session_state:
    st.session_state.bio_result = {}

# ================= 侧边栏导航 =================
with st.sidebar:
    st.title("🏭 系统导航")
    st.caption("五段Bardenpho工艺污水厂运维管理系统")
    st.markdown("---")
    page = st.radio(
        "功能模块",
        [
            "📝 基础参数设置",
            "💧 水力与负荷校核",
            "🧪 生化核心计算",
            "🏞️ 二沉池专项校核",
            "⚙️ 工况调节建议",
            "💰 成本经济核算",
            "📊 报表导出"
        ]
    )
    st.markdown("---")
    st.caption("工艺路线：厌氧→缺氧1→好氧1→缺氧2→好氧2→二沉池")
    st.caption("内回流：好氧1 → 缺氧1；好氧1自流至缺氧2深度反硝化")
    st.caption("好氧2功能：吹脱氮气 + 防止二沉池反硝化")


# ================= 页面1：基础参数设置 =================
if page == "📝 基础参数设置":
    st.header("📝 基础信息参数设置")
    st.caption("一次性录入水厂设计参数、动力学系数、经济单价，全局所有模块自动调用")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("一、水量与池体参数")
        Q_design = st.number_input("设计日处理水量 (m³/d)", value=st.session_state.base_params['Q_design'])
        Q_actual = st.number_input("实际日均进水量 (m³/d)", value=st.session_state.base_params['Q_actual'])
        Kz = st.number_input("总变化系数 Kz", value=st.session_state.base_params['Kz'])
        Q_max = st.number_input("最大时流量 (m³/h)", value=st.session_state.base_params['Q_max'])

        st.markdown("#### 各池有效容积 (m³)")
        V_ana = st.number_input("厌氧池", value=st.session_state.base_params['V_ana'])
        V_anox1 = st.number_input("第一缺氧池", value=st.session_state.base_params['V_anox1'])
        V_aero1 = st.number_input("第一好氧池", value=st.session_state.base_params['V_aero1'])
        V_anox2 = st.number_input("第二缺氧池", value=st.session_state.base_params['V_anox2'])
        V_aero2 = st.number_input("第二好氧池", value=st.session_state.base_params['V_aero2'])
        V_total = st.number_input("生化池总容积", value=st.session_state.base_params['V_total'])
        settler_area = st.number_input("二沉池总表面积 (m²)", value=st.session_state.base_params['settler_area'])
        settler_depth = st.number_input("二沉池有效水深 (m)", value=st.session_state.base_params['settler_depth'])

    with col2:
        st.subheader("二、生化动力学系数")
        Y = st.number_input("污泥产率系数 Y", value=st.session_state.base_params['Y'])
        Kd = st.number_input("内源衰减系数 Kd (d⁻¹)", value=st.session_state.base_params['Kd'])
        nitr_rate = st.number_input("硝化速率 kgNH3/(kgMLSS·d)", value=st.session_state.base_params['nitr_rate'])
        denitr_rate = st.number_input("反硝化速率 kgNO3/(kgMLSS·d)", value=st.session_state.base_params['denitr_rate'])
        mlvss_mlss = st.number_input("MLVSS / MLSS 比值", value=st.session_state.base_params['mlvss_mlss'])
        carbon_cod_eq = st.number_input("碳源COD当量基准值 (gCOD/g药剂)",
                                        value=st.session_state.base_params['carbon_cod_eq'])

        st.subheader("三、经济成本参数")
        elec_price = st.number_input("电价 (元/kWh)", value=st.session_state.base_params['elec_price'])
        # 除磷药剂双价格
        pac_price = st.number_input("PAC铝盐单价 (元/吨)", value=st.session_state.base_params['pac_price'])
        pfs_price = st.number_input("PFS铁盐单价 (元/吨)", value=st.session_state.base_params['pfs_price'])
        # 四类碳源单价
        naac_price = st.number_input("乙酸钠碳源单价 (元/吨)", value=st.session_state.base_params['naac_price'])
        methanol_price = st.number_input("甲醇碳源单价 (元/吨)", value=st.session_state.base_params['methanol_price'])
        glucose_price = st.number_input("葡萄糖碳源单价 (元/吨)", value=st.session_state.base_params['glucose_price'])
        composite_carbon_price = st.number_input("复合碳源单价 (元/吨)",
                                                 value=st.session_state.base_params['composite_carbon_price'])
        # 其他药剂
        naclo_price = st.number_input("次氯酸钠单价 (元/吨)", value=st.session_state.base_params['naclo_price'])
        pam_price = st.number_input("PAM絮凝剂单价 (元/吨)", value=st.session_state.base_params['pam_price'])
        hcl_price = st.number_input("盐酸单价 (元/吨，pH调节)", value=st.session_state.base_params['hcl_price'])
        # 污泥&人工运维
        sludge_dispose_price = st.number_input("污泥处置单价 (元/吨湿泥)",
                                               value=st.session_state.base_params['sludge_dispose_price'])
        staff_num = st.number_input("运维人员数量 (人)", value=st.session_state.base_params['staff_num'])
        staff_salary = st.number_input("人均月工资 (元)", value=st.session_state.base_params['staff_salary'])
        maintain_cost = st.number_input("月度设备维修费 (元)", value=st.session_state.base_params['maintain_cost'])
        other_cost = st.number_input("月度其他杂费 (元)", value=st.session_state.base_params['other_cost'])

    if st.button("💾 保存全部基础参数", type="primary", use_container_width=True):
        st.session_state.base_params.update({
            'Q_design': Q_design, 'Q_actual': Q_actual, 'Kz': Kz, 'Q_max': Q_max,
            'V_ana': V_ana, 'V_anox1': V_anox1, 'V_aero1': V_aero1,
            'V_anox2': V_anox2, 'V_aero2': V_aero2, 'V_total': V_total,
            'settler_area': settler_area, 'settler_depth': settler_depth,
            'Y': Y, 'Kd': Kd, 'nitr_rate': nitr_rate, 'denitr_rate': denitr_rate,
            'mlvss_mlss': mlvss_mlss, 'carbon_cod_eq': carbon_cod_eq,
            'elec_price': elec_price,
            # 除磷药剂
            'pac_price': pac_price,
            'pfs_price': pfs_price,
            # 四类碳源
            'naac_price': naac_price,
            'methanol_price': methanol_price,
            'glucose_price': glucose_price,
            'composite_carbon_price': composite_carbon_price,
            # 其他药剂（无defoam，替换hcl）
            'naclo_price': naclo_price,
            'pam_price': pam_price,
            'hcl_price': hcl_price,
            'sludge_dispose_price': sludge_dispose_price,
            'staff_num': staff_num, 'staff_salary': staff_salary,
            'maintain_cost': maintain_cost, 'other_cost': other_cost
        })
        st.success("✅ 所有基础参数已保存，全部计算模块将自动调用")


# ================= 页面2：水力与负荷校核 =================
elif page == "💧 水力与负荷校核":
    st.header("💧 水力停留时间与负荷校核")
    bp = st.session_state.base_params

    st.subheader("一、进水水质与运行参数")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: cod_in = st.number_input("进水COD (mg/L)", value=350)
    with col2: bod_in = st.number_input("进水BOD5 (mg/L)", value=180)
    with col3: tn_in = st.number_input("进水总氮 TN (mg/L)", value=40)
    with col4: nh3_in = st.number_input("进水氨氮 NH3-N (mg/L)", value=28)
    with col5: tp_in = st.number_input("进水总磷 TP (mg/L)", value=5)
    with col6: mlss = st.number_input("MLSS (mg/L)", value=3500)

    if st.button("开始校核计算", type="primary"):
        Q = bp['Q_actual']
        V_aero_total = bp['V_aero1']

        # ========== 1. 水力停留时间HRT ==========
        hrt_total = bp['V_total'] / Q * 24
        hrt_ana = bp['V_ana'] / Q * 24
        hrt_anox1 = bp['V_anox1'] / Q * 24
        hrt_aero1 = bp['V_aero1'] / Q * 24
        hrt_anox2 = bp['V_anox2'] / Q * 24
        hrt_aero2 = bp['V_aero2'] / Q * 24
        hrt_aero_total = V_aero_total / Q * 24

        # HRT判定
        hrt_total_status = "✅ 满足" if hrt_total >= 12 else "⚠️ 偏短"
        hrt_ana_status = "✅ 合理" if 1 <= hrt_ana <= 2 else "⚠️ 偏离"
        hrt_anox1_status = "✅ 合理" if 2 <= hrt_anox1 <= 4 else "⚠️ 偏离"
        hrt_aero1_status = "✅ 合理" if 4 <= hrt_aero1 <= 12 else "⚠️ 偏离"
        hrt_anox2_status = "✅ 合理" if 2 <= hrt_anox2 <= 4 else "⚠️ 偏离"
        hrt_aero2_status = "✅ 合理" if 0.5 <= hrt_aero2 <= 2 else "⚠️ 偏离"


        # ========== 3. 污泥负荷 ==========
        # BOD污泥负荷 Ns = 日BOD总量 / 曝气池MLSS总质量  单位：kgBOD/(kgMLSS·d)
        ns_bod = (bod_in * Q / 1000) / (V_aero_total * mlss / 1000)
        # COD污泥负荷
        ns_cod = (cod_in * Q / 1000) / (V_aero_total * mlss / 1000)

        # 污泥负荷判定
        if ns_bod < 0.05:
            ns_status = "⚠️ 负荷过低，污泥易老化"
        elif 0.05 <= ns_bod <= 0.15:
            ns_status = "✅ 脱氮除磷适宜范围"
        else:
            ns_status = "⚠️ 负荷过高，硝化效果受影响"

        # ========== 结果展示 ==========
        st.markdown("---")
        tab1, tab2 = st.tabs(["水力停留时间(HRT)", "污泥负荷校核"])
        with tab1:
            st.subheader("1. 各功能区水力停留时间")
            hrt_df = pd.DataFrame({
                "功能区": ["厌氧池", "第一缺氧池", "第一好氧池", "第二缺氧池", "第二好氧池", "曝气池合计", "生化池总HRT"],
                "停留时间 (h)": [hrt_ana, hrt_anox1, hrt_aero1, hrt_anox2, hrt_aero2, hrt_aero_total, hrt_total],
                "推荐范围 (h)": ["0.5~2", "2~4", "4~12", "2~4", "0.5~2", "4.5~14", "≥12"],
                "判定": [hrt_ana_status, hrt_anox1_status, hrt_aero1_status, hrt_anox2_status, hrt_aero2_status, "—", hrt_total_status]
            })
            st.dataframe(hrt_df, use_container_width=True, hide_index=True)
            st.info("第二好氧池仅用于吹脱氮气与维持DO，不承担硝化功能，停留时间按短HRT设计")


        with tab2:
            st.subheader("2. 系统污泥负荷校核")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("BOD污泥负荷 Ns", f"{ns_bod:.4f} kgBOD/(kgMLSS·d)")
                st.info(ns_status)
                st.caption("脱氮除磷工艺推荐污泥负荷：0.05~0.15 kgBOD/(kgMLSS·d)")
            with col2:
                st.metric("COD污泥负荷", f"{ns_cod:.4f} kgCOD/(kgMLSS·d)")

            st.markdown("---")
            st.write("**污泥负荷调控建议：**")
            st.write("- 负荷过高：加大排泥、提高MLSS、降低进水负荷")
            st.write("- 负荷过低：减少排泥、降低MLSS、缩短污泥龄，防止污泥老化解体")

        st.markdown("---")
        st.subheader("二、综合校核结论")
        conclusion = f"""
        当前工况下，五段Bardenpho(或Phoredox)系统水力停留时间{hrt_total_status.replace('✅ ','').replace('⚠️ ','')}脱氮除磷要求；
        污泥负荷{ns_status.replace('✅ ','').replace('⚠️ ','')}；
        若进水浓度进一步升高，可通过提高MLSS、调控溶解氧、加大内回流比、调整外回流比、补充外加碳源、投加除磷剂等措施保障出水达标。
        """
        st.write(conclusion)


# ================= 页面3：生化核心计算 =================
elif page == "🧪 生化核心计算":
    st.header("🧪 五段Bardenpho生化系统核心计算")
    bp = st.session_state.base_params

    st.subheader("一、运行参数输入")
    col1, col2, col3 = st.columns(3)
    with col1:
        cod_in = st.number_input("进水COD (mg/L)", value=350)
        bod_in = st.number_input("进水BOD5 (mg/L)", value=180)
        nh3_in = st.number_input("进水氨氮 (mg/L)", value=28)
        tn_in = st.number_input("进水总氮 TN (mg/L)", value=40)
        tp_in = st.number_input("进水总磷 TP (mg/L)", value=5)
    with col2:
        tn_out_target = st.number_input("出水TN目标 (mg/L)", value=15)
        tp_out_target = st.number_input("出水TP目标 (mg/L)", value=0.5)
        bod_eff = st.number_input("实际出水BOD5 (mg/L)", value=10)
        cod_eff = st.number_input("实际出水COD (mg/L)", value=50)
        nh3_eff = st.number_input("实际出水氨氮 (mg/L)", value=1.5)
        mlss = st.number_input("MLSS 混合液浓度 (mg/L)", value=3500)
        R = st.number_input("污泥回流比 R (%)", value=100) / 100
        R1 = st.number_input("内回流比 R1 (好氧1→缺氧1, %)", value=200) / 100
        waste_sludge_volume = st.number_input("每日外排剩余污泥量(m³/d)", value=20.0)

    with col3:
        phos_agent_type = st.selectbox("除磷药剂类型", ["聚合氯化铝 PAC（铝盐）", "聚合硫酸铁 PFS（铁盐）"])
        carbon_agent_type = st.selectbox("外加碳源类型", ["乙酸钠", "甲醇", "葡萄糖", "复合碳源"])
        st.info("工艺说明：好氧1完成全部硝化，出水自流进入第二缺氧池深度反硝化；第二好氧池仅吹脱氮气、防止二沉池反硝化，不承担硝化功能")

    if st.button("开始生化计算", type="primary"):
        st.session_state['eff_params'] = {
            'bod_eff': bod_eff,
            'cod_eff': cod_eff,
            'nh3_eff': nh3_eff
        }
        Q = bp['Q_actual']
        f = bp['mlvss_mlss']
        Y = bp['Y']
        Kd = bp['Kd']
        dn_rate = bp['denitr_rate']
        cod_eq = bp['carbon_cod_eq']
        V_total = bp['V_total']
        V_anox1 = bp['V_anox1']
        V_anox2 = bp['V_anox2']

        # ========== 1. 回流比与脱氮校核 ==========
        total_return = R + R1
        # 缺氧1反硝化后硝态氮浓度
        no3_after_anox1 = tn_in / (1 + total_return)
        # 好氧1完成全部硝化，氨氮转化为硝态氮
        no3_aero1 = no3_after_anox1 + nh3_in * 0.95
        # 自流进入缺氧2深度反硝化（反硝化效率按70%计）
        no3_after_anox2 = no3_aero1 * (1 - 0.7)
        # 出水总氮（含残留氨氮、有机氮）
        tn_theory = no3_after_anox2 + 3

        tn_status = "✅ 当前回流比可满足TN目标" if tn_theory <= tn_out_target else "⚠️ 内回流不足，需加大R1回流比"
        min_R1 = (tn_in / tn_out_target - 1 - R) * 100

        # ========== 2. 碳源投加量计算 ==========
        # ========== 药剂参数配置（内置行业标准参数） ==========
        # 除磷药剂：1mg/L TP所需药剂mg/L（摩尔比安全系数1.5，按工业有效含量换算）
        phos_agent_config = {
            "聚合氯化铝 PAC（铝盐）": {
                "dosage_factor": 1.5 * 27 / 31 / 0.1,  # Al2O3分子量102，含量28%  ， Al分子量27，磷31，含量10%
                "price_key": "pac_price"
            },
            "聚合硫酸铁 PFS（铁盐）": {
                "dosage_factor": 1.5 * 56 / 31 / 0.11,  # Fe2O3分子量160，含量19%  ， Fe分子量56，磷31，含量11%
                "price_key": "pfs_price"
            }
        }
        # 碳源药剂：COD当量（gCOD/g药剂）
        carbon_agent_config = {
            "乙酸钠": {"cod_eq": 0.68, "price_key": "naac_price"},
            "甲醇": {"cod_eq": 1.50, "price_key": "methanol_price"},
            "葡萄糖": {"cod_eq": 1.06, "price_key": "glucose_price"},
            "复合碳源": {"cod_eq": 0.85, "price_key": "composite_carbon_price"}
        }

        # ========== 2. 碳源投加量计算 ==========
        tn_remove = tn_in - tn_out_target
        endogenous_carbon = bod_in * 0.7  # 可生化内源碳
        need_carbon_total = tn_remove * 4  # 反硝化总需COD（C/N=4）
        carbon_deficit = max(0, need_carbon_total - endogenous_carbon)
        # 两级缺氧碳源分配 7:3
        carbon_anox1 = carbon_deficit * 0.7
        carbon_anox2 = carbon_deficit * 0.3
        # 按所选碳源换算实际药剂投加量
        carbon_cfg = carbon_agent_config[carbon_agent_type]
        carbon_dosage = carbon_deficit / carbon_cfg["cod_eq"]  # mg/L
        carbon_daily = carbon_dosage * Q / 1000 / 1000  # 吨/天
        cn_ratio = cod_in / tn_in
        carbon_status = "✅ 进水C/N充足，无需外加碳源" if carbon_deficit == 0 else f"⚠️ C/N比仅{cn_ratio:.1f}，需补充外加碳源"

        # ========== 碳磷比计算与判定 ==========
        cp_ratio = bod_in / tp_in
        if cp_ratio < 17:
            cp_status = "⚠️ 碳磷比不足（<17），生物除磷碳源欠缺，需补充碳源强化厌氧释磷"
            cp_need_carbon = True
        else:
            cp_status = "✅ 碳磷比充足（≥17），生物除磷碳源满足需求"
            cp_need_carbon = False

        # ========== 3. 化学除磷药剂计算 ==========
        tp_bio_remove = tp_in * 0.7  # 生物除磷70%
        tp_need_chem = max(0, tp_in - tp_bio_remove - tp_out_target)
        # 按所选除磷药剂换算投加量
        phos_cfg = phos_agent_config[phos_agent_type]
        phos_dosage = tp_need_chem * phos_cfg["dosage_factor"]  # mg/L
        phos_daily = phos_dosage * Q / 1000 / 1000  # 吨/天

        # ========== 4. 剩余污泥与污泥龄（GB50014‑2021标准公式） ==========
        bod_remove = bod_in - bod_eff
        delta_x_v = (Y * Q * bod_remove / 1000) - (Kd * V_total * mlss * f / 1000)
        delta_x_v = max(delta_x_v, 0)
        ash_fraction = 0.20
        delta_x_total = delta_x_v / (1 - ash_fraction)
        water_content = 0.992
        dry_ratio = 1 - water_content
        sludge_wet = delta_x_total / dry_ratio / 1000

        # 2、严格按照你给定的公式单独计算SRT（仅硝化段‑好氧1池）
        V_aero1 = bp["V_aero1"]
        waste_mlss = mlss * 2  # 排泥污泥浓度默认是生化池MLSS的2倍
        # 分子：好氧1池总污泥质量(kg)
        aer1_total_sludge = V_aero1 * mlss / 1000
        # 分母：每日外排污泥质量(kg/d)
        daily_waste_sludge = waste_sludge_volume * waste_mlss / 1000

        # 防止除零报错
        if daily_waste_sludge > 0:
            srt = aer1_total_sludge / daily_waste_sludge
        else:
            srt = 999



        # ========== 5. 污染物去除率（动态计算，无固定值） ==========
        cod_rate = (cod_in - cod_eff) / cod_in * 100
        nh3_rate = (nh3_in - nh3_eff) / nh3_in * 100
        tn_rate = tn_remove / tn_in * 100
        tp_rate = (tp_in - tp_out_target) / tp_in * 100

        # 将更新后的污泥参数存入session_state
        st.session_state.bio_result.update({
            'phos_daily': phos_daily,
            'phos_agent_name': phos_agent_type,
            'phos_price_key': phos_cfg["price_key"],
            'carbon_daily': carbon_daily,
            'carbon_agent_name': carbon_agent_type,
            'carbon_price_key': carbon_cfg["price_key"],
            'sludge_dry_daily': delta_x_total,
            'sludge_wet_daily': sludge_wet,
            'tn_theory': tn_theory,
            'srt': srt,
            'daily_waste_sludge_vol': waste_sludge_volume,
            'waste_mlss': waste_mlss
        })

        # 保存结果供成本模块调用
        st.session_state.bio_result = {
            'phos_daily': phos_daily,
            'phos_agent_name': phos_agent_type,
            'phos_price_key': phos_cfg["price_key"],
            'carbon_daily': carbon_daily,
            'carbon_agent_name': carbon_agent_type,
            'carbon_price_key': carbon_cfg["price_key"],
            'sludge_dry_daily': delta_x_total,
            'sludge_wet_daily': sludge_wet,
            'tn_theory': tn_theory,
            'srt': srt
        }

        # ========== 展示结果 ==========
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["回流比脱氮校核", "DO分区控制", "碳源投加计算", "除磷药剂计算", "污泥与去除率"])

        with tab1:
            st.subheader("1. 回流比校核与脱氮效果")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("污泥回流比", f"{R*100:.0f}%")
                st.metric("内回流比 R1（好氧1→缺氧1）", f"{R1*100:.0f}%")
                st.metric("系统总回流倍数", f"{1+total_return:.1f}倍")
            with col2:
                st.metric("理论出水TN", f"{tn_theory:.2f} mg/L")
                st.metric("达标所需最小内回流R1", f"{max(min_R1, 100):.1f}%")
                st.info(tn_status)
            st.write("💡 调节建议：氨氮偏高时优先加大内回流R1；总氮深度达标可配合缺氧2外加碳源")

        with tab2:
            st.subheader("2. 各功能区溶解氧DO控制标准")
            do_data = pd.DataFrame({
                "功能区": ["厌氧池", "第一缺氧池", "第一好氧池", "第二缺氧池", "第二好氧池"],
                "DO控制范围 (mg/L)": ["< 0.2", "< 0.5", "2.0 ~ 3.0", "< 0.3", "1.0 ~ 2.0"],
                "控制要点": [
                    "保证聚磷菌释磷环境，DO过高会彻底失效除磷",
                    "主反硝化区，控制DO减少碳源浪费",
                    "承担全部硝化功能，保证氨氮充分硝化为硝态氮",
                    "深度反硝化区，DO要求更严格，避免消耗外加碳源",
                    "吹脱氮气，维持出水DO，防止二沉池反硝化浮泥，不承担硝化功能"
                ]
            })
            st.dataframe(do_data, use_container_width=True, hide_index=True)

        with tab3:
            st.subheader(f"3. 碳源投加量计算（{carbon_agent_type}）")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("进水C/N比", f"{cn_ratio:.1f}")
                st.info(carbon_status)
                st.metric("进水碳磷比 (BOD₅/TP)", f"{cp_ratio:.1f}")
                if cp_need_carbon:
                    st.warning(cp_status)
                else:
                    st.success(cp_status)
                st.divider()
                st.write(f"总需脱除总氮：{tn_remove:.1f} mg/L")
                st.write(f"理论总需COD：{need_carbon_total:.1f} mg/L")
                st.write(f"内源可利用碳源：{endogenous_carbon:.1f} mg/L")
                st.write(f"碳源总缺口：{carbon_deficit:.1f} mg/L")
            with col2:
                st.write(f"- 第一缺氧区碳源分配：{carbon_anox1:.1f} mg/L（主反硝化）")
                st.write(f"- 第二缺氧区碳源分配：{carbon_anox2:.1f} mg/L（深度脱氮）")
                st.metric(f"{carbon_agent_type}投加浓度", f"{carbon_dosage:.2f} mg/L")
                st.metric(f"{carbon_agent_type}日投加量", f"{carbon_daily:.3f} 吨/天")

        with tab4:
            st.subheader(f"4. 化学除磷药剂计算（{phos_agent_type}）")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"生物除磷量：{tp_bio_remove:.2f} mg/L")
                st.write(f"需化学去除磷量：{tp_need_chem:.2f} mg/L")
            with col2:
                st.metric(f"{phos_agent_type}投加浓度", f"{phos_dosage:.2f} mg/L")
                st.metric(f"{phos_agent_type}日投加量", f"{phos_daily:.3f} 吨/天")

        with tab5:
            st.subheader("5. 剩余污泥、污泥龄与去除率")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("污泥龄 SRT", f"{srt:.1f} d")
                st.info("✅ 满足硝化菌世代时间要求" if srt > 10 else "⚠️ 泥龄偏短，硝化菌易流失")
                st.metric("每日剩余干污泥", f"{delta_x_total:.2f} kg/d")
                st.metric("湿污泥量（含水率99.2%）", f"{sludge_wet:.2f} m³/d")
            with col2:
                st.write("#### 污染物去除率")
                st.write(f"COD去除率：{cod_rate:.1f}%")
                st.write(f"氨氮去除率：{nh3_rate:.1f}%")
                st.write(f"总氮去除率：{tn_rate:.1f}%")
                st.write(f"总磷去除率：{tp_rate:.1f}%")


# ================= 页面4：二沉池专项校核 =================
elif page == "🏞️ 二沉池专项校核":
    st.header("🏞️ 二沉池专项校核")
    bp = st.session_state.base_params

    col1, col2 = st.columns(2)
    with col1:
        mlss = st.number_input("MLSS 混合液浓度 (mg/L)", value=3500)
        R = st.number_input("污泥回流比 R (%)", value=100) / 100
        sv30 = st.number_input("SV30 沉降比 (%)", value=25)
    with col2:
        Q_max = st.number_input("最大时流量 (m³/h)", value=bp['Q_max'])
        area = st.number_input("二沉池总表面积 (m²)", value=bp['settler_area'])
        depth = st.number_input("二沉池有效水深 (m)", value=bp['settler_depth'])

    if st.button("开始校核", type="primary"):
        # 计算
        q_surface = Q_max / area  # 表面水力负荷
        ssl = Q_max * (1+R) * mlss / 1000 / area * 24  # 固体负荷 kg/(m²·d)
        hrt = area * depth / Q_max  # 停留时间 h
        svi = sv30 * 10 / (mlss / 1000)  # SVI mL/g

        # 判定
        q_status = "✅ 表面负荷正常，沉淀效果良好" if q_surface < 1.5 else "⚠️ 表面负荷偏高，出水SS易超标"
        ssl_status = "✅ 固体负荷在安全范围" if ssl < 150 else "⚠️ 固体负荷过高，易发生跑泥"
        if 70 < svi < 150:
            svi_status = "✅ 污泥沉降性能良好"
        elif svi >= 150:
            svi_status = "⚠️ SVI过高，存在污泥膨胀风险"
        else:
            svi_status = "⚠️ SVI过低，污泥老化"

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("表面水力负荷", f"{q_surface:.3f} m³/(m²·h)")
            st.info(q_status)
            st.caption("推荐值：最大时 ≤ 0.6~1.5 m³/(m²·h)")
        with col2:
            st.metric("固体表面负荷", f"{ssl:.2f} kgMLSS/(m²·d)")
            st.info(ssl_status)
            st.caption("推荐值：≤ 150 kgMLSS/(m²·d)")
        with col3:
            st.metric("SVI 污泥体积指数", f"{svi:.1f} mL/g")
            st.info(svi_status)
            st.caption("正常范围：70 ~ 150 mL/g")

        st.markdown("---")
        st.subheader("故障处置建议")
        if svi >= 150:
            st.warning("污泥膨胀风险：建议降低MLSS、加大排泥量、提高好氧池DO、控制进水有机负荷")
        elif svi < 70:
            st.warning("污泥老化：建议减少排泥、适当提高污泥负荷、检查进水营养比")
        if ssl >= 150:
            st.warning("跑泥风险：建议提高污泥回流比、降低进水量、增加排泥频次")
        st.info("好氧池末端维持1.0~2.0mg/L DO，可有效防止二沉池内反硝化导致的污泥上浮")


# ================= 页面5：工况调节建议 =================
elif page == "⚙️ 工况调节建议":
    st.header("⚙️ 进水水质波动工况智能调节方案")
    st.caption("选择当前异常工况，自动输出全套量化调节参数")

    condition = st.selectbox("选择异常工况", [
        "水量冲击负荷",
        "进水氨氮突高",
        "进水总氮超标 / 低C/N",
        "进水总磷超标",
        "进水SS偏高 / 污泥膨胀风险",
        "进水COD突增",
        "出水SS升高 / 二沉池跑泥"
    ])

    plans = {
        "水量冲击负荷": """
        ### 📌 水量冲击负荷调节方案
        1. **回流系统**
           - 污泥回流比 R：从100%提升至 **120%~150%**，防止二沉池污泥堆积
           - 内回流比 R1：提升至 **250%~300%**，维持脱氮效果
        2. **曝气与DO**
           - 一级好氧DO提高至 **2.5~3.0 mg/L**，防止硝化崩溃
           - 二级好氧DO维持 **1.0~2.0 mg/L**，保证吹脱氮气效果
        3. **污泥控制**
           - 可适当提高MLSS，增强抗冲击能力
           - 排泥量加大 **10%~20%**，避免污泥在二沉池停留过久
        4. **药剂**
           - 可适当增加除磷剂的投加量
           - 若C/N不足，应及时增加碳源投加量
        5. **注意**：加强二沉池巡视，增加SV30监测频次至每2小时一次
        """,

        "进水氨氮突高": """
        ### 📌 进水氨氮突高调节方案
        1. **曝气系统**
           - 第一好氧池DO提升至 **2.5~3.5 mg/L**，强化硝化（全部硝化在好氧1完成）
           - 第二好氧DO同步维持 **1.5~2.0 mg/L**，防止出水带氨
        2. **污泥系统**
           - 排泥量减少 **20%~30%**，延长污泥龄SRT至 **15d以上**，保留硝化菌
           - 可适当提高MLSS
        3. **回流系统**
           - 内回流R1提升至 **250%~300%**，将硝化液充分送回缺氧池
        4. **应急措施**
           - 氨氮超幅>50%时，可临时投加硝化菌剂，缩短恢复周期
           - 除磷药剂维持不变，优先保障硝化
        """,

        "进水总氮超标 / 低C/N": """
                ### 📌 总氮超标 / 低C/N调节方案
                1. **碳源投加**
                   - 加大第一缺氧池碳源投加量，补充反硝化电子供体，优先保障主反硝化段脱氮效率
                   - 同步提升第二缺氧池碳源投加量，强化深度反硝化，进一步削减出水总氮
                   - 优先选用乙酸钠，反硝化速率快、响应及时，适配低C/N下快速提标需求
                2. **溶解氧管控**
                   - 控制好氧1段出水溶解氧水平，降低内回流液携带的溶氧量，避免破坏缺氧1段的缺氧反应环境，保障主反硝化稳定运行
                   - 严格维持缺氧1、缺氧2池内DO＜0.3 mg/L，保证反硝化菌活性与反应速率，避免硝态氮积累导致出水总氮超标
                   - 第二好氧DO控制在1.0~1.5 mg/L，保障末端残留氨氮硝化效果，同时吹脱水中夹带的氮气，改善二沉池污泥沉降性能
                3. **回流比优化**
                   - 提高内回流R1比例，将更多硝态氮输送至缺氧1段进行反硝化，提升系统总氮去除率
                   - 污泥回流比维持现有水平，不宜过高，避免过量溶氧随回流污泥进入厌氧/缺氧段
                4. **工艺调整**
                   - 在出水氨氮稳定达标、硝化效果有富余的前提下，可适度降低MLSS，减少污泥内源呼吸对碳源的无效消耗，将系统有限碳源优先供给反硝化脱氮
                   - 结合水温与出水氨氮动态调控污泥龄，常温工况控制在12~15d；低温期适当延长以保障硝化菌群，高温期可适度缩短以降低内源碳耗
                """,

        "进水总磷超标": """
                ### 📌 进水总磷超标调节方案
                1. **生物除磷强化（优先执行，降低药剂成本）**
                   - 严格控制厌氧池DO＜0.2 mg/L，减少回流污泥、内回流携带的溶解氧，保证聚磷菌厌氧释磷环境
                   - 保障厌氧池易降解碳源供给，进水C/P不足时，可在厌氧池前端补充少量碳源，强化聚磷菌释磷动力，提升后续好氧吸磷效率
                   - 在满足出水氨氮达标的前提下，适度缩短污泥龄SRT，富集聚磷菌；常温市政污水常规控制在8~12d，低温期需兼顾硝化适当延长
                   - 稳定加大剩余污泥排放量，通过排泥将富磷污泥排出系统；排泥过程维持生化池MLSS稳定，避免大幅波动冲击系统
                2. **化学除磷强化（补充生物除磷缺口，保障达标）**
                   - 日常调控优先采用同步投加（好氧池末端/二沉池进水渠），利用混合液紊流完成混凝反应，通过二沉池沉淀去除磷酸盐；超幅较大需应急提标时，启用二沉池后深度处理段后置投加
                   - 投加量按生物除磷后的出水TP缺口动态核算，缺口越大、出水标准越严，摩尔比（安全系数）取高值
                   - 进水TP超幅过大、PAC投加量接近上限时，可换用聚合硫酸铁（PFS）强化除磷；铁盐除磷效率更高，需同步监控出水pH与色度
                3. **配套管控注意事项**
                   - 控制好氧段末端DO不宜过高，避免大量溶解氧随回流污泥进入厌氧池，破坏释磷环境
                   - 强化二沉池运行管控，稳定污泥层高度，避免污泥流失导致颗粒态磷随出水超标
                   - 碳源充足工况优先挖掘生物除磷潜力，减少化学药剂投加量，降低污泥产量与运行成本
                """,

        "进水SS偏高 / 污泥膨胀风险": """
        ### 📌 进水SS偏高 / 污泥膨胀调节方案
        1. **前端预处理**
           - 检查格栅、沉砂池运行状态，强化初沉池沉淀效果
           - 可在初沉池临时投加PAC，降低进水SS负荷
        2. **生化系统**
           - 好氧池DO提高至 **2.5~3.0 mg/L**，抑制丝状菌膨胀
           - 加大排泥量，降低MLSS，缩短污泥龄
           - 严格控制厌氧池DO，防止丝状菌过度繁殖
        3. **药剂辅助**
           - 好氧池可少量投加PAC，改善污泥沉降性能
           - 消泡剂按需投加，防止泡沫夹带污泥流失
        4. **监测**：每2小时测一次SV30和SVI，跟踪沉降性能变化
        """,

        "进水COD突增": """
                ### 📌 进水COD突增调节方案
                1. **曝气系统调控**
                   - 第一好氧池为主降解与硝化段，DO提升至2.5~3.0 mg/L，保障异养菌降解有机物与自养菌硝化的需氧量，避免DO不足导致出水COD、氨氮同步超标
                   - 第二好氧池DO维持1.5~2.0 mg/L，保障末端有机物与氨氮深度处理，同时吹脱水中夹带的氮气，抑制二沉池反硝化浮泥风险
                   - 冲击期间加密监测末端DO，避免过曝气浪费能耗、增加回流带氧损耗
                2. **污泥系统管控**
                   - 稳定控制MLSS
                   - 定期监测SVI，防止高负荷下DO不足诱发丝状菌污泥膨胀
                3. **脱氮除磷优化**
                   - 进水碳源充足时，根据出水总氮、硝态氮数据，在达标前提下逐步减少直至停止外加碳源投加，降低运行成本
                   - 厌氧段碳源提升会强化生物除磷效果，可在出水总磷稳定达标的前提下，适当降低化学除磷药剂投加量
                4. **运行注意事项**
                   - 加密巡查二沉池污泥层高度，防止污泥增殖过快、固体负荷升高引发跑泥
                   - 冲击幅度过大时，优先保障出水COD与氨氮达标，同步管控总氮、总磷指标
                """,

        "出水SS升高 / 二沉池跑泥": """
                ### 📌 出水SS升高 / 二沉池跑泥调节方案
                1. **水力与负荷排查**
                   - 若为进水水量骤增导致表面负荷超限，优先启用调蓄池削峰错峰，控制进水流量；无调蓄条件的严控进水负荷，严禁未经达标处理的超越排放
                   - 核算二沉池固体负荷，若因MLSS过高、回流比过大导致负荷超限，优先加大剩余污泥排放，降低系统污泥总量
                2. **污泥性状排查与处置**
                   - 检测SVI并配合污泥镜检，判断是否发生丝状菌污泥膨胀；若确认膨胀，按污泥膨胀专项方案处置
                   - 若为污泥老化（SVI偏低、絮体细碎、泥质松散），需加大剩余污泥排放量，缩短污泥龄，提高污泥负荷，改善污泥絮凝沉降性能
                   - 若为二沉池反硝化浮泥（泥面夹带小气泡、污泥成片上浮），提高第二好氧池DO，同时加大污泥回流与排泥，减少二沉池污泥停留时间
                3. **运行参数调整**
                   - 二沉池泥层过高、存在跑泥风险时，可临时提高污泥回流比10%~20%，快速压低污泥层高度；需同步核算固体负荷，避免负荷超限加剧跑泥
                   - 反硝化浮泥情况下，适当提升第二好氧池DO至2.0 mg/L左右，抑制二沉池内反硝化反应
                4. **应急处置措施**
                   - 出水SS超标严重时，可在二沉池进水渠临时投加PAC助凝沉淀；非紧急情况不投加PAM，避免长期投加恶化污泥活性
                   - 检查刮吸泥机运行状态，确保排泥通畅；间歇运行设备可临时增加运行频次，及时排出池底积泥
                """,
    }

    st.markdown("---")
    st.markdown(plans[condition])


# ================= 页面6：成本经济核算 =================
elif page == "💰 成本经济核算":
    st.header("💰 全厂运行成本经济核算")
    bp = st.session_state.base_params

    tab1, tab2, tab3, tab4 = st.tabs(["电耗成本", "药剂成本", "污泥处置成本", "全成本汇总"])

    # 电耗成本
    with tab1:
        st.subheader("一、全厂电耗成本核算")
        col1, col2 = st.columns(2)
        with col1:
            aeration_kw = st.number_input("曝气风机总功率 (kW)", value=220)
            backflow_kw = st.number_input("污泥回流泵总功率 (kW)", value=30)
            internal_kw = st.number_input("内回流泵总功率 (kW)", value=45)
            mix_kw = st.number_input("搅拌/推流器总功率 (kW)", value=25)
        with col2:
            pump_kw = st.number_input("进水泵房总功率 (kW)", value=55)
            dewater_kw = st.number_input("污泥脱水系统功率 (kW)", value=37)
            dewater_h = st.number_input("脱水系统日运行时长 (h)", value=8)
            other_kw = st.number_input("辅助设备总功率 (kW)", value=20)

        if st.button("计算电耗成本", type="primary"):
            # 日电耗
            e_aeration = aeration_kw * 24
            e_backflow = backflow_kw * 24
            e_internal = internal_kw * 24
            e_mix = mix_kw * 24
            e_pump = pump_kw * 24
            e_dewater = dewater_kw * dewater_h
            e_other = other_kw * 24

            e_total_day = e_aeration + e_backflow + e_internal + e_mix + e_pump + e_dewater + e_other
            cost_day = e_total_day * bp['elec_price']
            cost_month = cost_day * 30
            Q = bp['Q_actual']
            unit_power = e_total_day / Q
            unit_cost = cost_day / Q

            st.session_state.power_cost_month = cost_month

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### 分项日电耗")
                power_data = pd.DataFrame({
                    "设备类别": ["曝气系统", "污泥回流泵", "内回流泵", "搅拌推流器", "进水泵房", "污泥脱水", "辅助设备"],
                    "日耗电量 (kWh)": [e_aeration, e_backflow, e_internal, e_mix, e_pump, e_dewater, e_other]
                })
                st.dataframe(power_data, use_container_width=True, hide_index=True)
            with col2:
                st.metric("日总耗电量", f"{e_total_day:.1f} kWh")
                st.metric("日电费", f"{cost_day:.2f} 元")
                st.metric("月电费", f"{cost_month:,.2f} 元")
                st.metric("吨水电耗", f"{unit_power:.3f} kWh/m³")
                st.metric("吨水电费", f"{unit_cost:.3f} 元/m³")

            st.info(f"💡 节能提示：曝气系统占总电耗 {e_aeration/e_total_day*100:.1f}%，采用DO变频曝气可节电15%~25%")

    # 药剂成本
    with tab2:
        st.subheader("二、药剂成本核算")
        col1, col2 = st.columns(2)
        with col1:
            naclo_daily = st.number_input("次氯酸钠日用量 (吨)", value=0.5)
            pam_daily = st.number_input("PAM日用量 (吨)", value=0.08)
            hcl_daily = st.number_input("盐酸日用量 (吨，pH调节)", value=0.05)
        with col2:
            st.info("除磷药剂、碳源用量自动读取生化计算结果")
            if st.button("加载生化计算药剂用量"):
                if st.session_state.bio_result:
                    bio = st.session_state.bio_result
                    st.success(f"已加载：{bio['phos_agent_name']} {bio['phos_daily']:.3f}吨/天，{bio['carbon_agent_name']} {bio['carbon_daily']:.3f}吨/天")
                else:
                    st.warning("请先在「生化核心计算」页完成计算")

        if st.button("计算药剂总成本", type="primary"):
            # 优先读生化结果，没有用默认
            bio = st.session_state.bio_result if st.session_state.bio_result else {}
            phos_daily = bio.get('phos_daily', 0.3)
            carbon_daily = bio.get('carbon_daily', 1.2)
            phos_price = bp[bio.get('phos_price_key', 'pac_price')]
            carbon_price = bp[bio.get('carbon_price_key', 'naac_price')]
            phos_name = bio.get('phos_agent_name', 'PAC(除磷)')
            carbon_name = bio.get('carbon_agent_name', '乙酸钠(碳源)')

            cost_phos = phos_daily * phos_price
            cost_carbon = carbon_daily * carbon_price
            cost_naclo = naclo_daily * bp['naclo_price']
            cost_pam = pam_daily * bp['pam_price']
            cost_hcl = hcl_daily * bp['hcl_price']

            total_day = cost_phos + cost_carbon + cost_naclo + cost_pam + cost_hcl
            total_month = total_day * 30
            Q = bp['Q_actual']
            unit_cost = total_day / Q

            st.session_state.med_cost_month = total_month

            col1, col2 = st.columns(2)
            with col1:
                med_data = pd.DataFrame({
                    "药剂名称": [phos_name, carbon_name, "次氯酸钠(消毒)", "PAM(助凝)", "盐酸(pH调节)"],
                    "日用量 (吨)": [phos_daily, carbon_daily, naclo_daily, pam_daily, hcl_daily],
                    "日成本 (元)": [cost_phos, cost_carbon, cost_naclo, cost_pam, cost_hcl]
                })
                st.dataframe(med_data, use_container_width=True, hide_index=True)
            with col2:
                st.metric("日药剂总成本", f"{total_day:.2f} 元")
                st.metric("月药剂总成本", f"{total_month:,.2f} 元")
                st.metric("吨水药剂成本", f"{unit_cost:.3f} 元/m³")

    # 污泥处置成本
    with tab3:
        st.subheader("三、剩余污泥处置成本")
        col1, col2 = st.columns(2)
        with col1:
            water_rate = st.number_input("脱水后污泥含水率 (%)", value=80) / 100
            pam_dosage = st.number_input("吨干泥PAM投加量 (kg/t)", value=4)
        with col2:
            st.info("污泥产量自动读取生化计算结果")
            if st.button("加载生化计算污泥量"):
                if st.session_state.bio_result:
                    st.success(f"已加载：每日干污泥 {st.session_state.bio_result['sludge_dry_daily']:.2f} kg")
                else:
                    st.warning("请先在「生化核心计算」页完成计算")

        if st.button("计算污泥处置成本", type="primary"):
            dry_daily = st.session_state.bio_result.get('sludge_dry_daily', 700)  # kg/d
            wet_daily = dry_daily / (1 - water_rate) / 1000  # 吨/天

            pam_daily = dry_daily / 1000 * pam_dosage / 1000  # 吨/天
            cost_pam_day = pam_daily * bp['pam_price']
            cost_dispose_day = wet_daily * bp['sludge_dispose_price']

            total_day = cost_pam_day + cost_dispose_day
            total_month = total_day * 30
            Q = bp['Q_actual']
            unit_cost = total_day / Q

            st.session_state.sludge_cost_month = total_month

            col1, col2 = st.columns(2)
            with col1:
                st.metric("每日脱水湿污泥量", f"{wet_daily:.2f} 吨")
                st.write(f"脱水PAM日费用：{cost_pam_day:.2f} 元")
                st.write(f"污泥外运处置费：{cost_dispose_day:.2f} 元")
            with col2:
                st.metric("日污泥处置总成本", f"{total_day:.2f} 元")
                st.metric("月污泥处置总成本", f"{total_month:,.2f} 元")
                st.metric("吨水污泥处置成本", f"{unit_cost:.3f} 元/m³")

    # 全成本汇总
    with tab4:
        st.subheader("四、全厂全成本汇总分析")
        if st.button("生成全成本报表", type="primary"):
            power_cost = getattr(st.session_state, 'power_cost_month', 70000)
            med_cost = getattr(st.session_state, 'med_cost_month', 120000)
            sludge_cost = getattr(st.session_state, 'sludge_cost_month', 45000)
            staff_cost = bp['staff_num'] * bp['staff_salary']
            maintain_cost = bp['maintain_cost']
            other_cost = bp['other_cost']

            total_month = power_cost + med_cost + sludge_cost + staff_cost + maintain_cost + other_cost
            Q_month = bp['Q_actual'] * 30
            unit_cost = total_month / Q_month

            # 构造Plotly绘图数据集
            labels = ["电费", "药剂费", "污泥处置", "人员工资", "维修耗材", "其他"]
            values = [power_cost, med_cost, sludge_cost, staff_cost, maintain_cost, other_cost]
            cost_df_plot = pd.DataFrame({
                "成本类别": labels,
                "月度费用": values
            })

            # 绘制环形空心饼图
            fig = px.pie(
                cost_df_plot,
                values="月度费用",
                names="成本类别",
                hole=0.4,  # 空心环形
                color_discrete_sequence=["#36a2eb", "#4bc0c0", "#ff9f40", "#ff6384", "#9966ff", "#c9cbcf"],
                title="月度运行成本构成占比"
            )
            # 配置文字样式，百分比+标签外部展示
            fig.update_traces(
                textposition="outside",
                texttemplate="%{label}<br>%{percent:.1%}",
                textfont_size=14
            )
            fig.update_layout(
                font_size=14,
                showlegend=False,
                title_x=0.5,
                width=700,
                height=600
            )

            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.write("#### 月度成本分项明细")
                cost_data = pd.DataFrame({
                    "成本类别": ["电费", "药剂费", "污泥处置费", "人员工资", "设备维修费", "其他杂费"],
                    "月度费用 (元)": [power_cost, med_cost, sludge_cost, staff_cost, maintain_cost, other_cost],
                    "占比": [f"{v / total_month * 100:.1f}%" for v in values]
                })
                st.dataframe(cost_data, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.metric("📌 月度运行总成本", f"{total_month:,.2f} 元")
                st.metric("📌 年度运行总成本", f"{total_month * 12:,.2f} 元")
                st.metric("📌 吨水处理综合成本", f"{unit_cost:.3f} 元/吨")

            with col2:
                st.plotly_chart(fig, use_container_width=True)


# ================= 页面7：报表导出 =================
elif page == "📊 报表导出":
    st.header("📊 计算报表导出")
    st.caption("将当前所有计算结果汇总导出为中文CSV报表（Excel/WPS直接打开）")

    # 基础参数中英文映射字典（和系统界面完全对应）
    # 基础参数中英文映射字典（和系统界面完全对应，适配多碳源、铝/铁盐除磷、替换盐酸）
    param_name_map = {
        "Q_design": "设计日处理水量 (m³/d)",
        "Q_actual": "实际日均进水量 (m³/d)",
        "Kz": "总变化系数 Kz",
        "Q_max": "最大时流量 (m³/h)",
        "V_ana": "厌氧池有效容积 (m³)",
        "V_anox1": "第一缺氧池有效容积 (m³)",
        "V_aero1": "第一好氧池有效容积 (m³)",
        "V_anox2": "第二缺氧池有效容积 (m³)",
        "V_aero2": "第二好氧池有效容积 (m³)",
        "V_total": "生化池总容积 (m³)",
        "settler_area": "二沉池总表面积 (m²)",
        "settler_depth": "二沉池有效水深 (m)",
        "Y": "污泥产率系数 Y",
        "Kd": "内源衰减系数 Kd (d⁻¹)",
        "nitr_rate": "硝化速率 (kgNH3/(kgMLSS·d))",
        "denitr_rate": "反硝化速率 (kgNO3/(kgMLSS·d))",
        "mlvss_mlss": "MLVSS/MLSS 比值",
        "carbon_cod_eq": "碳源COD当量基准参数 (gCOD/g药剂)",
        "elec_price": "电价 (元/kWh)",
        "pac_price": "聚合氯化铝PAC单价 (元/吨，铝盐除磷)",
        "pfs_price": "聚合硫酸铁PFS单价 (元/吨，铁盐除磷)",
        "naac_price": "乙酸钠碳源单价 (元/吨)",
        "methanol_price": "甲醇碳源单价 (元/吨)",
        "glucose_price": "葡萄糖碳源单价 (元/吨)",
        "composite_carbon_price": "复合碳源单价 (元/吨)",
        "naclo_price": "次氯酸钠单价 (元/吨)",
        "pam_price": "PAM絮凝剂单价 (元/吨)",
        "hcl_price": "盐酸单价 (元/吨，pH调节)",
        "sludge_dispose_price": "污泥处置单价 (元/吨湿泥)",
        "staff_num": "运维人员数量 (人)",
        "staff_salary": "人均月工资 (元)",
        "maintain_cost": "月度设备维修费 (元)",
        "other_cost": "月度其他杂费 (元)"
    }

    if st.button("生成并下载报表", type="primary", use_container_width=True):
        bp = st.session_state.base_params
        bio = st.session_state.bio_result

        all_text = ""
        # 1. 水厂基础参数（中文化）
        all_text += "===== 水厂基础设计参数 =====\n"
        # 转换为中文参数名
        chinese_params = []
        for en_key, value in bp.items():
            cn_name = param_name_map.get(en_key, en_key)
            chinese_params.append({"参数名称": cn_name, "参数值": value})
        base_df = pd.DataFrame(chinese_params)
        all_text += base_df.to_csv(index=False, encoding="utf-8-sig")
        all_text += "\n\n===== 生化系统计算结果 =====\n"

        # 2. 生化计算结果（全中文）
        if bio:
            bio_df = pd.DataFrame({
                "指标名称": [
                    f"{bio.get('phos_agent_name','除磷药剂')}日投加量 (吨/天)",
                    f"{bio.get('carbon_agent_name','碳源')}日投加量 (吨/天)",
                    "每日剩余干污泥量 (kg/d)",
                    "湿污泥量 (m³/d，含水率99.2%)",
                    "理论出水总氮TN (mg/L)",
                    "污泥龄 SRT (d)"
                ],
                "计算结果": [
                    bio['phos_daily'],
                    bio['carbon_daily'],
                    bio['sludge_dry_daily'],
                    bio['sludge_wet_daily'],
                    bio['tn_theory'],
                    bio['srt']
                ]
            })
            all_text += bio_df.to_csv(index=False, encoding="utf-8-sig")
        else:
            all_text += "暂无生化计算数据，请先完成「生化核心计算」\n"
        all_text += "\n\n===== 月度运行成本核算 =====\n"

        # 3. 成本核算（全中文）
        cost_data = {
            "成本类别": [
                "月度电费",
                "月度药剂费",
                "月度污泥处置费",
                "月度人员工资",
                "月度设备维修费",
                "月度其他杂费"
            ],
            "月度金额 (元)": [
                getattr(st.session_state, 'power_cost_month', 0),
                getattr(st.session_state, 'med_cost_month', 0),
                getattr(st.session_state, 'sludge_cost_month', 0),
                bp['staff_num'] * bp['staff_salary'],
                bp['maintain_cost'],
                bp['other_cost']
            ]
        }
        cost_df = pd.DataFrame(cost_data)
        all_text += cost_df.to_csv(index=False, encoding="utf-8-sig")

        st.success("✅ 中文报表生成完成，CSV文件可用Excel/WPS直接打开编辑")
        st.download_button(
            label="📥 下载中文CSV报表",
            data=all_text.encode("utf-8-sig"),
            file_name="五段Bardenpho污水厂运行报表_中文.csv",
            mime="text/csv",
            use_container_width=True
        )