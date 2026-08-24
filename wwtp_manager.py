import base64
from datetime import timedelta
import numpy as np
import os
import re
try:
    import streamlit as st
    from streamlit.components.v1 import html as _st_html
except ImportError:
    st = None
    _st_html = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
try:
    from plotly.subplots import make_subplots
except ImportError:
    make_subplots = None
try:
    import reportlab
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
from io import BytesIO

# 可选：若项目根目录存在 .env，则自动加载其中的环境变量（如 OPENAI_API_KEY）
# 显式按脚本所在目录查找 .env，避免「从上级目录启动 streamlit」时找不到。
# 优先用 python-dotenv；未安装或失败时，用标准库兜底解析，确保任意环境都能生效。
def _load_env_from_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
    except Exception:
        _load_env_from_file(dotenv_path)

# ================= 页面基础配置 =================

# ===== 核心算法 / AI 内核（已内联，单文件自包含，无需 wwtp_core.py）=====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def holt_winters_forecast(y, season=24, h=24, alpha=0.3, beta=0.05, gamma=0.3):
    """Holt-Winters 三指数平滑（加性季节），纯 numpy 实现。
    返回水平/趋势/季节分量与 h 步点预测、残差标准差。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * season:
        season = max(1, n // 2)
    if season < 1:
        season = 1
    l0 = y[:season].mean()
    b0 = (y[season:2 * season].mean() - y[:season].mean()) / season if n >= 2 * season else 0.0
    s0 = y[:season] - l0
    s0 = s0 - s0.mean()
    level, trend = l0, b0
    fitted = np.zeros(n)
    seas = s0.copy()
    for t in range(n):
        if t == 0:
            fitted[t] = level + seas[0]
            continue
        pl, pt = level, trend
        level = alpha * (y[t] - seas[t % season]) + (1 - alpha) * (pl + pt)
        trend = beta * (level - pl) + (1 - beta) * pt
        seas[t % season] = gamma * (y[t] - level) + (1 - gamma) * seas[t % season]
        fitted[t] = level + trend + seas[t % season]
    fc = np.zeros(h)
    resid = y - fitted
    sigma = resid.std(ddof=1) if n > 5 else abs(resid).mean()
    sigma = sigma if sigma > 0 else 1e-6
    for k in range(1, h + 1):
        idx = (n + k - 1) % season
        fc[k - 1] = level + k * trend + seas[idx]
    return {"fitted": fitted, "forecast": fc, "seas": seas,
            "level": level, "trend": trend, "sigma": sigma}


def _harmonic_forecast(y, season, h):
    """谐波回归：OLS 拟合 趋势 + 一/二阶日周期，外推 h 步（纯 numpy）。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * season:
        return np.repeat(y.mean(), h)
    t = np.arange(n, dtype=float)
    X = np.column_stack([
        np.ones(n), t,
        np.sin(2 * np.pi * t / season), np.cos(2 * np.pi * t / season),
        np.sin(4 * np.pi * t / season), np.cos(4 * np.pi * t / season),
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    tt = np.arange(n, n + h, dtype=float)
    Xf = np.column_stack([
        np.ones(h), tt,
        np.sin(2 * np.pi * tt / season), np.cos(2 * np.pi * tt / season),
        np.sin(4 * np.pi * tt / season), np.cos(4 * np.pi * tt / season),
    ])
    return np.clip(Xf @ beta, 0, None)


def _snaive_drift(y, season, h):
    """季节朴素 + 漂移：重复最近一个周期，叠加近期平均增量。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < season:
        return np.repeat(y[-1], h)
    last_season = y[-season:]
    drift = (y[-1] - y[-season]) / season if n > season else 0.0
    out = np.tile(last_season, int(np.ceil(h / season)))[:h] + drift * np.arange(1, h + 1)
    return out


def _backtest(y, season, test_len):
    """留一法回测：用尾部 test_len 点评估各模型精度，返回指标与三模型预测。"""
    n = len(y)
    if n <= test_len + 2 * season:
        return None
    train, test = y[:-test_len], y[-test_len:]
    r = holt_winters_forecast(train, season, h=test_len)
    fc_hw = r["forecast"]
    fc_hm = _harmonic_forecast(train, season, test_len)
    fc_sn = _snaive_drift(train, season, test_len)
    models = {"Holt-Winters": fc_hw, "谐波回归": fc_hm, "季节朴素+漂移": fc_sn}
    metrics = {}
    for name, p in models.items():
        p = np.asarray(p[:len(test)])
        denom = np.where(test == 0, 1e-9, test)
        metrics[name] = {
            "RMSE": float(np.sqrt(np.mean((p - test) ** 2))),
            "MAE": float(np.mean(np.abs(p - test))),
            "MAPE(%)": float(np.mean(np.abs((p - test) / denom)) * 100),
        }
    return metrics, models, test


def smart_forecast(y, season=24, h=24):
    """智能集成预测：多模型回测选优 + 逆误差加权 + 经验置信区间 + 异常检测 + 时序分解。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * season:
        season = max(1, n // 2)
    if season < 1:
        season = 1

    # ---- 回测选优 ----
    test_len = min(h, max(season * 2, n // 4))
    bt = _backtest(y, season, test_len)
    if bt:
        metrics, _, _ = bt
        rmses = {k: v["RMSE"] for k, v in metrics.items()}
        inv = {k: 1.0 / (v + 1e-9) for k, v in rmses.items()}
        tot = sum(inv.values())
        weights = {k: iv / tot for k, iv in inv.items()}
    else:
        metrics, weights = None, {"Holt-Winters": 1.0}

    # ---- 各模型全量预测 ----
    r = holt_winters_forecast(y, season, h=h)
    fc_hw = r["forecast"]
    fc_hm = _harmonic_forecast(y, season, h)
    fc_sn = _snaive_drift(y, season, h)

    # ---- 逆误差加权集成 ----
    if bt:
        fc = (weights["Holt-Winters"] * fc_hw
              + weights["谐波回归"] * fc_hm
              + weights["季节朴素+漂移"] * fc_sn)
    else:
        fc = fc_hw
    fc = np.clip(fc, 0, None)

    # ---- 经验置信区间（基于 HW 残差分位数，比参数法稳健）----
    resid = y - r["fitted"]
    lo_q, hi_q = np.percentile(resid, [5, 95])
    band = max((hi_q - lo_q) / 2.0, 1.96 * r["sigma"])
    lower = fc - band
    upper = fc + band

    # ---- 历史异常检测（|残差| > 3σ）----
    sigma = r["sigma"]
    anomalies = np.where(np.abs(resid) > 3.0 * sigma)[0]

    # ---- 加法分解（趋势 / 季节 / 残差）----
    half = max(1, season // 2)
    trend = np.full(n, np.nan)
    for t in range(half, n - half):
        trend[t] = y[t - half:t + half + 1].mean()
    if n > half:
        trend[0:half] = trend[half]
        trend[n - half:] = trend[n - half - 1]
    detr = y - trend
    seasonal = np.zeros(season)
    for i in range(season):
        vals = detr[i::season]
        seasonal[i] = vals.mean() if len(vals) else 0.0
    seasonal = seasonal - seasonal.mean()
    seas_full = np.tile(seasonal, int(np.ceil(n / season)))[:n]
    resid_full = y - trend - seas_full

    return {
        "series": y, "fitted": r["fitted"], "forecast": fc,
        "lower": lower, "upper": upper, "resid": resid_full,
        "trend": trend, "seasonal": seas_full, "sigma": sigma,
        "anomalies": anomalies, "metrics": metrics, "weights": weights,
        "season": season,
    }


# ============================================================
# 加药优化（线性规划最小成本）
# ============================================================
def optimize_dosing(bp, bio):
    """在满足出水标准约束下，用线性规划最小化药剂成本。
    变量：4 种碳源(mg/L) + 2 种混凝剂(mg/L)；约束：外加碳源 COD 与化学除磷量达标。"""
    Q = float(bp.get('Q_actual', 1e4))
    carbon_deficit = float(bio.get('carbon_deficit', 0.0))   # mg/L COD 缺口
    tp_need_chem = float(bio.get('tp_need_chem', 0.0))       # mg/L 需化学除磷(P)
    carbon_sources = [
        ("乙酸钠", 0.68, float(bp.get('naac_price', 0))),
        ("甲醇", 1.50, float(bp.get('methanol_price', 0))),
        ("葡萄糖", 1.06, float(bp.get('glucose_price', 0))),
        ("复合碳源", 0.85, float(bp.get('composite_carbon_price', 0))),
    ]
    coags = [
        ("PAC", 1.5 * 27 / 31 / 0.10, float(bp.get('pac_price', 0))),
        ("PFS", 1.5 * 56 / 31 / 0.11, float(bp.get('pfs_price', 0))),
    ]
    cur_carbon = float(bio.get('carbon_daily', 0.0))
    cur_phos = float(bio.get('phos_daily', 0.0))
    cur_carbon_price = float(bp.get(bio.get('carbon_price_key', ''), 0))
    cur_phos_price = float(bp.get(bio.get('phos_price_key', ''), 0))
    cur_cost = cur_carbon * cur_carbon_price + cur_phos * cur_phos_price
    rec_carbon = {name: 0.0 for name, _, _ in carbon_sources}
    rec_phos = {name: 0.0 for name, _, _ in coags}
    try:
        from scipy.optimize import linprog
        nc, np_ = len(carbon_sources), len(coags)
        c = [Q / 1e6 * p for _, _, p in carbon_sources] + [Q / 1e6 * p for _, _, p in coags]
        A = [[ce for _, ce, _ in carbon_sources] + [0.0] * np_,
             [0.0] * nc + [1.0 / f for _, f, _ in coags]]
        b = [carbon_deficit, tp_need_chem]
        A_ub = [[-a for a in row] for row in A]
        b_ub = [-bi for bi in b]
        bounds = [(0, None)] * (nc + np_)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError("LP 未收敛")
        x = res.x
        opt_cost = float(res.fun)
        for i, (name, _, _) in enumerate(carbon_sources):
            rec_carbon[name] = x[i] * Q / 1e6
        for j, (name, _, _) in enumerate(coags):
            rec_phos[name] = x[nc + j] * Q / 1e6
    except Exception:
        best_c = min(carbon_sources, key=lambda t: t[2] / max(t[1], 1e-9))
        best_p = min(coags, key=lambda t: t[2] * t[1])
        if carbon_deficit > 0:
            rec_carbon[best_c[0]] = carbon_deficit / best_c[1] * Q / 1e6
        if tp_need_chem > 0:
            rec_phos[best_p[0]] = tp_need_chem * best_p[1] * Q / 1e6
        opt_cost = (rec_carbon[best_c[0]] * best_c[2] if carbon_deficit > 0 else 0) + \
                   (rec_phos[best_p[0]] * best_p[2] if tp_need_chem > 0 else 0)
    saving = cur_cost - opt_cost
    saving_pct = (saving / cur_cost * 100) if cur_cost > 0 else 0.0
    return {"rec_carbon": rec_carbon, "rec_phos": rec_phos,
            "cur_cost": cur_cost, "opt_cost": opt_cost,
            "saving": saving, "saving_pct": saving_pct,
            "carbon_deficit": carbon_deficit, "tp_need_chem": tp_need_chem}


# ============================================================
# 工艺诊断（可解释根因排序）
# ============================================================
def diagnose_process(bp, bio):
    """异常诊断规则引擎：症状 -> 可能根因(置信度) -> 处置建议。可解释 AI。"""
    issues = []
    tn_target = bio.get('tn_out_target', 15)
    if bio.get('tn_theory', 0) > tn_target:
        issues.append((
            f"总氮 TN 超标风险（理论出水 {bio['tn_theory']:.2f} > 目标 {tn_target:.1f} mg/L）",
            [("内回流比 R1 不足，硝态氮回流量不够", f"加大内回流 R1 至 ≥ {max(bio.get('min_R1', 0), 100):.0f}%", 0.9),
             ("外加碳源不足 / 碳氮比偏低", f"C/N={bio.get('cn_ratio', 0):.1f}<4，按反硝化缺口补充碳源", 0.7),
             ("缺氧区 DO 偏高，消耗外加碳源", "控制第一缺氧池 DO<0.5 mg/L", 0.5)]
        ))
    if bio.get('srt', 99) <= 10:
        issues.append((
            f"硝化安全风险（SRT={bio.get('srt', 0):.1f} d ≤ 10 d）",
            [("污泥龄偏短，硝化菌易流失", "降低排泥量或提高好氧1池 MLSS，使 SRT>10 d", 0.9),
             ("低温导致硝化速率下降", "冬季提高 MLSS 或延长 SRT", 0.4)]
        ))
    if bio.get('carbon_deficit', 0) > 0:
        issues.append((
            f"碳源缺口（缺口 {bio['carbon_deficit']:.1f} mg/L COD）",
            [("进水碳氮比不足", "投加外加碳源（乙酸钠/甲醇/复合碳源）", 0.9),
             ("碳磷比不足影响生物除磷", "厌氧段补充碳源强化释磷", 0.5)]
        ))
    if bio.get('tp_need_chem', 0) > 0:
        issues.append((
            f"需化学除磷（化学除磷量 {bio['tp_need_chem']:.2f} mg/L）",
            [("生物除磷能力不足", f"投加 {bio.get('phos_agent_name', '')} 并按需化学除磷量核算", 0.9),
             ("厌氧区硝酸盐回流抑制释磷", "优化污泥回流，降低厌氧区 NO3 进入", 0.5)]
        ))
    if bio.get('cp_need_carbon', False):
        issues.append(("碳磷比不足（<17）", [("生物除磷碳源欠缺", "补充碳源强化厌氧释磷", 0.8)]))
    return issues


# ============================================================
# 参数输入校验（item 8）—— 纯函数，便于单元测试
# ============================================================
def validate_base_params(bp):
    """参数输入校验：返回 (errors, warnings)。
    errors 为阻断级（如流量/容积<=0 将导致除零或 NaN）；warnings 为非致命建议。
    不依赖 Streamlit。
    """
    errors, warnings = [], []

    # 必须为正的关键物理量（参与除法 / 容积计算）
    positive = [
        ("Q_design", "设计日处理水量"), ("Q_actual", "实际日均进水量"),
        ("Q_max", "最大时流量"), ("V_ana", "厌氧池容积"), ("V_anox1", "第一缺氧池容积"),
        ("V_aero1", "第一好氧池容积"), ("V_anox2", "第二缺氧池容积"), ("V_aero2", "第二好氧池容积"),
        ("V_total", "生化池总容积"), ("settler_area", "二沉池表面积"), ("settler_depth", "二沉池有效水深"),
    ]
    for k, label in positive:
        v = bp.get(k)
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            errors.append(f"{label}（{k}）必须为正值")

    # 0~1 比例类
    ratio01 = [("mlvss_mlss", "MLVSS/MLSS"), ("carbon_cod_eq", "碳源COD当量基准")]
    for k, label in ratio01:
        v = bp.get(k)
        if v is None or not isinstance(v, (int, float)) or v < 0 or v > 1:
            warnings.append(f"{label}（{k}）应在 0~1 之间（当前 {v}）")
    # 总变化系数 Kz 通常 ≥1
    kz = bp.get("Kz")
    if kz is not None and isinstance(kz, (int, float)) and kz < 1:
        warnings.append(f"总变化系数 Kz 通常应 ≥1（当前 {kz}）")

    # 动力学系数应为正
    for k, label in [("Y", "污泥产率系数"), ("Kd", "内源衰减系数"),
                     ("nitr_rate", "硝化速率"), ("denitr_rate", "反硝化速率")]:
        v = bp.get(k)
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            errors.append(f"{label}（{k}）必须为正值")

    # 单价 / 人工类不应为负
    for k, label in [("elec_price", "电价"), ("pac_price", "PAC单价"), ("naac_price", "乙酸钠单价"),
                     ("staff_num", "运维人数"), ("staff_salary", "人均工资"),
                     ("sludge_dispose_price", "污泥处置单价")]:
        v = bp.get(k)
        if v is not None and isinstance(v, (int, float)) and v < 0:
            warnings.append(f"{label}（{k}）不应为负（当前 {v}）")

    # 实际水量不应超过设计水量过多
    qd, qa = bp.get("Q_design"), bp.get("Q_actual")
    if isinstance(qd, (int, float)) and isinstance(qa, (int, float)) and qd > 0 and qa > qd * 1.3:
        warnings.append(f"实际进水量（{qa}）超过设计水量（{qd}）30%，可能存在超负荷运行风险")

    return errors, warnings


# ============================================================
# 知识库检索（BM25 风格，轻量 RAG）
# ============================================================
def find_kb_dir():
    """查找知识库目录（可选 RAG 增强）。找不到则返回 None，由规则引擎兜底。"""
    cands = [
        os.path.join(SCRIPT_DIR, "kb"),
        r"D:\应用软件\WorkBuddy\Knowledge_Base",
        os.path.expanduser("~/Knowledge_Base"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "Knowledge_Base"),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def retrieve_kb(query, kb_dir, top_k=3):
    """轻量关键词检索（BM25 风格打分），返回最相关文本块。"""
    chunks = []
    for root, _, files in os.walk(kb_dir):
        for f in files:
            if f.lower().endswith(('.md', '.txt', '.csv')):
                try:
                    txt = open(os.path.join(root, f), encoding='utf-8', errors='ignore').read()
                except Exception:
                    continue
                for blk in re.split(r'\n{2,}', txt):
                    if len(blk.strip()) > 30:
                        chunks.append(blk.strip())
    q = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query))
    scored = []
    for c in chunks:
        words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', c))
        score = len(q & words) / (len(q) + 1)
        if score > 0:
            scored.append((score, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]


# ============================================================
# 规则引擎（item 9）—— 打分意图路由，对问法变化更鲁棒
# ============================================================
def _rule_reply(q, ctx):
    """规则引擎（打分意图路由）：对问题分词，按带权重的关键词/同义词集合打分，
    取最高分且超过阈值的意图作答；未命中任何强意图时给出通用兜底。
    相比硬 `if keyword in q`，对问法变化更鲁棒。"""
    # 意图定义：名称 -> (关键词(含同义词):权重, 回答)
    intents = [
        ("碳源投加", {
            "碳源": 3, "外加碳源": 3, "碳氮": 3, "cn": 2, "c/n": 2, "投加": 1,
            "乙酸钠": 2, "甲醇": 2, "复合碳源": 2, "葡萄糖": 2, "缺碳": 2, "碳不足": 2,
            "反硝化碳": 2, "cod当量": 2,
        }, "【碳源建议】当进水 C/N<4 时需补充外加碳源；优先选 COD 当量高、单价低的碳源"
           "（如甲醇 COD 当量 1.5）。反硝化每去除 1 mg/L NO₃-N 约需 2.86 mg/L COD。"
           "详见『生化核心计算-碳源投加计算』。"),
        ("化学除磷", {
            "除磷": 3, "化学除磷": 3, "磷": 2, "tp": 2, "pac": 2, "pfs": 2,
            "铝盐": 2, "铁盐": 2, "总磷": 2, "bio-p": 1, "释磷": 2, "厌氧除磷": 1,
        }, "【除磷建议】生物除磷约去除 70% 总磷，剩余需化学除磷；铝盐(PAC)/铁盐(PFS)"
           "按摩尔比安全系数 1.5 核算。厌氧段应严格控制 DO<0.2 mg/L 避免抑制释磷。"
           "详见『生化核心计算-除磷药剂计算』。"),
        ("回流与脱氮", {
            "回流": 3, "内回流": 3, "r1": 3, "总氮": 2, "tn": 2, "脱氮": 3,
            "反硝化": 2, "硝态氮": 2, "缺氧": 1, "硝酸盐": 1, "氮去除": 2, "深度脱氮": 2,
        }, "【脱氮建议】总氮深度达标依赖内回流 R1 将硝态氮带回缺氧池反硝化；R1 不足时加大 R1"
           "至达标所需最小值。第一缺氧池 DO 应<0.5 mg/L。详见『生化核心计算-回流比校核』。"),
        ("成本与优化", {
            "成本": 3, "费用": 3, "药耗": 3, "电费": 2, "优化": 2, "节约": 2,
            "药剂": 2, "能耗": 2, "曝气": 2, "省钱": 3, "运行费": 2, "降本": 3,
        }, "【成本建议】可在『成本经济核算-药剂成本』页查看『药剂成本优化方案』，由线性规划在可选药剂中"
           "自动挑选最省组合以最小化药剂成本；曝气能耗可通过优化 DO 设定与精确曝气进一步降低。"),
        ("污泥与泥龄", {
            "污泥": 3, "泥龄": 3, "srt": 3, "排泥": 2, "mlss": 2, "硝化": 2,
            "沉降": 1, "二沉": 1, "剩余污泥": 2, "跑泥": 2, "浮泥": 2,
        }, "【污泥建议】硝化安全泥龄建议 SRT>10 d；泥龄过短硝态菌易流失致氨氮升高，"
           "应降低排泥量或提高 MLSS。二沉池需维持足够泥位与回流比防止跑泥/反硝化浮泥。"),
        ("预测与预警", {
            "预测": 3, "预警": 3, "未来": 2, "趋势": 2, "预报": 2, "负荷": 1,
            "出水水质": 1, "前瞻": 1, "超标风险": 2,
        }, "【预测预警】可在『AI 预测预警』页基于历史数据预测未来进水负荷与出水水质，"
           "并给出达标风险概率。"),
        ("知识库", {
            "知识库": 3, "规范": 2, "标准": 2, "手册": 2, "rag": 2, "规程": 2,
            "国标": 2, "指南": 1, "设计规程": 2,
        }, "【知识检索】当前未配置外部知识库，已使用内置工艺规则作答；"
           "配置 Knowledge_Base 目录后可启用 RAG 检索增强。"),
    ]
    ql = (q or "").lower()
    tokens = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', ql))
    best, best_score = None, 0.0
    for name, kws, answer in intents:
        score = 0.0
        for kw, w in kws.items():
            if kw in ql or kw in tokens:
                score += w
        if score > best_score:
            best, best_score = (name, answer), score
    if best_score >= 2:  # 至少命中一个核心词
        return best[1]
    return ("我已读取当前系统的计算结果（参数/生化/成本）。您可以问我关于：碳源投加、化学除磷、"
            "回流比与脱氮、运行成本优化、污泥龄与排泥、预测预警等方面的问题；"
            "也可先在相关页面完成计算再提问。")


# ============================================================
# LLM 配置读取（st.secrets / 环境变量 / .env 兜底）
# ============================================================
def _get_llm_config():
    """读取 LLM 配置，优先级：st.secrets > 环境变量 > 默认值。
    支持 Streamlit Cloud（secrets.toml）与本地的 .env / os.environ 多种部署方式。"""
    key = base = model = None
    if st is not None:
        try:
            # 本地未配置 secrets.toml 时，st.secrets.get 会抛出
            # StreamlitSecretNotFoundError（该版本未捕获"文件不存在"），
            # 此处整体兜底，使其静默降级到环境变量 / .env。
            s = st.secrets
            key = s.get("OPENAI_API_KEY") or s.get("WWTP_LLM_KEY") or None
            base = s.get("OPENAI_BASE_URL") or None
            model = s.get("OPENAI_MODEL") or None
        except Exception:
            key = base = model = None
    if not key:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("WWTP_LLM_KEY")
    if not base:
        base = os.environ.get("OPENAI_BASE_URL")
    if not model:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return key, base, model


# ============================================================
# LLM 助手（流式，无 Key 时降级规则引擎）
# ============================================================
def ai_assistant_stream(user_q, ctx, kb_dir=None):
    """LLM 工艺助手（流式）：有 Key 时调用大模型逐字输出（支持 RAG），否则降级规则引擎一次性返回。"""
    try:
        key, base, model = _get_llm_config()
        if key:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
            kb = retrieve_kb(user_q, kb_dir) if kb_dir else []
            sys_p = ("你是污水处理厂工艺工程师智能助手，仅依据提供的运行计算结果与知识库作答，"
                     "语言简洁、专业、可操作。\n==== 当前运行数据 ====\n" + ctx)
            if kb:
                sys_p += "\n==== 参考知识库 ====\n" + "\n----\n".join(kb)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_p},
                          {"role": "user", "content": user_q}],
                temperature=0.3,
                stream=True)
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return
    except Exception as e:
        # 大模型调用失败时，把真实异常暴露给 UI（便于排查：网络/路径/鉴权），
        # 然后再降级到规则引擎——保证演示永不断流。
        err_type = type(e).__name__
        err_msg = str(e)[:240].replace("`", "'")
        yield (f"\n\n> ⚠️ **大模型调用失败，已自动降级到规则引擎。**\n>"
               f" 错误类型：`{err_type}`\n>"
               f" 错误信息：`{err_msg}`\n"
               f"> 💡 常见原因：①网络/代理阻断 ②base_url 路径错（GLM 需 `…/api/paas/v4/`） ③Key 失效或欠费\n\n---\n\n")
    yield _rule_reply(user_q, ctx)


# ============================================================
# LLM 配置自检（offline / local / cloud）
# ============================================================
def check_llm_config():
    """启动时自检 LLM 配置，返回 (mode, detail, ok)。
    mode: offline(离线规则引擎) / local(本地自托管) / cloud(云端) / unknown。
    支持 st.secrets、环境变量与 .env 文件三种来源。"""
    key, base, model = _get_llm_config()
    if not key and not base:
        return ("offline", "未配置 Key/BaseURL，AI 助手以「离线规则引擎」运行（无需联网/Key，任意网络可用）", True)
    if base and ("localhost" in base or "127.0.0.1" in base or "0.0.0.0" in base):
        try:
            import urllib.request
            import json as _json
            url = base.rstrip("/").rsplit("/v1", 1)[0] + "/api/tags"
            with urllib.request.urlopen(url, timeout=2) as r:
                tags = _json.loads(r.read().decode()).get("models", [])
            names = [m.get("name") for m in tags]
            if model in names:
                return ("local", f"本地自托管已就绪（Ollama，模型 {model}）", True)
            if not names:
                return ("local", f"本地 Ollama 已启动，但尚未拉取模型；请先 `ollama pull {model}`", False)
            return ("local", f"Ollama 已启动但未找到 {model}；可用：{names}。请先 `ollama pull {model}`", False)
        except Exception as e:
            return ("local", f"检测到本地地址 {base} 但无法连接（{e}）。请确认已执行 `ollama serve`", False)
    if key and base:
        return ("cloud", f"云端大模型已配置（{base}，模型 {model}）", True)
    if key and not base:
        return ("cloud", "已设 Key 但未设 OPENAI_BASE_URL，将直连 OpenAI 官方（国内可能不通）", False)
    return ("unknown", "配置不完整", False)


# ============================================================
# 预测预警增强（item 10）—— 概率化与机理联动（纯函数，可单测）
# ============================================================
def _norm_cdf(x):
    """标准正态 CDF 近似（Abramowitz-Stegun 公式），无需 scipy。支持标量或数组输入。"""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.sign(x) * np.sqrt(np.clip(1.0 - np.exp(-2.0 * x * x / np.pi), 0.0, 1.0)))


def mechanism_advice(var, bio=None):
    """根据超标变量给出机理联动处置建议（纯函数，bio 为生化计算结果 dict）。"""
    v = (var or "")
    vl = v.lower()
    if "tn" in vl or "总氮" in v:
        r1 = (bio or {}).get("min_R1", 100)
        return (f"总氮深度达标依赖内回流 R1 将硝态氮带回缺氧池反硝化："
                f"建议将 R1 提高至 ≥{max(r1, 100):.0f}%，并控制第一缺氧池 DO<0.5 mg/L；"
                f"若 C/N<4 需同步补充外加碳源。")
    if "tp" in vl or "磷" in v:
        return "总磷超标时优先确认生物除磷是否充足（厌氧段 DO<0.2、无硝酸盐回流抑制），必要时提高 PAC/PFS 投加量并按摩尔比 1.5 核算。"
    if "nh3" in vl or "氨氮" in v:
        return "氨氮超标多因硝化不足：建议提高好氧1池 DO 至 2~3 mg/L、延长 SRT>10 d、必要时提高 MLSS。"
    if "cod" in vl:
        return "出水 COD 偏高可能与二沉池泥水分离或前置生化不完全有关：检查二沉池泥位/回流比，确认好氧段曝气充足。"
    return "建议结合『生化核心计算』复核回流比、碳源与除磷药剂投加，并关注二沉池运行状况。"


def influent_surge_note(var, series, forecast, season=24):
    """若预测的是进水类变量且呈显著上升冲击，给出负荷冲击机理提示（纯函数）。"""
    if not any(k in (var or "") for k in ["进水", "流量", "负荷"]):
        return None
    hist = np.asarray(series, dtype=float)
    if len(hist) < season or season < 1:
        return None
    base = hist[-season:].mean()
    if base <= 0:
        return None
    fut = np.asarray(forecast, dtype=float)
    peak_rise = (fut.max() - base) / base * 100.0
    if peak_rise >= 15.0:
        return (f"预测显示进水负荷峰值较近期均值上升约 {peak_rise:.0f}%，"
                f"易形成冲击负荷：建议提前预留调节池缓冲、适当提高污泥浓度(MLSS)与内回流 R1，"
                f"防止出水水质波动与二沉池跑泥。")
    return None


def _build_sample_df(hours=24*60, seed=20260808):
    """生成 AI 预测页使用的合成演示数据（与 gen_sample_data.py 等价）。
    工况设定：市政污水，设计规模 2 万 m³/d，二沉池后接浸没式超滤（UF）深度处理，
    出水执行准四类标准（COD≤30、NH3-N≤1.5、TN≤15、TP≤0.3）；
    NH₃-N 实时检测值围绕 1.5×30% = 0.45 mg/L 上下波动（实际 0.34–0.56 mg/L），
    远低于 1.5 限值、稳定达标，可清晰展示系统的处理余量。
    部署时若 sample_wwtp_history.csv 缺失，可即时在内存生成，无需额外文件。"""
    if pd is None:
        raise RuntimeError("需要 pandas 才能生成示例数据")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-06-01 00:00")
    t = pd.date_range(start, periods=hours, freq="h")
    hour = t.hour.to_numpy()
    dow = t.dayofweek.to_numpy()
    day_idx = np.arange(hours)

    # 日变化系数：市政污水早晚两个高峰
    diurnal = 0.5 * np.sin(2 * np.pi * (hour - 4) / 24) + 0.5 * np.sin(2 * np.pi * (hour - 16) / 24)
    diurnal = (diurnal - diurnal.min()) / (diurnal.max() - diurnal.min())
    week_factor = np.where(dow >= 5, 0.92, 1.0)
    trend = 1.0 + 0.06 * np.sin(2 * np.pi * day_idx / (24 * 30))

    # 设计规模 2 万 m³/d → 平均 833 m³/h；归一化使日均流量精确等于 2 万方，峰值约 1100 m³/h（Kz≈1.32）
    base_flow = 20000.0 / 24.0
    shape = (0.55 + 0.55 * diurnal) * week_factor * trend
    shape *= (1 + rng.normal(0, 0.05, hours))
    shape /= shape.mean()
    flow = base_flow * shape
    flow = np.clip(flow, 450, 1100)

    # 进水：典型市政污水（COD 250–450 / NH3-N 30–50 / TN 45–60 / TP 4–7）
    def polluant(base, cv, spike_prob, spike_mul):
        val = base * (1 + rng.normal(0, cv, hours))
        spike = rng.random(hours) < spike_prob
        val[spike] *= rng.uniform(spike_mul, spike_mul + 0.6, spike.sum())
        return np.clip(val, base * 0.3, None)

    cod_in = polluant(350, 0.12, 0.04, 1.4)
    nh3_in = polluant(35, 0.12, 0.03, 1.3)
    tn_in = nh3_in + polluant(20, 0.10, 0.02, 1.2)
    tp_in = polluant(5.0, 0.12, 0.03, 1.4)

    # 实时检测排放值：围绕 1.5×30% = 0.45 mg/L 上下波动（NH₃-N 特殊项），其余指标落在
    # 对应设计限值的 20%–60% 区间内。中心值稳定远离限值，保留处理余量。
    def effluent_in_range(limit, lo=0.20, hi=0.60):
        lo_v, hi_v = lo * limit, hi * limit
        center = (lo_v + hi_v) / 2.0
        half = (hi_v - lo_v) / 2.0
        # 3 天慢漂移 + 小幅噪声，使序列在区间内自然波动
        drift = 0.5 * np.sin(2 * np.pi * day_idx / (24 * 3) + 1.0)
        val = center + half * drift + rng.normal(0, half * 0.22, hours)
        return np.clip(val, lo_v * 0.95, hi_v * 1.03)

    cod_out = effluent_in_range(30.0)    # 6.0–18.0 mg/L
    nh3_out = effluent_in_range(0.45, lo=0.80, hi=1.20)   # 0.36–0.54 mg/L，围绕 0.45 波动
    tn_out = effluent_in_range(15.0)     # 3.0–9.0 mg/L
    tp_out = effluent_in_range(0.3)      # 0.06–0.18 mg/L

    # 浸没式超滤（UF）深度处理：截留悬浮物/胶体/浊度，溶解态污染物基本不变
    # UF 出水 COD 略低于二沉出水（去除颗粒态部分），浊度 <0.1 NTU，跨膜压差 TMP 随运行波动
    uf_cod_out = cod_out * (1 - rng.uniform(0.02, 0.06, hours))
    uf_turb = rng.uniform(0.03, 0.09, hours)
    tmp = 18.0 + 10.0 * np.sin(2 * np.pi * day_idx / (24 * 7)) + rng.normal(0, 1.2, hours)

    energy = 1500 + 0.40 * flow + rng.normal(0, 40, hours)
    energy = np.clip(energy, 900, None)
    chem = 30 + 0.02 * flow + 6 * (tp_out > 0.19) + rng.normal(0, 2.5, hours)
    chem = np.clip(chem, 10, None)

    return pd.DataFrame({
        "时间": t.strftime("%Y-%m-%d %H:%M"),
        "进水流量(m3/h)": np.round(flow, 1),
        "进水COD(mg/L)": np.round(cod_in, 1),
        "进水NH3-N(mg/L)": np.round(nh3_in, 1),
        "进水TN(mg/L)": np.round(tn_in, 1),
        "进水TP(mg/L)": np.round(tp_in, 2),
        "出水COD(mg/L)": np.round(cod_out, 1),
        "出水NH3-N(mg/L)": np.round(nh3_out, 2),
        "出水TN(mg/L)": np.round(tn_out, 2),
        "出水TP(mg/L)": np.round(tp_out, 3),
        "UF出水COD(mg/L)": np.round(uf_cod_out, 1),
        "UF出水浊度(NTU)": np.round(uf_turb, 2),
        "跨膜压差(kPa)": np.round(tmp, 1),
        "电耗(kWh/h)": np.round(energy, 1),
        "药耗(kg/h)": np.round(chem, 1),
    })


def simulate_plant(n_steps=24*60, dt=1.0, params=None, load_profile=None,
                   control=None, seed=20260808):
    """五段 Bardenpho 虚拟水厂机理仿真引擎（Phase 1 数据源，双驱动之「机理」翼）。

    用简化动力学（Monod 型硝化/反硝化、DO-曝气-能耗耦合、膜污染、药耗成本）
    生成物理自洽、强耦合的「虚拟传感器」时序数据——进水氨氮升高会自动传导为
    曝气量↑、风机功率↑(三次方)、能耗↑、出水 NH3 波动。数据遵循物理因果而非
    随机噪声，为后续 AI 预测/优化/诊断/协同提供可信数据源。

    数据契约（DataFrame 列）：
      [兼容旧版] 时间 / 进水流量 / 进水COD/NH3-N/TN/TP / 出水COD/NH3-N/TN/TP /
                 UF出水COD / UF出水浊度 / 跨膜压差 / 电耗 / 药耗
      [机理扩展] DO_好氧池 / DO_缺氧1 / DO_缺氧2 / MLSS / 水温 / 曝气量 /
                 风机功率 / 泵功率 / 内回流比 / 污泥回流比 / 碳源投加 /
                 除磷剂投加 / 运行成本 / 吨水电耗 / 达标 / 超标项 / 反洗事件
    控制量（control 字典）为 Phase 3「AI 优化→一键执行」预留的决策变量：
      DO_setpoint / R_internal / R_sludge / carbon_active / chem_p_active。

    纯函数：不依赖 Streamlit，可单测。数值稳定：物理量经 clip 约束，无 NaN/Inf。
    """
    rng = np.random.default_rng(seed)

    # ---- 默认工艺参数（可覆盖） ----
    p = dict(
        Q_design=20000.0 / 24.0,   # m³/h 设计小时流量（2 万 m³/d）
        MLSS=4.0,                  # g/L 好氧池污泥浓度
        V_ae=3000.0,               # m³ 好氧池有效容积
        DO_sat=9.0,                # mg/L 饱和溶解氧
        T_water=18.0,              # ℃ 水温
        aer_rated_kw=220.0,        # kW 曝气风机额定功率
        pump_rated_kw=45.0,        # kW 内/外回流泵额定功率
        elec_price=0.7,            # 元/kWh
        carbon_price=2.5,          # 元/kg 碳源（乙酸钠）
        chem_p_price=3.0,          # 元/kg 除磷剂（PAC）
        tmp_init=14.0,             # kPa 反洗后 TMP 基线
        tmp_wash=26.0,             # kPa 反洗触发阈值
        k_foul=0.055,              # kPa/h 膜污染速率系数
    )
    if params:
        p.update(params)
    Q_d = float(p["Q_design"])

    # ---- 控制变量（默认固定控制；Phase 3 由 AI 优化模型接管） ----
    c = dict(DO_setpoint=2.0, R_internal=3.0, R_sludge=1.0,
             carbon_active=True, chem_p_active=True)
    if control:
        c.update(control)

    # ---- 时间轴 ----
    if dt == 1.0:
        freq = "h"
    elif dt < 1.0:
        freq = f"{int(round(dt * 60))}min"
    else:
        freq = f"{int(dt)}h"
    start = pd.Timestamp("2026-06-01 00:00")
    t = pd.date_range(start, periods=n_steps, freq=freq)
    hour = t.hour.to_numpy(dtype=float)
    dow = t.dayofweek.to_numpy(dtype=float)
    day_idx = np.arange(n_steps)

    # ---- 负荷发生器：日双峰 + 周因子 + 30 天慢趋势 + 冲击事件 ----
    diurnal = 0.5 * np.sin(2 * np.pi * (hour - 4) / 24) + 0.5 * np.sin(2 * np.pi * (hour - 16) / 24)
    diurnal = (diurnal - diurnal.min()) / max(diurnal.max() - diurnal.min(), 1e-9)
    week_f = np.where(dow >= 5, 0.92, 1.0)
    trend = 1.0 + 0.06 * np.sin(2 * np.pi * day_idx / (24 * 30.0))

    impact = np.ones(n_steps)
    if load_profile is not None:
        impact = np.asarray(load_profile, dtype=float) * impact
    else:
        n_evt = max(int(n_steps * 0.012), 1) if n_steps > 80 else 0
        for _ in range(n_evt):
            i0 = int(rng.integers(0, max(n_steps - 20, 1)))
            dur = int(rng.integers(8, 17))
            mult = float(rng.uniform(1.25, 1.6))
            impact[i0:i0 + dur] = np.maximum(impact[i0:i0 + dur], mult)
    impact = impact * (1 + rng.normal(0, 0.04, n_steps))

    shape = (0.55 + 0.55 * diurnal) * week_f * trend * impact
    shape /= shape.mean()
    Q = np.clip(Q_d * shape, Q_d * 0.5, Q_d * 1.6)

    def _inload(base, cv, q_dep):
        v = base * (1 + q_dep * (shape / shape.mean() - 1)) * (1 + rng.normal(0, cv, n_steps))
        return np.clip(v, base * 0.4, base * 2.2)

    cod_in = _inload(350.0, 0.10, 0.25)
    nh3_in = _inload(35.0, 0.12, 0.30)
    org_n = _inload(18.0, 0.10, 0.25)
    tn_in = np.clip(nh3_in + org_n, 30.0, 90.0)
    tp_in = _inload(5.0, 0.12, 0.28)
    T = float(p["T_water"])
    fT = 1.0 + 0.02 * (T - 18.0)   # 温度对硝化速率小调制

    # ---- 逐时推进（简化集中参数动力学，保持物理因果） ----
    n = n_steps
    K_DO = 1.0
    do_ae = np.full(n, float(c["DO_setpoint"])) * (1 + rng.normal(0, 0.02, n))
    do_ax1 = np.full(n, 0.2) * (1 + rng.normal(0, 0.05, n))
    do_ax2 = np.full(n, 0.3) * (1 + rng.normal(0, 0.05, n))
    mlss = np.full(n, float(p["MLSS"])) * (1 + rng.normal(0, 0.03, n))
    cod_out = np.zeros(n); nh3_out = np.zeros(n); tn_out = np.zeros(n); tp_out = np.zeros(n)
    uf_cod = np.zeros(n); uf_turb = np.zeros(n)
    aer_flow = np.zeros(n); aer_power = np.zeros(n); pump_power = np.zeros(n)
    carbon_dose = np.zeros(n); chem_p = np.zeros(n)
    tmp = np.zeros(n); wash = np.zeros(n, dtype=int)
    energy = np.zeros(n); op_cost = np.zeros(n)
    std_ok = np.ones(n, dtype=int); std_fail = [""] * n
    tmp_v = float(p["tmp_init"])

    for i in range(n):
        d = float(do_ae[i])
        # 硝化：DO 指数调制（DO 不足 → 出水 NH3 恶化，物理正确）
        eta_nit = 1.0 - 0.015 * np.exp(3.2 * max(1.0 - d / 2.0, 0.0)) * (1.0 / fT)
        eta_nit = float(np.clip(eta_nit, 0.55, 0.992))
        nh3_out[i] = float(np.clip(nh3_in[i] * (1.0 - eta_nit) + 0.05, 0.02, 30.0))

        # COD 去除：基准 + DO/MLSS 小调制（常态去除率 ≥93%，冲击负荷时偶发超标）
        dof = d / (K_DO + d)
        eta_cod = float(np.clip(0.965 * (0.92 + 0.08 * dof) * (mlss[i] / 4.0) ** 0.1, 0.90, 0.985))
        cod_out[i] = float(np.clip(cod_in[i] * (1.0 - eta_cod) + 2.0, 1.0, 60.0))

        # 反硝化：C/N 充足度 + 缺氧池 DO；碳源不足时投加（药耗↑、TN↓）
        cn = (cod_in[i] * 0.4) / max(tn_in[i], 1e-9)
        eta_denit = 0.78 + 0.05 * min(cn / 4.0, 1.2) - 0.12 * do_ax1[i] / (0.2 + do_ax1[i])
        if c["carbon_active"] and cn < 4.0:
            eta_denit += 0.04
            carbon_dose[i] = max((4.0 - cn) * tn_in[i] * Q[i] / 1000.0 * 0.06, 0.0)
        eta_denit = float(np.clip(eta_denit, 0.55, 0.86))
        tn_out[i] = float(np.clip(tn_in[i] * (1.0 - eta_denit) + 0.5, 0.5, 40.0))

        # 除磷：生物去除 + 化学投加（按需投加至目标出水 TP，稳定达标）
        tp_bio = float(np.clip(tp_in[i] * 0.30 + 0.02, 0.02, 2.0))
        tp_target = 0.15
        if c["chem_p_active"] and tp_bio > tp_target:
            need_rem = tp_bio - tp_target
            chem_p[i] = need_rem * Q[i] / 1000.0 * 1.8   # kg/h（PAC，1.8 折算系数）
            tp_out[i] = float(np.clip(tp_target + rng.normal(0, 0.01), 0.02, 2.0))
        else:
            tp_out[i] = tp_bio

        # DO 平衡 → 曝气量 → 风机功率（风机相似定律：功率 ∝ 流量³）
        our_nit = 4.57 * (nh3_in[i] - nh3_out[i]) * Q[i] / p["V_ae"]
        our_cod = 0.55 * (cod_in[i] - cod_out[i]) * Q[i] / p["V_ae"]
        our = max(our_nit + our_cod + 0.3, 0.3)
        kla_req = (our + d * 0.02) / max(p["DO_sat"] - d, 0.1)
        aer_flow[i] = float(kla_req * p["V_ae"] / 16.0)   # /16 标定：典型 1500–3800 m³/h，避免功率恒饱和
        aer_power[i] = float(p["aer_rated_kw"] * (min(aer_flow[i] / 4000.0, 1.0)) ** 3)

        # 回流泵功率（∝ 流量 × 回流强度）
        pump_power[i] = float(p["pump_rated_kw"] * (Q[i] / Q_d) *
                              (0.7 + 0.15 * c["R_internal"] + 0.08 * c["R_sludge"]))

        # 能耗与运行成本
        energy[i] = aer_power[i] + pump_power[i]
        op_cost[i] = (energy[i] * dt * p["elec_price"]
                      + carbon_dose[i] * dt * p["carbon_price"]
                      + chem_p[i] * dt * p["chem_p_price"])

        # UF 跨膜压差：污染上升 — 反洗回落
        flux = Q[i] / Q_d
        tmp_v += p["k_foul"] * flux * dt
        if tmp_v >= p["tmp_wash"]:
            tmp_v = float(p["tmp_init"]) * (1.0 + rng.normal(0, 0.01))
            wash[i] = 1
        tmp[i] = tmp_v
        uf_cod[i] = cod_out[i] * (1.0 - float(rng.uniform(0.02, 0.05)))
        uf_turb[i] = float(rng.uniform(0.03, 0.09))

        # 达标（准四类：COD≤30、NH3-N≤1.5、TN≤15、TP≤0.3）
        fails = []
        if cod_out[i] > 30: fails.append("COD")
        if nh3_out[i] > 1.5: fails.append("NH3-N")
        if tn_out[i] > 15: fails.append("TN")
        if tp_out[i] > 0.3: fails.append("TP")
        std_ok[i] = 0 if fails else 1
        std_fail[i] = "、".join(fails)

    return pd.DataFrame({
        "时间": t.strftime("%Y-%m-%d %H:%M"),
        "进水流量(m3/h)": np.round(Q, 1),
        "进水COD(mg/L)": np.round(cod_in, 1),
        "进水NH3-N(mg/L)": np.round(nh3_in, 1),
        "进水TN(mg/L)": np.round(tn_in, 1),
        "进水TP(mg/L)": np.round(tp_in, 2),
        "出水COD(mg/L)": np.round(cod_out, 1),
        "出水NH3-N(mg/L)": np.round(nh3_out, 2),
        "出水TN(mg/L)": np.round(tn_out, 2),
        "出水TP(mg/L)": np.round(tp_out, 3),
        "UF出水COD(mg/L)": np.round(uf_cod, 1),
        "UF出水浊度(NTU)": np.round(uf_turb, 2),
        "跨膜压差(kPa)": np.round(tmp, 1),
        "电耗(kWh/h)": np.round(energy, 1),
        "药耗(kg/h)": np.round(carbon_dose + chem_p, 1),
        "DO_好氧池(mg/L)": np.round(do_ae, 2),
        "DO_缺氧1(mg/L)": np.round(do_ax1, 2),
        "DO_缺氧2(mg/L)": np.round(do_ax2, 2),
        "MLSS(g/L)": np.round(mlss, 2),
        "水温(℃)": np.round(np.full(n, T), 1),
        "曝气量(m3/h)": np.round(aer_flow, 0),
        "风机功率(kW)": np.round(aer_power, 1),
        "泵功率(kW)": np.round(pump_power, 1),
        "内回流比": np.full(n, c["R_internal"]),
        "污泥回流比": np.full(n, c["R_sludge"]),
        "碳源投加(kg/h)": np.round(carbon_dose, 2),
        "除磷剂投加(kg/h)": np.round(chem_p, 2),
        "运行成本(元/h)": np.round(op_cost, 1),
        "吨水电耗(kWh/m3)": np.round(energy / Q, 3),
        "达标": std_ok,
        "超标项": std_fail,
        "反洗事件": wash,
    })


def gen_plant_csv(path=None, n_steps=24 * 60, seed=20260808, **kw):
    """用机理引擎生成历史运行 CSV 并落盘（Phase 2 起作为 AI 页数据源）。
    返回文件路径；path 缺省时输出到脚本目录 plant_sim_history.csv。"""
    df = simulate_plant(n_steps=n_steps, seed=seed, **kw)
    path = path or os.path.join(SCRIPT_DIR, "plant_sim_history.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_pdf_report(bp, bio, cost):
    """生成排版精美的中文 PDF 运行报表，返回 BytesIO。

    中文字体嵌入策略（确保电脑/手机均可正确显示）：
      1. 优先使用系统无衬线 CJK 字体（Windows 微软雅黑/黑体、macOS PingFang、Linux Noto/WQY）；
      2. 找不到时回退到 reportlab 内置 STSong-Light（宋体 CID，免字体文件）。
    因此生成的 PDF 自带中文字形，在 Windows / macOS / Linux / Android / iOS 上都不会出现方块。
    纯函数：不依赖 Streamlit，便于测试。
    """
    if not HAS_REPORTLAB:
        raise ModuleNotFoundError(
            "reportlab 未安装。请在 requirements.txt 中添加 'reportlab==4.2.2' 并重新部署，"
            "或在当前环境执行：pip install reportlab==4.2.2"
        )
    import io
    from datetime import datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, PageBreak)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.legends import Legend

    # ---------- 中文字体 ----------
    cjk = "STSong-Light"
    _ttf = [
        r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for fp in _ttf:
        if os.path.exists(fp):
            try:
                pdfmetrics.registerFont(TTFont("CJK", fp, subfontIndex=0))
                cjk = "CJK"
                break
            except Exception:
                continue
    if cjk == "STSong-Light":
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        except Exception:
            pass

    # ---------- 样式 ----------
    NAVY = colors.HexColor("#0b3d63")
    LIGHT = colors.HexColor("#eef4f8")
    GRID = colors.HexColor("#bcccd9")

    def _table_style():
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), cjk),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, GRID),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])

    ss = getSampleStyleSheet()
    st_title = ParagraphStyle("t", parent=ss["Title"], fontName=cjk, fontSize=20,
                              leading=26, textColor=NAVY, alignment=TA_CENTER)
    st_sub = ParagraphStyle("sub", parent=ss["BodyText"], fontName=cjk, fontSize=10.5,
                            leading=15, textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
    st_h = ParagraphStyle("h", parent=ss["Heading1"], fontName=cjk, fontSize=14,
                          leading=18, textColor=NAVY, spaceBefore=12, spaceAfter=6)
    st_body = ParagraphStyle("b", parent=ss["BodyText"], fontName=cjk, fontSize=9.5, leading=14)
    st_small = ParagraphStyle("s", parent=ss["BodyText"], fontName=cjk, fontSize=8,
                              leading=11, textColor=colors.HexColor("#666666"))
    st_cell = ParagraphStyle("c", parent=ss["BodyText"], fontName=cjk, fontSize=9, leading=12)
    st_cellw = ParagraphStyle("cw", parent=st_cell, textColor=colors.white)
    st_bullet = ParagraphStyle("bl", parent=ss["BodyText"], fontName=cjk, fontSize=9.5,
                               leading=15, leftIndent=10)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=16 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            title="五段Bardenpho污水厂运行报表")
    flow = []

    # ---------- 封面 ----------
    flow.append(Paragraph("五段 Bardenpho 污水处理厂", st_title))
    flow.append(Paragraph("运行计算报表", st_title))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph("工艺校核 · 生化计算 · 成本分析 · AI 辅助决策", st_sub))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(f"报表生成时间：{datetime.now():%Y-%m-%d %H:%M}", st_small))
    flow.append(Spacer(1, 10))

    if cost:
        tn_txt = f"<b>{bio['tn_theory']:.2f}</b><br/>理论出水 TN (mg/L)" if bio else "—"
        kpi = [[
            Paragraph(f"<b>{cost['total_month']:,.0f}</b><br/>月度总成本 (元)", st_cell),
            Paragraph(f"<b>{cost['unit_cost']:.3f}</b><br/>吨水综合成本 (元/吨)", st_cell),
            Paragraph(tn_txt, st_cell),
        ]]
        kt = Table(kpi, colWidths=[58 * mm] * 3)
        kt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, GRID),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, GRID),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        flow.append(kt)
        flow.append(Spacer(1, 6))

    # ---------- 一、基础参数 ----------
    param_map = {
        "Q_design": "设计日处理水量 (m³/d)", "Q_actual": "实际日均进水量 (m³/d)",
        "Kz": "总变化系数 Kz", "Q_max": "最大时流量 (m³/h)",
        "V_ana": "厌氧池有效容积 (m³)", "V_anox1": "第一缺氧池有效容积 (m³)",
        "V_aero1": "第一好氧池有效容积 (m³)", "V_anox2": "第二缺氧池有效容积 (m³)",
        "V_aero2": "第二好氧池有效容积 (m³)", "V_total": "生化池总容积 (m³)",
        "settler_area": "二沉池总表面积 (m²)", "settler_depth": "二沉池有效水深 (m)",
        "Y": "污泥产率系数 Y", "Kd": "内源衰减系数 Kd (d⁻¹)",
        "nitr_rate": "硝化速率 (kgNH3/(kgMLSS·d))", "denitr_rate": "反硝化速率 (kgNO3/(kgMLSS·d))",
        "mlvss_mlss": "MLVSS/MLSS 比值", "elec_price": "电价 (元/kWh)",
        "pac_price": "PAC单价 (元/吨)", "pfs_price": "PFS单价 (元/吨)",
        "naac_price": "乙酸钠单价 (元/吨)", "methanol_price": "甲醇单价 (元/吨)",
        "glucose_price": "葡萄糖单价 (元/吨)", "composite_carbon_price": "复合碳源单价 (元/吨)",
        "naclo_price": "次氯酸钠单价 (元/吨)", "pam_price": "PAM单价 (元/吨)",
        "hcl_price": "盐酸单价 (元/吨)", "sludge_dispose_price": "污泥处置单价 (元/吨湿泥)",
        "staff_num": "运维人员数量 (人)", "staff_salary": "人均月工资 (元)",
        "maintain_cost": "月度设备维修费 (元)", "other_cost": "月度其他杂费 (元)",
    }
    flow.append(Paragraph("一、水厂基础设计参数", st_h))
    rows = [[Paragraph("参数名称", st_cellw), Paragraph("参数值", st_cellw)]]
    for k, v in bp.items():
        rows.append([Paragraph(param_map.get(k, k), st_cell), Paragraph(fmt(v), st_cell)])
    t = Table(rows, colWidths=[120 * mm, 58 * mm], repeatRows=1)
    t.setStyle(_table_style())
    flow.append(t)
    flow.append(PageBreak())

    # ---------- 二、生化结果 ----------
    flow.append(Paragraph("二、生化系统计算结果", st_h))
    if bio:
        bio_rows = [
            ("理论出水总氮 TN", f"{bio['tn_theory']:.2f} mg/L", bio.get('tn_status', '')),
            ("达标所需最小内回流 R1", f"{max(bio['min_R1'], 100):.1f} %", ""),
            ("进水碳氮比 C/N", f"{bio['cn_ratio']:.2f}", bio.get('carbon_status', '')),
            ("碳源缺口", f"{bio['carbon_deficit']:.1f} mg/L (COD)", ""),
            (f"{bio['carbon_agent_name']}日投加量", f"{bio['carbon_daily']:.3f} 吨/天", ""),
            ("需化学除磷量（P）", f"{bio['tp_need_chem']:.2f} mg/L", ""),
            (f"{bio['phos_agent_name']}日投加量", f"{bio['phos_daily']:.3f} 吨/天", ""),
            ("污泥龄 SRT（硝化段）", f"{bio['srt']:.1f} d", "满足硝化菌世代要求" if bio['srt'] > 10 else "泥龄偏短，硝化菌易流失"),
            ("每日剩余干污泥量", f"{bio['sludge_dry_daily']:.2f} kg/d", ""),
            ("湿污泥量（含水率99.2%）", f"{bio['sludge_wet_daily']:.2f} m³/d", ""),
        ]
        r2 = [[Paragraph("指标", st_cellw), Paragraph("计算结果", st_cellw), Paragraph("说明 / 状态", st_cellw)]]
        for a, b, c in bio_rows:
            r2.append([Paragraph(a, st_cell), Paragraph(b, st_cell), Paragraph(c, st_cell)])
        t2 = Table(r2, colWidths=[60 * mm, 50 * mm, 68 * mm], repeatRows=1)
        t2.setStyle(_table_style())
        flow.append(t2)
    else:
        flow.append(Paragraph("暂无生化计算数据，请先完成「生化核心计算」页面。", st_body))
    flow.append(PageBreak())

    # ---------- 三、成本 + 饼图 ----------
    flow.append(Paragraph("三、月度运行成本核算", st_h))
    if cost:
        labels = ["电费", "药剂费", "污泥处置", "人员工资", "维修耗材", "其他"]
        vals = [cost['power_cost'], cost['med_cost'], cost['sludge_cost'],
                cost['staff_cost'], cost['maintain_cost'], cost['other_cost']]
        tt = sum(vals)
        palette = [colors.HexColor("#36a2eb"), colors.HexColor("#4bc0c0"),
                   colors.HexColor("#ff9f40"), colors.HexColor("#ff6384"),
                   colors.HexColor("#9966ff"), colors.HexColor("#c9cbcf")]
        d = Drawing(460, 200)
        pie = Pie()
        pie.x, pie.y, pie.width, pie.height = 25, 30, 140, 140
        pie.data = vals
        pie.labels = [f"{v / tt * 100:.1f}%" if v / tt * 100 >= 5 else "" for v in vals]
        pie.slices.fontName = cjk
        pie.slices.fontSize = 9
        pie.slices.strokeColor = colors.white
        pie.slices.strokeWidth = 1
        for i, col in enumerate(palette):
            pie.slices[i].fillColor = col
        d.add(pie)
        leg = Legend()
        leg.x, leg.y, leg.dx, leg.dy = 195, 175, 8, 9
        leg.fontName, leg.fontSize, leg.boxAnchor, leg.columnMaximum = cjk, 8.5, "nw", 8
        leg.colorNamePairs = [(palette[i], f"{labels[i]}  ¥{vals[i]:,.0f}  ({vals[i] / tt * 100:.1f}%)")
                              for i in range(len(labels))]
        d.add(leg)
        r3 = [[Paragraph("成本类别", st_cellw), Paragraph("月度费用 (元)", st_cellw), Paragraph("占比", st_cellw)]]
        for i in range(len(labels)):
            r3.append([Paragraph(labels[i], st_cell), Paragraph(f"{vals[i]:,.2f}", st_cell),
                       Paragraph(f"{vals[i] / tt * 100:.1f}%", st_cell)])
        r3.append([Paragraph("<b>合计</b>", st_cell), Paragraph(f"<b>{tt:,.2f}</b>", st_cell), Paragraph("100%", st_cell)])
        t3 = Table(r3, colWidths=[40 * mm, 52 * mm, 30 * mm], repeatRows=1)
        t3.setStyle(_table_style())
        flow.append(d)
        flow.append(Spacer(1, 4))
        flow.append(t3)
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(
            f"📌 月度运行总成本 <b>{cost['total_month']:,.2f}</b> 元，年度约 "
            f"<b>{cost['total_month'] * 12:,.2f}</b> 元，吨水处理综合成本 "
            f"<b>{cost['unit_cost']:.3f}</b> 元/吨。", st_body))
    else:
        flow.append(Paragraph("暂无成本核算数据，请先完成「成本核算」页面。", st_body))
    flow.append(PageBreak())

    # ---------- 四、结论与建议 ----------
    flow.append(Paragraph("四、关键结论与运行建议", st_h))
    bullets = []
    if bio:
        tgt = bio.get('tn_out_target', 15)
        if bio['tn_theory'] > tgt:
            bullets.append(f"总氮达标风险：理论出水 TN {bio['tn_theory']:.2f} mg/L 高于目标 {tgt:.1f} mg/L，"
                           f"建议加大内回流 R1 至 ≥ {max(bio['min_R1'], 100):.0f}%。")
        else:
            bullets.append(f"总氮控制良好：理论出水 TN {bio['tn_theory']:.2f} mg/L 满足目标 {tgt:.1f} mg/L。")
        if bio['carbon_deficit'] > 0:
            bullets.append(f"碳源不足：C/N={bio['cn_ratio']:.1f} < 4，存在碳源缺口 {bio['carbon_deficit']:.1f} mg/L，"
                           f"建议补充 {bio['carbon_agent_name']}（{bio['carbon_daily']:.3f} 吨/天）。")
        else:
            bullets.append(f"碳源充足：C/N={bio['cn_ratio']:.1f} ≥ 4，无需外加碳源。")
        if bio['srt'] <= 10:
            bullets.append(f"硝化风险：污泥龄 SRT={bio['srt']:.1f} d ≤ 10 d，硝化菌易流失，建议延长排泥周期。")
        else:
            bullets.append(f"硝化安全：污泥龄 SRT={bio['srt']:.1f} d，满足硝化菌世代时间要求。")
        if bio['tp_need_chem'] > 0:
            bullets.append(f"需化学除磷：化学除磷需求 {bio['tp_need_chem']:.2f} mg/L，"
                           f"建议投加 {bio['phos_agent_name']}（{bio['phos_daily']:.3f} 吨/天）。")
    else:
        bullets.append("暂无可分析的生化结果，请先完成「生化核心计算」。")
    bullets.append("本报告由 AI 辅助运维系统自动汇总，计算公式基于五段 Bardenpho 工艺经验模型，仅供参考，"
                   "实际运行请以现场监测为准。")
    for b in bullets:
        flow.append(Paragraph(f"• {b}", st_bullet))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(cjk, 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(16 * mm, 10 * mm, "五段 Bardenpho 污水厂运行报表 · AI 辅助运维系统生成")
        canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"第 {doc_.page} 页")
        canvas.restoreState()

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf


if st is not None:
    st.set_page_config(
        page_title="CoreMate 污水厂智慧运维平台",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 兼容旧会话：如果侧边栏 LLM 状态已缓存，先清除，避免修改 .env 后仍显示旧状态
    for _k in ["_llm_status"]:
        st.session_state.pop(_k, None)


    # ================= 全局视觉样式（高端统一主题） =================
    GLOBAL_CSS = r"""
    <style>
    :root{
      --bg1:#eef4f9; --bg2:#e1ebf3;
      --surface:#ffffff;
      --primary:#0e7490; --primary2:#0891b2; --accent:#14b8a6;
      --text:#0f172a; --text2:#475569; --muted:#94a3b8;
      --line:#e6edf3;
      --ok:#059669; --warn:#d97706; --err:#dc2626;
      --radius:14px;
      --shadow:0 6px 24px rgba(15,23,42,0.08);
    }
    html, body, .stApp{
      font-family:'Microsoft YaHei','PingFang SC','Hiragino Sans GB',-apple-system,'Segoe UI',Roboto,sans-serif;
    }
    .stApp{ background:linear-gradient(135deg,var(--bg1) 0%,var(--bg2) 100%) !important; }
    #MainMenu, footer{ display:none !important; }
    /* 隐藏 Streamlit 顶部横幅（Deploy/Running man），消除标题上方空白 */
    header[data-testid="stHeader"]{ display:none !important; }
    .stApp > header{ display:none !important; }
    /* 压缩主内容区顶部边距 */
    .main .block-container,
    .block-container{ max-width:100% !important; padding-left:1.5rem !important; padding-right:1.5rem !important; padding-top:0rem !important; padding-bottom:3rem; }
    .appview-container,
    [data-testid="stAppViewContainer"]{ padding-top:0rem !important; }
    .appview-container > .main,
    [data-testid="stAppViewContainer"] > .main{ padding-top:0rem !important; margin-top:0rem !important; }

    /* 标题体系 */
    h1{ font-size:1.7rem; font-weight:700; color:var(--text);
        border-left:5px solid var(--primary); padding-left:14px; margin-top:0; margin-bottom:.6rem; }
    h2{ font-size:1.28rem; font-weight:650; color:var(--text); margin-top:0; }
    h3{ font-size:1.05rem; font-weight:600; color:var(--text); margin-top:0; }
    .stCaption{ color:var(--text2) !important; }
    .stMarkdown p{ color:var(--text2); }

    /* 侧边栏：天蓝色渐变 + 深字 */
    [data-testid="stSidebar"]{
      background:linear-gradient(180deg,#E3F2FD 0%,#90CAF9 100%);
      border-right:1px solid rgba(13,71,161,0.18);
    }
    [data-testid="stSidebar"] *{ color:#0D2A4A; }
    [data-testid="stSidebar"] .css-1oe5cao, [data-testid="stSidebar"] h1{ color:#0B3D6E !important; }
    [data-testid="stSidebar"] h1{ border-left:none !important; padding-left:0 !important; margin-bottom:0.85rem !important; }
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] .stMarkdown p{ color:#1A4F86 !important; }
    [data-testid="stSidebar"] hr{ border-color:rgba(13,71,161,0.20) !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
      gap:6px;
    }
    [data-testid="stSidebar"] [role="radio"]{
      padding:10px 14px; border-radius:9px; transition:.15s;
    }
    [data-testid="stSidebar"] [aria-checked="true"]{
      background:rgba(13,71,161,0.12) !important;
      color:#0B3D6E !important; font-weight:700;
      box-shadow:inset 0 0 0 1px rgba(13,71,161,0.35);
    }

    /* 指标卡片 */
    [data-testid="stMetric"]{
      background:var(--surface); border:1px solid var(--line);
      border-radius:var(--radius); padding:16px 18px; box-shadow:var(--shadow);
    }
    [data-testid="stMetricLabel"]{ color:var(--text2) !important; font-size:.9rem; }
    [data-testid="stMetricValue"]{ color:var(--primary) !important; font-size:1.6rem; font-weight:700; }

    /* 数据表 */
    [data-testid="stDataFrame"]{
      border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow);
      border:1px solid var(--line);
    }
    [data-testid="stDataFrame"] table th{
      background:#0e7490 !important; color:#fff !important;
      font-weight:600; text-align:center;
    }
    [data-testid="stDataFrame"] table td{ text-align:center; }

    /* 选项卡 */
    [data-baseweb="tab"]{ font-weight:600; color:var(--text2); }
    [aria-selected="true"]{ color:var(--primary) !important; border-bottom:2px solid var(--primary) !important; }

    /* 按钮 */
    .stButton>button{
      border-radius:10px; border:none; font-weight:600;
      background:linear-gradient(135deg,var(--primary),var(--primary2)); color:#fff;
      box-shadow:0 2px 10px rgba(14,116,144,0.30); transition:.2s; padding:.5rem 1rem;
    }
    .stButton>button:hover{ transform:translateY(-1px); box-shadow:0 6px 18px rgba(14,116,144,0.35); }

    /* 提示框 */
    .stAlert{ border-radius:12px !important; box-shadow:var(--shadow); }
    </style>
    """
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ================= 密码登录校验（新增部分） =================
    # 初始化登录状态
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # 未登录时显示登录页

    if not st.session_state.logged_in:
        # ============== 登录页 v3：暖调实景背景 + 左信息区 + 右登录卡 ==============
        import base64 as _b64
        # 路径 C（v18-fix）：登录页背景图压到 ~170KB 的 JPEG；编码结果缓存进 session_state，
        # 避免每次 rerun 重复读盘 + base64 编码（即便有 fragment 刷新也不重复计算）。
        if "login_bg_uri" not in st.session_state:
            _bg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "登录页背景图_水务平台暖调版.jpg")
            try:
                with open(_bg_file, "rb") as _f:
                    st.session_state.login_bg_uri = "data:image/jpeg;base64," + _b64.b64encode(_f.read()).decode()
            except Exception:
                st.session_state.login_bg_uri = ""
        _bg_uri = st.session_state.login_bg_uri

        # 第①段：背景图 base64 + 左暗右亮遮罩 + 全部登录页 CSS（合并成一个独立 <style> 调用）
        # 关键修复：f-string 会把 CSS 里的 `{...}` 当作表达式去 evaluate，必须改用 % formatting（%s）。
        # Streamlit 对"内容只含 <style>"的 st.html 调用走 dompurifyConfig 路径，<style> 整块进 head 全局生效。
        _bg_value = ('url('+_bg_uri+')' if _bg_uri else 'linear-gradient(180deg,#071c33,#0e3a61)')

        # CSS 主体（含全部登录页样式；用一个 %s 占位符替代背景图 url）
        _css_full = r"""<style>
            .stApp{ background:__BG_PLACEHOLDER__ !important;
                background-size:cover !important; background-position:center center !important;
                background-repeat:no-repeat !important; background-attachment:fixed !important; }
            /* 左暗右亮渐变：让左侧白色文字清晰，右侧白卡不与背景冲突 */
            .stApp::before{ content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
                background:
                    linear-gradient(90deg, rgba(7,28,51,0.62) 0%, rgba(7,28,51,0.30) 45%, rgba(255,255,255,0.04) 100%),
                    radial-gradient(ellipse at top, transparent 35%, rgba(7,28,51,0.18) 100%); }
            /* 隐藏 Streamlit 默认头/工具/菜单/页脚，让背景图占满 */
            [data-testid="stHeader"]{ background:transparent !important; }
            [data-testid="stToolbar"]{ display:none !important; }
            #MainMenu{ visibility:hidden !important; }
            footer{ visibility:hidden !important; }
            /* 隐藏登录页左右两列之间的 Streamlit 列间距，让左右两个半屏真正紧贴 */
            [data-testid="stHorizontalBlock"]{ gap:0 !important; padding:0 !important; align-items:flex-start !important; }
            [data-testid="column"]{ padding:0 !important; }

            /* ===== 顶部品牌条 ===== */
            .top-brand{ position:fixed; top:24px; left:36px; z-index:5;
            display:flex; align-items:center; gap:12px;
            color:#fff; font-size:0.95rem; font-weight:600; letter-spacing:1.5px;
            text-shadow:0 2px 8px rgba(0,0,0,0.45); }
            .top-brand-mark{
            width:34px; height:34px; border-radius:8px;
            background:linear-gradient(135deg,#378ADD 0%,#185FA5 100%);
            display:flex; align-items:center; justify-content:center;
            box-shadow:0 6px 18px rgba(56,138,221,0.45);
            position:relative; overflow:hidden;
            }
            /* 纯 CSS 小水滴 v10：用 base64 内联 SVG 作 background-image，
            形状精确（顶部圆、底部尖、白色高光），不受 sub-pixel 影响。 */
            .tb-drop{
            width:16px; height:20px;
            background-image:url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAzMCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJnIiB4MT0iMCIgeTE9IjAiIHgyPSIwIiB5Mj0iMSI+PHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjRkZGRkZGIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjRTBFQUZBIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHBhdGggZD0iTTEyIDEuNSBDIDEyIDEuNSwgMiAxNCwgMiAxOS41IEMgMiAyNC43LCA2LjUgMjguNSwgMTIgMjguNSBDIDE3LjUgMjguNSwgMjIgMjQuNywgMjIgMTkuNSBDIDIyIDE0LCAxMiAxLjUsIDEyIDEuNSBaIiBmaWxsPSJ1cmwoI2cpIi8+PC9zdmc+");
            background-repeat:no-repeat;
            background-size:contain;
            background-position:center;
            filter:drop-shadow(0 1px 2px rgba(15,30,60,0.25));
            animation:tbDropPulse 2.6s ease-in-out infinite;
            }
            @keyframes tbDropPulse{
            0%,100%{ transform:scale(1); }
            50%{ transform:scale(1.12); }
            }
            .top-version{
            position:fixed; top:28px; right:36px; z-index:5;
            color:rgba(255,255,255,0.92); font-size:0.78rem; letter-spacing:1.2px;
            background:rgba(15,23,42,0.35);
            border:1px solid rgba(255,255,255,0.28);
            padding:6px 14px; border-radius:999px;
            backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
            text-shadow:0 2px 6px rgba(0,0,0,0.45);
            }

            /* ===== 左半屏：品牌信息 ===== */
            .left-info{ position:relative; z-index:2;
            max-width:620px; margin:0;
            color:#fff; padding:100px 36px 60px; }
            .left-eyebrow{ display:inline-block; padding:6px 14px; border-radius:999px;
            font-size:0.78rem; letter-spacing:2px; font-weight:600;
            background:rgba(255,255,255,0.16);
            border:1px solid rgba(255,255,255,0.35);
            backdrop-filter:blur(8px);
            color:#FFE0B8; margin-bottom:22px; }
            .left-title{ font-size:2.6rem; font-weight:900; line-height:1.1;
            letter-spacing:1.5px; margin:0 0 8px;
            text-shadow:0 4px 18px rgba(0,0,0,0.5);
            white-space:nowrap; display:flex; align-items:baseline; flex-wrap:nowrap; }
            .left-title .core{
            color:#ffffff !important; -webkit-text-fill-color:#ffffff !important;
            background:none !important; background-clip:initial !important;
            font-weight:900; display:inline-block; letter-spacing:1px;
            }
            .left-title .accent{
            background:linear-gradient(90deg,#FFB36B,#FFD58A,#FFCB88);
            -webkit-background-clip:text; background-clip:text; color:transparent;
            -webkit-text-fill-color:transparent;
            margin-left:14px; display:inline-block; font-weight:900;
            }
            .left-sub{ margin:14px 0 0; font-size:0.95rem; font-weight:500;
            color:rgba(255,255,255,0.72);
            text-shadow:0 2px 10px rgba(0,0,0,0.45);
            line-height:1.35; }
            .left-en{ margin-top:6px; font-size:0.72rem; letter-spacing:4px;
            color:rgba(255,231,200,0.72); font-weight:500;
            text-shadow:0 2px 10px rgba(0,0,0,0.45); }
            .left-divider{ width:60px; height:4px; border-radius:3px; margin:30px 0 22px;
            background:linear-gradient(90deg,#FFB36B,#FFD58A);
            box-shadow:0 2px 10px rgba(255,179,107,0.55); }
            .left-features{ list-style:none; padding:0; margin:0; }
            .left-features li{ display:flex; align-items:center; gap:14px;
            padding:10px 0;
            color:rgba(255,255,255,0.96); font-size:1.02rem; font-weight:500;
            text-shadow:0 2px 8px rgba(0,0,0,0.45); }
            .left-features .check{
            width:22px; height:22px; border-radius:50%;
            background:linear-gradient(135deg,#FFB36B,#FFD58A);
            display:flex; align-items:center; justify-content:center;
            flex-shrink:0;
            color:#fff; font-size:14px; font-weight:900; line-height:1;
            box-shadow:0 4px 12px rgba(255,179,107,0.45); }
            .left-features .check svg{ width:12px; height:12px; }
            .left-tagline{ margin-top:34px; padding:14px 18px;
            background:rgba(255,255,255,0.10);
            border:1px solid rgba(255,255,255,0.28);
            border-radius:12px;
            color:rgba(255,255,255,0.95); font-size:0.92rem; line-height:1.6;
            backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }
            .left-tagline strong{ color:#FFD58A; font-weight:700; }
            .left-tagline .hot{ color:#FFD58A; font-weight:700; }
            .left-tagline .bolt{ display:inline-block; margin-right:6px;
            color:#FFB36B; font-weight:800; transform:translateY(-1px); }

            /* ===== 右半屏：登录卡 ===== */
            .right-login-wrap{ position:relative; z-index:2;
            width:320px; max-width:320px; margin:70px auto 0; padding:0; }
            .login-card{
            position:relative; z-index:2; width:320px;
            background:rgba(255,255,255,0.96); backdrop-filter:blur(20px);
            -webkit-backdrop-filter:blur(20px);
            border:1px solid rgba(255,255,255,0.7); border-radius:22px;
            padding:30px 24px 22px;
            box-sizing:border-box;
            box-shadow:0 30px 80px rgba(2,12,27,0.5), 0 0 40px rgba(255,255,255,0.18) inset;
            text-align:center;
            }
            .login-logo{ position:relative; width:72px; height:72px; border-radius:50%; margin:0 auto 16px;
            display:flex; align-items:center; justify-content:center;
            background:linear-gradient(135deg,#378ADD,#185FA5);
            box-shadow:0 12px 30px rgba(24,95,165,0.45), 0 0 0 4px rgba(255,255,255,0.7); }
            /* 水波纹：v9 同心圆扩散，初始 size=22px 与水滴同大，向外淡出，
            颜色与水滴（白）保持同源，强化"水从水滴外溢"的语义。 */
            .login-ripple{ position:absolute; left:50%; top:50%;
            width:22px; height:22px; margin:-11px 0 0 -11px;
            border:1.5px solid rgba(255,255,255,0.9); border-radius:50%;
            opacity:0; transform:scale(0.55);
            animation:loginRipple 2.6s ease-out infinite; }
            .login-ripple.r2{ animation-delay:0.9s; }
            .login-ripple.r3{ animation-delay:1.7s; }
            @keyframes loginRipple{
            0%{ transform:scale(0.55); opacity:0.85; }
            100%{ transform:scale(2.4); opacity:0; }
            }
            /* 纯 CSS 白水滴 v10：与 .tb-drop 同一 SVG 资源，仅尺寸更大、阴影更明显 */
            .login-drop{ position:relative; width:28px; height:34px;
            background-image:url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAzMCI+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJnIiB4MT0iMCIgeTE9IjAiIHgyPSIwIiB5Mj0iMSI+PHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjRkZGRkZGIi8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjRDhFNUY4Ii8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHBhdGggZD0iTTEyIDEuNSBDIDEyIDEuNSwgMiAxNCwgMiAxOS41IEMgMiAyNC43LCA2LjUgMjguNSwgMTIgMjguNSBDIDE3LjUgMjguNSwgMjIgMjQuNywgMjIgMTkuNSBDIDIyIDE0LCAxMiAxLjUsIDEyIDEuNSBaIiBmaWxsPSJ1cmwoI2cpIi8+PC9zdmc+");
            background-repeat:no-repeat;
            background-size:contain;
            background-position:center;
            filter:drop-shadow(0 2px 3px rgba(15,30,60,0.30));
            animation:loginDropBreath 3.0s ease-in-out infinite; }
            @keyframes loginDropBreath{
            0%,100%{ transform:scale(1); }
            50%{ transform:scale(1.08); }
            }

            .login-title{ font-size:1.65rem; font-weight:800; color:#0f172a;
            letter-spacing:1px; margin:0; }
            .login-sub{ margin-top:6px; font-size:0.78rem; font-weight:600;
            color:#185FA5; letter-spacing:1.5px; }
            .login-en{ margin-top:4px; font-size:0.62rem; letter-spacing:3px;
            color:#64748b; text-transform:uppercase; }
            .login-divider-card{ width:50px; height:2.5px; border-radius:2px;
            margin:14px auto 4px;
            background:linear-gradient(90deg,#185FA5,#534AB7); }

            /* ===== 表单融入登录卡 ===== */
            /* 说明：st.html(.right-login-wrap) 与 st.form() 是兄弟节点（并列在列内），
            因此不能用 .right-login-wrap 作后代锚点。改用全局 data-testid 选择器。 */
            [data-testid="stForm"]{
            background:transparent !important; border:none !important;
            padding:14px 0 0 !important; margin:0 auto !important;
            width:320px !important; max-width:320px !important; min-width:320px !important;
            box-sizing:border-box !important; }
            [data-testid="stTextInput"]{
            width:320px !important; max-width:320px !important; min-width:320px !important;
            margin:0 auto 14px !important; display:block !important;
            box-sizing:border-box !important; }
            [data-testid="stTextInput"] > label{ display:none !important; }
            [data-testid="stTextInput"] input,
            input[type="password"]{
            background:#F4F7FA !important; color:#0f172a !important;
            border-radius:12px !important;
            border:1.5px solid rgba(24,95,165,0.35) !important;
            padding:13px 16px !important; font-size:0.98rem !important;
            transition:border-color .15s, box-shadow .15s, background .15s !important;
            width:100% !important; max-width:100% !important; min-width:100% !important;
            box-shadow:none !important; box-sizing:border-box !important; }
            [data-testid="stTextInput"] input::placeholder,
            input[type="password"]::placeholder{
            color:#94a3b8 !important; }
            [data-testid="stTextInput"] input:focus,
            input[type="password"]:focus{
            background:#fff !important; border-color:#185FA5 !important;
            box-shadow:0 0 0 3px rgba(24,95,165,0.18) !important;
            outline:none !important; }

            /* 登录按钮：蓝渐变，覆盖主题 primary 红 */
            [data-testid="stFormSubmitButton"] button,
            button[kind="primaryFormSubmit"]{
            background:linear-gradient(135deg,#185FA5 0%,#378ADD 100%) !important;
            color:#fff !important; border:none !important; border-radius:12px !important;
            font-weight:800 !important; letter-spacing:4px !important;
            font-size:1.02rem !important;
            box-shadow:0 10px 24px rgba(24,95,165,0.45) !important;
            transition:transform .15s ease, box-shadow .15s ease !important;
            padding:14px 18px !important;
            width:100% !important; min-width:320px !important; max-width:320px !important;
            margin:6px auto 0 !important; display:block !important;
            box-sizing:border-box !important; }
            [data-testid="stFormSubmitButton"] button:hover,
            button[kind="primaryFormSubmit"]:hover{
            transform:translateY(-2px) !important;
            box-shadow:0 14px 30px rgba(24,95,165,0.6) !important; }

            /* 底部链接：忘记密码/没有账号 */
            .login-bottom-links{ margin-top:18px; display:flex;
            justify-content:space-between; font-size:0.82rem; color:#64748b; }
            .login-bottom-links a{ color:#185FA5; text-decoration:none; font-weight:600; transition:color .15s; }
            .login-bottom-links a:hover{ color:#378ADD; }

            /* ===== 底部品牌条 ===== */
            .bottom-brand{
            position:fixed; left:0; bottom:0; z-index:5; width:100%;
            padding:18px 36px;
            display:flex; justify-content:space-between; align-items:center;
            color:rgba(255,255,255,0.78); font-size:0.78rem; letter-spacing:1.2px;
            background:linear-gradient(180deg, transparent, rgba(7,28,51,0.55));
            text-shadow:0 2px 8px rgba(0,0,0,0.45); pointer-events:none; }
            .bottom-brand .copy{ display:flex; align-items:center; gap:8px; }
            .bottom-brand .links{ display:flex; gap:18px; }
            .bottom-brand .links a{
            color:rgba(255,255,255,0.78); text-decoration:none; pointer-events:auto; }
            .bottom-brand .links a:hover{ color:#FFD58A; }

            /* 响应式：≤900 改为上下堆叠 */
            @media (max-width: 900px){
            .top-brand{ left:18px; top:14px; }
            .top-version{ right:18px; top:18px; font-size:0.7rem; }
            .left-info{ padding:60px 24px 24px; }
            .left-title{ font-size:2.4rem; }
            .left-sub{ font-size:1rem; }
            .right-login-wrap{ margin-top:20px; }
            .bottom-brand{ padding:14px 18px; flex-direction:column; gap:6px; }
            }
            </style>"""
        # 关键修复：st.html() 渲染的 <style> 被 Streamlit 作用域隔离/净化，无法全局生效；
        # 全局 CSS 必须用 st.markdown(..., unsafe_allow_html=True) 注入（已实证可全局生效）。
        st.markdown(_css_full.replace("__BG_PLACEHOLDER__", _bg_value), unsafe_allow_html=True)

        # 第②.⑤段：顶部品牌条 + 底部版权条 — 用 st.html() 独立渲染（仅结构，不含 <style>）
        st.html(r"""
        <div class="top-brand">
            <div class="top-brand-mark">
                <div class="tb-drop"></div>
            </div>
            <span>CoreMate · 智慧水务</span>
        </div>
        <div class="top-version">v1.2.0 · 2026 · 山东招金膜天</div>

        <div class="bottom-brand">
            <div class="copy">© 2026 CoreMate · AI-Driven WWTP Intelligent O&amp;M Platform</div>
            <div class="links">
                <a href="javascript:void(0)">用户手册</a>
                <a href="javascript:void(0)">工艺支持</a>
                <a href="javascript:void(0)">联系我们</a>
            </div>
        </div>
        """)
        col_left, col_right = st.columns([1.05, 1])

        with col_left:
            st.html(r"""
            <div class="left-info">
                <div class="left-eyebrow">智慧水务 · AI 驱动 · 数字孪生</div>
                <h1 class="left-title"><span class="core">CoreMate</span><span class="accent">智慧水务</span></h1>
                <p class="left-sub">基于五段 Bardenpho 工艺 + AI 大模型的</p>
                <p class="left-sub">污水处理厂全场景智慧运维平台</p>
                <div class="left-en">AI-DRIVEN WWTP INTELLIGENT O&amp;M PLATFORM</div>

                <div class="left-divider"></div>

                <ul class="left-features">
                    <li>
                        <span class="check">✓</span>
                        五段 Bardenpho 工艺内嵌，全工艺流程可视化建模
                    </li>
                    <li>
                        <span class="check">✓</span>
                        AI 预测出水水质、加药量、能耗等关键运行参数
                    </li>
                    <li>
                        <span class="check">✓</span>
                        数字孪生驱动降碳增效，动态求解运行成本最优解
                    </li>
                    <li>
                        <span class="check">✓</span>
                        自然语言交互，毫秒级响应工艺咨询与运行决策
                    </li>
                </ul>

                <div class="left-tagline">
                    <span class="bolt">▸</span> 从 <span class="hot">机理模型</span> 到 <span class="hot">AI 模型</span> 到 <span class="hot">运维界面</span> <span class="hot">全栈自研</span>
                </div>
            </div>
            """)

        with col_right:
            st.html(r"""
            <div class="right-login-wrap">
                <div class="login-card">
                    <div class="login-logo">
                        <div class="login-ripple r1"></div>
                        <div class="login-ripple r2"></div>
                        <div class="login-ripple r3"></div>
                        <div class="login-drop"></div>
                    </div>
                    <h2 class="login-title">欢迎登录</h2>
                    <div class="login-sub">CORE&nbsp;MATE&nbsp;&middot;&nbsp;智慧水务</div>
                    <div class="login-en">Welcome to CoreMate</div>
                    <div class="login-divider-card"></div>
                </div>
                <div class="login-bottom-links">
                    <a href="javascript:void(0)">忘记密码？</a>
                    <a href="javascript:void(0)">注册新账号</a>
                </div>
            </div>
            """)

            with st.form("login_form_v3", clear_on_submit=False):
                input_pwd = st.text_input("访问密码", type="password",
                                          placeholder="🔒  请输入访问密码",
                                          help="默认密码：123456", label_visibility="collapsed")
                submitted = st.form_submit_button("登 录 系 统", type="primary",
                                                  use_container_width=True)
                if submitted:
                    try:
                        correct_pwd = st.secrets["access_password"]
                    except Exception:
                        correct_pwd = "123456"
                    if input_pwd == correct_pwd:
                        st.session_state.logged_in = True
                        st.success("登录成功，正在进入系统...")
                        st.rerun()
                    else:
                        st.error("密码错误，请重试")
        st.stop()  # 密码验证不通过，停止执行后面所有代码


    # matplotlib中文显示：设置跨平台中文字体回退栈，覆盖 Windows/macOS/Linux 及常见服务器环境
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "SimSun", "DengXian",
        "PingFang SC", "Hiragino Sans GB", "STHeiti",
        "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans CN",
        "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # ---- 侧边栏保底：若被浏览器 localStorage 记录为"收起"（如微信内置浏览器收起后刷新无法恢复），
    # 加载时自动展开并改写 localStorage，确保导航栏始终可见。无可见 UI，纯兜底纠正，不改变默认交互。
    # 注意：st.markdown 注入的 <script> 会被 Streamlit 净化剥离，故改用 components.html（iframe 内脚本可
    # 执行且同源可访问父页面 DOM）来点击展开箭头；height=0 使其不可见。 ----
    # 侧边栏保底脚本只需在会话首次进入时纠正一次（改写 localStorage 后即持久），
    # 用 session_state 守卫避免每次重跑/每次刷新都新建 iframe（线上每次交互都走网络，能省则省）。
    if _st_html is not None and not st.session_state.get("_sb_fixed_done", False):
        _st_html(
            "<script>(function(){"
            "function te(){"
            "  var s=window.parent.document.querySelector('[data-testid=\"stSidebar\"]');"
            "  if(!s)return false;"
            "  if(s.getBoundingClientRect().width>50)return true;"
            "  var b=Array.prototype.slice.call(window.parent.document.querySelectorAll('button'));"
            "  var r=b.find(function(x){return (x.textContent||'')==='keyboard_double_arrow_right';});"
            "  if(r){r.click();return true;}return false;"
            "}"
            "var n=0,iv=setInterval(function(){try{if(te())clearInterval(iv);}catch(e){}"
            "if(++n>15)clearInterval(iv);},300);"
            "})();</script>",
            height=0, scrolling=False)
        st.session_state["_sb_fixed_done"] = True

    # 计算结果 schema 版本号：用于识别 session_state 中残留的旧版计算结果。
    # 代码更新后，旧 dict 可能缺少新字段（如 bio['R']），据此安全提示重算而非崩溃。
    RESULT_SCHEMA = "2026-08-08"


    def _judge_hrt(value, low, high, tol=0.20):
        """HRT 双边界判定：严格在 [low, high] 内为合理；在 ±tol 容差内为临界；否则偏离。"""
        if low <= value <= high:
            return "✅ 合理"
        eff_low, eff_high = low * (1 - tol), high * (1 + tol)
        if eff_low <= value <= eff_high:
            return "⚠️ 临界"
        return "⚠️ 偏离"


    def _judge_hrt_min(value, low, tol=0.10):
        """HRT 单下界判定：≥low 为满足；在 -tol 容差内为临界；否则偏短。"""
        if value >= low:
            return "✅ 满足"
        if value >= low * (1 - tol):
            return "⚠️ 临界"
        return "⚠️ 偏短"


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
            'sludge_dispose_price': 220,  # 污泥处置单价 元/吨湿泥
            'staff_num': 12,        # 运维人数
            'staff_salary': 6800,   # 人均月薪 元
            'maintain_cost': 36000, # 月度维修费 元
            'other_cost': 22000      # 其他杂费 元/月
        }

    if 'bio_result' not in st.session_state:
        st.session_state.bio_result = {}

    # ============================================================
    # AI 能力模块（预测 / 优化 / 诊断 / 认知）—— 感知·预测·优化·决策·认知 闭环
    # 说明：核心算法与 AI 内核已内联进本文件（单文件自包含，无需 wwtp_core.py）；
    #      LLM 助手在无 API Key 时自动降级为规则引擎，保证演示永不中断。
    # ============================================================
    SAMPLE_CSV = os.path.join(SCRIPT_DIR, "sample_wwtp_history.csv")


    # 跨页数据契约访问器（item 7）：统一校验 RESULT_SCHEMA，避免散落的 .get('schema') 判断
    def get_compute_result(name):
        """返回 schema 匹配的 result dict，否则 None。集中 RESULT_SCHEMA 校验。"""
        r = st.session_state.get(name)
        if isinstance(r, dict) and r.get("schema") == RESULT_SCHEMA:
            return r
        return None


    def num_input(label, value=None, *args, **kwargs):
        """包装 st.number_input：自动对齐 min_value / max_value 与 value 的类型，
        避免 StreamlitMixedNumericTypesError（value 为 int 而 min_value 为 float 等）。"""
        numeric_keys = ("min_value", "max_value", "step")
        if value is not None:
            for key in numeric_keys:
                if key in kwargs and kwargs[key] is not None:
                    cur = kwargs[key]
                    if isinstance(value, int) and not isinstance(cur, int):
                        kwargs[key] = int(cur)
                    elif isinstance(value, float) and not isinstance(cur, float):
                        kwargs[key] = float(cur)
        return st.number_input(label, value=value, *args, **kwargs)


    # ================= 侧边栏导航 =================
    with st.sidebar:
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        if not os.path.exists(logo_path):
            # 兼容从父目录运行的情况
            logo_path = os.path.join(os.path.dirname(__file__), "污水厂管理系统", "assets", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=260)
        st.title("🏭 系统导航")
        st.caption("五段Bardenpho工艺污水厂运维管理系统")
        st.markdown("---")
        page = st.radio(
            "功能模块",
            [
                "🛰️ 数字孪生驾驶舱",
                "📝 基础参数设置",
                "💧 水力与负荷校核",
                "🧪 生化核心计算",
                "🏞️ 二沉池专项校核",
                "💠 浸没式超滤工况",
                "💰 成本经济核算",
                "🔮 AI 预测预警",
                "💬 AI 工艺助手",
                "📊 报表导出"
            ]
        )
        st.markdown("---")
        if "_llm_status" not in st.session_state:
            st.session_state._llm_status = check_llm_config()
        _mode, _detail, _ok = st.session_state._llm_status
        _icon = {"offline": "🟢", "local": "🔵", "cloud": "🟣"}.get(_mode, "⚪")
        st.markdown(f"{_icon} **AI 模式**：{_detail}")


    # ================= 页面1：基础参数设置 =================
    if page == "📝 基础参数设置":
        st.header("📝 基础信息参数设置")
        st.caption("一次性录入水厂设计参数、动力学系数、经济单价，全局所有模块自动调用")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("一、水量与池体参数")
            Q_design = num_input("设计日处理水量 (m³/d)", value=st.session_state.base_params['Q_design'], min_value=0.0)
            Q_actual = num_input("实际日均进水量 (m³/d)", value=st.session_state.base_params['Q_actual'], min_value=0.0)
            Kz = num_input("总变化系数 Kz", value=st.session_state.base_params['Kz'], min_value=0.0)
            Q_max = num_input("最大时流量 (m³/h)", value=st.session_state.base_params['Q_max'], min_value=0.0)

            st.markdown("#### 各池有效容积 (m³)")
            V_ana = num_input("厌氧池", value=st.session_state.base_params['V_ana'], min_value=0.0)
            V_anox1 = num_input("第一缺氧池", value=st.session_state.base_params['V_anox1'], min_value=0.0)
            V_aero1 = num_input("第一好氧池", value=st.session_state.base_params['V_aero1'], min_value=0.0)
            V_anox2 = num_input("第二缺氧池", value=st.session_state.base_params['V_anox2'], min_value=0.0)
            V_aero2 = num_input("第二好氧池", value=st.session_state.base_params['V_aero2'], min_value=0.0)
            V_total = num_input("生化池总容积", value=st.session_state.base_params['V_total'], min_value=0.0)
            settler_area = num_input("二沉池总表面积 (m²)", value=st.session_state.base_params['settler_area'], min_value=0.0)
            settler_depth = num_input("二沉池有效水深 (m)", value=st.session_state.base_params['settler_depth'], min_value=0.0)

            st.subheader("二、生化动力学系数")
            Y = num_input("污泥产率系数 Y", value=st.session_state.base_params['Y'], min_value=0.0)
            Kd = num_input("内源衰减系数 Kd (d⁻¹)", value=st.session_state.base_params['Kd'], min_value=0.0)
            nitr_rate = num_input("硝化速率 kgNH3/(kgMLSS·d)", value=st.session_state.base_params['nitr_rate'], min_value=0.0)
            denitr_rate = num_input("反硝化速率 kgNO3/(kgMLSS·d)", value=st.session_state.base_params['denitr_rate'], min_value=0.0)
            mlvss_mlss = num_input("MLVSS / MLSS 比值", value=st.session_state.base_params['mlvss_mlss'], min_value=0.0)
            carbon_cod_eq = num_input("碳源COD当量基准值 (gCOD/g药剂)",
                                            value=st.session_state.base_params['carbon_cod_eq'], min_value=0.0)

        with col2:
            st.subheader("三、经济成本参数")
            elec_price = num_input("电价 (元/kWh)", value=st.session_state.base_params['elec_price'], min_value=0.0)
            # 除磷药剂双价格
            pac_price = num_input("PAC铝盐单价 (元/吨)", value=st.session_state.base_params['pac_price'], min_value=0.0)
            pfs_price = num_input("PFS铁盐单价 (元/吨)", value=st.session_state.base_params['pfs_price'], min_value=0.0)
            # 四类碳源单价
            naac_price = num_input("乙酸钠碳源单价 (元/吨)", value=st.session_state.base_params['naac_price'], min_value=0.0)
            methanol_price = num_input("甲醇碳源单价 (元/吨)", value=st.session_state.base_params['methanol_price'], min_value=0.0)
            glucose_price = num_input("葡萄糖碳源单价 (元/吨)", value=st.session_state.base_params['glucose_price'], min_value=0.0)
            composite_carbon_price = num_input("复合碳源单价 (元/吨)",
                                                     value=st.session_state.base_params['composite_carbon_price'], min_value=0.0)
            # 其他药剂
            naclo_price = num_input("次氯酸钠单价 (元/吨)", value=st.session_state.base_params['naclo_price'], min_value=0.0)
            pam_price = num_input("PAM絮凝剂单价 (元/吨)", value=st.session_state.base_params['pam_price'], min_value=0.0)
            hcl_price = num_input("盐酸单价 (元/吨，pH调节)", value=st.session_state.base_params['hcl_price'], min_value=0.0)
            # 污泥&人工运维
            sludge_dispose_price = num_input("污泥处置单价 (元/吨湿泥)",
                                                   value=st.session_state.base_params['sludge_dispose_price'], min_value=0.0)
            staff_num = num_input("运维人员数量 (人)", value=st.session_state.base_params['staff_num'], min_value=0.0)
            staff_salary = num_input("人均月工资 (元)", value=st.session_state.base_params['staff_salary'], min_value=0.0)
            maintain_cost = num_input("月度设备维修费 (元)", value=st.session_state.base_params['maintain_cost'], min_value=0.0)
            other_cost = num_input("月度其他杂费 (元)", value=st.session_state.base_params['other_cost'], min_value=0.0)

        if st.button("💾 保存全部基础参数", type="primary", use_container_width=True):
            # item 8：保存前校验——先组装待保存参数，校验通过才写入，避免脏数据导致后续计算除零/NaN
            candidate = {
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
                'maintain_cost': maintain_cost, 'other_cost': other_cost,
            }
            errors, warnings = validate_base_params(candidate)
            for w in warnings:
                st.warning("⚠️ " + w)
            if errors:
                for e in errors:
                    st.error("⛔ " + e)
                st.error("存在致命参数错误，已取消保存，请修正后重试。")
            else:
                st.session_state.base_params.update(candidate)
                st.success("✅ 所有基础参数已保存，全部计算模块将自动调用")


    # ================= 页面2：水力与负荷校核 =================
    elif page == "💧 水力与负荷校核":
        st.header("💧 水力停留时间与负荷校核")
        bp = st.session_state.base_params

        st.subheader("一、进水水质与运行参数")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1: cod_in = num_input("进水COD (mg/L)", value=350)
        with col2: bod_in = num_input("进水BOD5 (mg/L)", value=180)
        with col3: tn_in = num_input("进水总氮 TN (mg/L)", value=40)
        with col4: nh3_in = num_input("进水氨氮 NH3-N (mg/L)", value=28)
        with col5: tp_in = num_input("进水总磷 TP (mg/L)", value=5)
        with col6: mlss = num_input("MLSS (mg/L)", value=3500)

        if st.button("开始校核计算", type="primary"):
            Q = bp['Q_actual']
            V_aero_total = bp['V_aero1'] + bp['V_aero2']   # 曝气池总容积 = 好氧1 + 好氧2

            # ========== 1. 水力停留时间HRT ==========
            hrt_total = bp['V_total'] / Q * 24
            hrt_ana = bp['V_ana'] / Q * 24
            hrt_anox1 = bp['V_anox1'] / Q * 24
            hrt_aero1 = bp['V_aero1'] / Q * 24
            hrt_anox2 = bp['V_anox2'] / Q * 24
            hrt_aero2 = bp['V_aero2'] / Q * 24
            hrt_aero_total = V_aero_total / Q * 24

            # HRT判定（采用±20%容差：严格在范围内为合理，边界附近为临界，否则为偏离）
            hrt_total_status = _judge_hrt_min(hrt_total, 12, tol=0.10)
            hrt_ana_status = _judge_hrt(hrt_ana, 0.5, 2, tol=0.20)
            hrt_anox1_status = _judge_hrt(hrt_anox1, 2, 4, tol=0.20)
            hrt_aero1_status = _judge_hrt(hrt_aero1, 4, 12, tol=0.20)
            hrt_anox2_status = _judge_hrt(hrt_anox2, 2, 4, tol=0.20)
            hrt_aero2_status = _judge_hrt(hrt_aero2, 0.5, 2, tol=0.20)


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

            # ========== 结果存储（持久化，供切换页面后展示） ==========
            conclusion = f"""
            当前工况下，五段Bardenpho(或Phoredox)系统水力停留时间{hrt_total_status.replace('✅ ','').replace('⚠️ ','')}脱氮除磷要求；
            污泥负荷{ns_status.replace('✅ ','').replace('⚠️ ','')}；
            若进水浓度进一步升高，可通过提高MLSS、调控溶解氧、加大内回流比、调整外回流比、补充外加碳源、投加除磷剂等措施保障出水达标。
            """
            st.session_state.hydro_result = {
                'schema': RESULT_SCHEMA,
                'hrt_total': hrt_total, 'hrt_ana': hrt_ana, 'hrt_anox1': hrt_anox1,
                'hrt_aero1': hrt_aero1, 'hrt_anox2': hrt_anox2, 'hrt_aero2': hrt_aero2,
                'hrt_aero_total': hrt_aero_total,
                'hrt_total_status': hrt_total_status, 'hrt_ana_status': hrt_ana_status,
                'hrt_anox1_status': hrt_anox1_status, 'hrt_aero1_status': hrt_aero1_status,
                'hrt_anox2_status': hrt_anox2_status, 'hrt_aero2_status': hrt_aero2_status,
                'ns_bod': ns_bod, 'ns_cod': ns_cod, 'ns_status': ns_status,
                'conclusion': conclusion
            }



        # ===== 持久展示：切换页面后仍保留计算结果 =====
        _hydro_stale = bool(st.session_state.get('hydro_result')) and st.session_state.hydro_result.get('schema') != RESULT_SCHEMA
        if _hydro_stale:
            st.info("计算结果格式已更新，请重新点击本页「开始校核计算」按钮以刷新结果。")
        if st.session_state.get('hydro_result') and st.session_state.get('hydro_result').get('schema') == RESULT_SCHEMA:
            res = st.session_state.hydro_result
            st.markdown("---")
            tab1, tab2 = st.tabs(["水力停留时间(HRT)", "污泥负荷校核"])
            with tab1:
                st.subheader("1. 各功能区水力停留时间")
                hrt_df = pd.DataFrame({
                    "功能区": ["厌氧池", "第一缺氧池", "第一好氧池", "第二缺氧池", "第二好氧池", "曝气池合计", "生化池总HRT"],
                    "停留时间 (h)": [res['hrt_ana'], res['hrt_anox1'], res['hrt_aero1'], res['hrt_anox2'], res['hrt_aero2'], res['hrt_aero_total'], res['hrt_total']],
                    "推荐范围 (h)": ["0.5~2", "2~4", "4~12", "2~4", "0.5~2", "4.5~14", "≥12"],
                    "判定": [res['hrt_ana_status'], res['hrt_anox1_status'], res['hrt_aero1_status'], res['hrt_anox2_status'], res['hrt_aero2_status'], "—", res['hrt_total_status']]
                })
                st.dataframe(hrt_df, use_container_width=True, hide_index=True)
                st.info("第二好氧池仅用于吹脱氮气与维持DO，不承担硝化功能，停留时间按短HRT设计")

            with tab2:
                st.subheader("2. 系统污泥负荷校核")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("BOD污泥负荷 Ns", f"{res['ns_bod']:.4f} kgBOD/(kgMLSS·d)")
                    st.info(res['ns_status'])
                    st.caption("脱氮除磷工艺推荐污泥负荷：0.05~0.15 kgBOD/(kgMLSS·d)")
                with col2:
                    st.metric("COD污泥负荷", f"{res['ns_cod']:.4f} kgCOD/(kgMLSS·d)")

                st.markdown("---")
                st.write("**污泥负荷调控建议：**")
                st.write("- 负荷过高：加大排泥、提高MLSS、降低进水负荷")
                st.write("- 负荷过低：减少排泥、降低MLSS、缩短污泥龄，防止污泥老化解体")

            st.markdown("---")
            st.subheader("二、综合校核结论")
            st.write(res['conclusion'])

    # ================= 页面3：生化核心计算 =================
    elif page == "🧪 生化核心计算":
        st.header("🧪 五段Bardenpho生化系统核心计算")
        bp = st.session_state.base_params

        st.subheader("一、运行参数输入")
        col1, col2, col3 = st.columns(3)
        with col1:
            cod_in = num_input("进水COD (mg/L)", value=210)
            bod_in = num_input("进水BOD5 (mg/L)", value=130)
            nh3_in = num_input("进水氨氮 (mg/L)", value=28)
            tn_in = num_input("进水总氮 TN (mg/L)", value=57)
            tp_in = num_input("进水总磷 TP (mg/L)", value=5)
            R1 = num_input("内回流比 R1 (好氧1→缺氧1, %)", value=200) / 100
            waste_sludge_volume = num_input("每日外排剩余污泥量(m³/d)", value=20.0)
        with col2:
            tn_out_target = num_input("出水TN目标 (mg/L)", value=15)
            tp_out_target = num_input("出水TP目标 (mg/L)", value=0.5)
            bod_eff = num_input("实际出水BOD5 (mg/L)", value=10)
            cod_eff = num_input("实际出水COD (mg/L)", value=50)
            nh3_eff = num_input("实际出水氨氮 (mg/L)", value=1.5)
            mlss = num_input("MLSS 混合液浓度 (mg/L)", value=3500)
            R = num_input("污泥回流比 R (%)", value=100) / 100

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
            total_return = R + R1   # 仅用于展示"系统总回流倍数"
            # 脱氮质量平衡：仅内回流R1携带硝态氮回到缺氧池（污泥回流R基本不带硝态氮，不计入）
            # 缺氧1完全反硝化后，残留TN浓度≈ TN_in / (1 + R1)（已包含硝化生成的氮）
            no3_res_stage1 = tn_in / (1 + R1)
            # 自流进入缺氧2深度反硝化（反硝化效率按70%计，该比例依容积/碳源而定）
            no3_after_anox2 = no3_res_stage1 * (1 - 0.7)
            # 出水总氮（含残留有机氮、出水氨氮，取3 mg/L近似）
            tn_theory = no3_after_anox2 + 3

            tn_status = "✅ 当前回流比可满足TN目标" if tn_theory <= tn_out_target else "⚠️ 内回流不足，需加大R1回流比"
            # 满足出水TN目标所需最小内回流比（基于缺氧1主反硝化段，保守值，仅与R1相关）
            min_R1 = (tn_in / tn_out_target - 1) * 100

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
            cn_ratio = cod_in / tn_in
            tn_remove = tn_in - tn_out_target
            endogenous_carbon = bod_in * 0.5  # 可生化内源碳（按易降解COD≈50% BOD计，保守取值）
            need_carbon_total = tn_remove * 4  # 反硝化总需COD（C/N=4，含安全余量；理论最小约2.86 gCOD/gNO3-N）

            # 状态判定与投加量统一以 C/N 阈值为准，避免"界面说无需、却仍算投加量"的矛盾：
            # COD/TN ≥ 4 视为内碳充足，不外加碳源（缺口强制为0）；C/N < 4 才按反硝化缺口细化计算
            if cn_ratio >= 4:
                carbon_deficit = 0.0
                carbon_status = f"✅ C/N比{cn_ratio:.1f}，无需补充碳源"
            else:
                carbon_deficit = max(0, need_carbon_total - endogenous_carbon)
                carbon_status = f"⚠️ C/N比仅{cn_ratio:.1f}，需补充碳源"

            # 两级缺氧碳源分配 7:3（求和后相互抵消，对总投加量无影响，仅内部拆分）
            carbon_anox1 = carbon_deficit * 0.7
            carbon_anox2 = carbon_deficit * 0.3
            carbon_cfg = carbon_agent_config[carbon_agent_type]
            carbon_dosage = (carbon_anox1 + carbon_anox2) / carbon_cfg["cod_eq"]  # mg/L
            carbon_daily = carbon_dosage * Q / 1000 / 1000  # 吨/天

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

            # ========== 4. 剩余污泥与污泥龄 ==========
            bod_remove = bod_in - bod_eff

            # --- 4.1 污泥龄 SRT（按实际排泥量计算，仅硝化段‑好氧1池） ---
            V_aero1 = bp["V_aero1"]
            waste_mlss = mlss * 2  # 排泥污泥浓度默认是生化池MLSS的2倍
            aer1_total_sludge = V_aero1 * mlss / 1000          # 好氧1池总污泥质量(kg)
            daily_waste_sludge = waste_sludge_volume * waste_mlss / 1000  # 每日外排污泥质量(kg/d)
            if daily_waste_sludge > 0:
                srt = aer1_total_sludge / daily_waste_sludge
            else:
                srt = 999

            # --- 4.2 剩余污泥产量（表观产率系数法，与SRT一致，恒为非负） ---
            # 注：直接用 ΔX=Y·Q·ΔBOD−Kd·V·X 在MLSS/排泥独立输入时可能出现负值，
            #      故改用表观产率 Y_obs=Y/(1+Kd·SRT) 关联系统SRT，结果稳定合理。
            Y_obs = Y / (1 + Kd * srt)
            delta_x_v = Y_obs * Q * bod_remove / 1000          # kg VSS/d
            ash_fraction = 0.20
            delta_x_total = delta_x_v / (1 - ash_fraction)
            water_content = 0.992
            dry_ratio = 1 - water_content
            sludge_wet = delta_x_total / dry_ratio / 1000



            # ========== 5. 污染物去除率（动态计算，无固定值） ==========
            cod_rate = (cod_in - cod_eff) / cod_in * 100
            nh3_rate = (nh3_in - nh3_eff) / nh3_in * 100
            tn_rate = tn_remove / tn_in * 100
            tp_rate = (tp_in - tp_out_target) / tp_in * 100

            # 保存结果供成本模块调用（单一赋值，避免重复写入导致键丢失）
            st.session_state.bio_result = {
                'schema': RESULT_SCHEMA,
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
                'waste_mlss': waste_mlss,
                # —— 以下为界面持久化展示所需的派生值 ——
                'R': R, 'R1': R1, 'total_return': total_return,
                'min_R1': min_R1, 'tn_status': tn_status,
                'tn_out_target': tn_out_target, 'tp_out_target': tp_out_target,
                'carbon_agent_type': carbon_agent_type, 'cn_ratio': cn_ratio,
                'carbon_status': carbon_status, 'cp_ratio': cp_ratio,
                'cp_need_carbon': cp_need_carbon, 'cp_status': cp_status,
                'tn_remove': tn_remove, 'need_carbon_total': need_carbon_total,
                'endogenous_carbon': endogenous_carbon, 'carbon_deficit': carbon_deficit,
                'carbon_dosage': carbon_dosage,
                'phos_agent_type': phos_agent_type, 'tp_bio_remove': tp_bio_remove,
                'tp_need_chem': tp_need_chem, 'phos_dosage': phos_dosage,
                'cod_rate': cod_rate, 'nh3_rate': nh3_rate, 'tn_rate': tn_rate, 'tp_rate': tp_rate
            }




        # ===== 持久展示：切换页面后仍保留计算结果 =====
        _stale = bool(st.session_state.get('bio_result')) and st.session_state.bio_result.get('schema') != RESULT_SCHEMA
        if _stale:
            st.info("计算结果格式已更新，请重新点击本页「计算」按钮以刷新结果。")
        if st.session_state.get('bio_result') and st.session_state.get('bio_result').get('schema') == RESULT_SCHEMA:
            bio = st.session_state.get('bio_result')
            st.markdown("---")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["回流比脱氮校核", "DO分区控制", "碳源投加计算", "除磷药剂计算", "污泥与去除率"])

            with tab1:
                st.subheader("1. 回流比校核与脱氮效果")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("污泥回流比", f"{bio['R']*100:.0f}%")
                    st.metric("内回流比 R1（好氧1→缺氧1）", f"{bio['R1']*100:.0f}%")
                    st.metric("系统总回流比 (R+R1)", f"{bio['total_return']:.1f}倍")
                with col2:
                    st.metric("理论出水TN", f"{bio['tn_theory']:.2f} mg/L")
                    st.metric("达标所需最小内回流R1", f"{max(bio['min_R1'], 100):.1f}%")
                    st.info(bio['tn_status'])
                st.write("调节建议：氨氮偏高时优先加大内回流R1；总氮深度达标可配合缺氧2外加碳源")

            with tab2:
                st.subheader("2. 各功能区溶解氧DO控制标准")
                do_data = pd.DataFrame({
                    "功能区": ["厌氧池", "第一缺氧池", "第一好氧池", "第二缺氧池", "第二好氧池"],
                    "DO控制范围 (mg/L)": ["< 0.2", "< 0.5", "1.5 ~ 3.0", "< 0.3", "1.0 ~ 2.0"],
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
                st.subheader(f"3. 碳源投加量计算（{bio['carbon_agent_type']}）")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("进水C/N比", f"{bio['cn_ratio']:.1f}")
                    st.info(bio['carbon_status'])
                    st.metric("进水碳磷比 (BOD₅/TP)", f"{bio['cp_ratio']:.1f}")
                    if bio['cp_need_carbon']:
                        st.warning(bio['cp_status'])
                    else:
                        st.success(bio['cp_status'])
                    st.divider()
                    st.write(f"总需脱除总氮：{bio['tn_remove']:.1f} mg/L")
                    st.write(f"理论总需COD：{bio['need_carbon_total']:.1f} mg/L")
                    st.write(f"内源可利用碳源：{bio['endogenous_carbon']:.1f} mg/L")
                    st.write(f"碳源总缺口：{bio['carbon_deficit']:.1f} mg/L")
                with col2:
                    st.metric(f"{bio['carbon_agent_type']}投加浓度", f"{bio['carbon_dosage']:.2f} mg/L")
                    st.metric(f"{bio['carbon_agent_type']}日投加量", f"{bio['carbon_daily']:.3f} 吨/天")

            with tab4:
                st.subheader(f"4. 化学除磷药剂计算（{bio['phos_agent_type']}）")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"生物除磷量：{bio['tp_bio_remove']:.2f} mg/L")
                    st.write(f"需化学去除磷量：{bio['tp_need_chem']:.2f} mg/L")
                with col2:
                    st.metric(f"{bio['phos_agent_type']}投加浓度", f"{bio['phos_dosage']:.2f} mg/L")
                    st.metric(f"{bio['phos_agent_type']}日投加量", f"{bio['phos_daily']:.3f} 吨/天")

            with tab5:
                st.subheader("5. 剩余污泥、污泥龄与去除率")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("污泥龄 SRT（硝化段）", f"{bio['srt']:.1f} d")
                    st.info("满足硝化菌世代时间要求" if bio['srt'] > 10 else "泥龄偏短，硝化菌易流失")
                    st.caption("注：此处SRT仅按硝化段(好氧1池)污泥量/排泥量计算，为硝化安全泥龄下限；全系统SRT需用V_total核算")
                    st.metric("每日剩余干污泥", f"{bio['sludge_dry_daily']:.2f} kg/d")
                    st.metric("湿污泥量（含水率99.2%）", f"{bio['sludge_wet_daily']:.2f} m³/d")
                with col2:
                    st.write("#### 污染物去除率")
                    st.write(f"COD去除率：{bio['cod_rate']:.1f}%")
                    st.write(f"氨氮去除率：{bio['nh3_rate']:.1f}%")
                    st.write(f"总氮去除率：{bio['tn_rate']:.1f}%")
                    st.write(f"总磷去除率：{bio['tp_rate']:.1f}%")
    # ================= 页面4：二沉池专项校核 =================
    elif page == "🏞️ 二沉池专项校核":
        st.header("🏞️ 二沉池专项校核")
        bp = st.session_state.base_params

        col1, col2 = st.columns(2)
        with col1:
            mlss = num_input("MLSS 混合液浓度 (mg/L)", value=3500)
            R = num_input("污泥回流比 R (%)", value=100) / 100
            sv30 = num_input("SV30 沉降比 (%)", value=25)
        with col2:
            Q_max = num_input("最大时流量 (m³/h)", value=bp['Q_max'])
            area = num_input("二沉池总表面积 (m²)", value=bp['settler_area'])
            depth = num_input("二沉池有效水深 (m)", value=bp['settler_depth'])

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
            # 持久化结果，便于切换页面后再次查看
            st.session_state.settler_result = {
                'schema': RESULT_SCHEMA,
                'q_surface': q_surface, 'q_status': q_status,
                'ssl': ssl, 'ssl_status': ssl_status,
                'svi': svi, 'svi_status': svi_status,
                'hrt': hrt
            }
            st.success("✅ 二沉池校核完成，结果已保存，可切换页面后再回来查看")



        # ===== 持久展示：切换页面后仍保留计算结果 =====
        _stale = bool(st.session_state.get('settler_result')) and st.session_state.settler_result.get('schema') != RESULT_SCHEMA
        if _stale:
            st.info("计算结果格式已更新，请重新点击本页「计算」按钮以刷新结果。")
        if st.session_state.get('settler_result') and st.session_state.get('settler_result').get('schema') == RESULT_SCHEMA:
            res = st.session_state.settler_result
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("表面水力负荷", f"{res['q_surface']:.3f} m³/(m²·h)")
                st.info(res['q_status'])
                st.caption("推荐值：最大时 ≤ 0.6~1.5 m³/(m²·h)")
            with col2:
                st.metric("固体表面负荷", f"{res['ssl']:.2f} kgMLSS/(m²·d)")
                st.info(res['ssl_status'])
                st.caption("推荐值：≤ 150 kgMLSS/(m²·d)")
            with col3:
                st.metric("SVI 污泥体积指数", f"{res['svi']:.1f} mL/g")
                st.info(res['svi_status'])
                st.caption("正常范围：70 ~ 150 mL/g")

            st.metric("二沉池水力停留时间 (HRT)", f"{res['hrt']:.2f} h")
            st.caption("推荐值：按最大时流量校核，一般 ≥ 1.5~2.0 h")

            st.markdown("---")
            st.subheader("故障处置建议")
            if res['svi'] >= 150:
                st.warning("污泥膨胀风险：建议降低MLSS、加大排泥量、提高好氧池DO、控制进水有机负荷")
            elif res['svi'] < 70:
                st.warning("污泥老化：建议减少排泥、适当提高污泥负荷、检查进水营养比")
            if res['ssl'] >= 150:
                st.warning("跑泥风险：建议提高污泥回流比、降低进水量、增加排泥频次")
            st.info("好氧池末端维持1.0~2.0mg/L DO，可有效防止二沉池内反硝化导致的污泥上浮")
    # ================= 页面5：浸没式超滤工况 =================
    elif page == "💠 浸没式超滤工况":
        st.header("💠 浸没式超滤实时工况监控")
        st.caption("模拟实时监测 + 预警 + AI 运维指导")
        #：运行通量 22~26 LMH、跨膜压差(TMP) 5~15 kPa 缓慢上升、"
        #           "曝气强度 60~70 m³/(m²·h) 波动、出水浊度 0.02~0.06 NTU 波动；TMP 报警限值 35 kPa
        bp = st.session_state.base_params

        # ---------- ① 现场信息（膜系统配置，用于 CEB 预测与校核） ----------
        with st.expander("① 现场信息录入（膜系统配置，用于 CEB 预测与校核）", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                Q_uf = num_input("UF 设计处理水量 (m³/d)", value=bp['Q_design'], min_value=0.0, key="uf_Q")
                area_per_module = num_input("单支膜组件面积 (m²)", value=18.0, min_value=0.0, key="uf_area")
            with c2:
                modules_per_rack = num_input("单膜箱/膜架组件数", value=42, min_value=1, key="uf_mod")
                rack_count = num_input("膜箱/膜架数量", value=40, min_value=1, key="uf_rack")
            with c3:
                recovery = num_input("设计回收率 (%)", value=96.0, min_value=50.0, max_value=100.0, key="uf_rec")
                tmp_alarm = num_input("TMP 报警限值 (kPa)", value=35.0, min_value=10.0, key="uf_alarm")
            with c4:
                ceb_thr = num_input("CEB 反洗触发 TMP (kPa)", value=15.0, min_value=10.0, key="uf_ceb")
                run_days = num_input("已连续运行天数", value=42, min_value=0, key="uf_days")
            area_installed = area_per_module * modules_per_rack * rack_count
            st.caption(f"已安装膜面积 ≈ {area_installed:.0f} m²（{modules_per_rack:.0f}×{rack_count:.0f} 支组件）；"
                       f"设计产水量 ≈ {Q_uf*recovery/100:.0f} m³/d")

        # ---------- 模拟状态初始化 ----------
        if 'uf_sim' not in st.session_state:
            st.session_state.uf_sim = {'t': [], 'flux': [], 'tmp': [], 'aer': [], 'turb': []}
            st.session_state.uf_t = 0
        sim = st.session_state.uf_sim

        def uf_step():
            t = st.session_state.uf_t + 1
            # TMP 缓慢上升趋势：起始约 5 kPa，随运行天数与采样帧数线性抬升 + 噪声
            base_tmp = 5.0 + 0.06 * run_days + 0.05 * t
            tmp = base_tmp + float(np.random.normal(0, 0.35))
            tmp = max(tmp, 3.0)
            flux = float(np.random.uniform(22, 26))
            aer = float(np.random.uniform(60, 70))
            turb = float(np.random.uniform(0.02, 0.06))
            sim['t'].append(t)
            sim['flux'].append(flux)
            sim['tmp'].append(tmp)
            sim['aer'].append(aer)
            sim['turb'].append(turb)
            st.session_state.uf_t = t

        # ---------- ②③ 实时监测 + 指标自包含 fragment（每 1.5s 自动重跑） ----------
        # 关键改动：把"采集下一帧 / 自动滚动 checkbox / 4 个指标 metric / 报警文案"
        # 全部塞进同一个 @st.fragment(run_every=1.5)，避免 v12 时 fragment 内只推进数据、
        # 但 4 个 metric 在 fragment 外、run_every 不触发整页 rerun → 数值看似"不刷新"的 bug。
        @st.fragment(run_every=2)
        def _uf_live():
            if page != "💠 浸没式超滤工况":
                return
            # 先按需推进一帧（首次或自动滚动开启时）
            if not sim['t']:
                uf_step()
            elif st.session_state.get("uf_auto", True):
                uf_step()

            # ---------- ② 控制按钮 + 自动滚动开关 ----------
            st.markdown("---")
            st.subheader("② 实时监测模拟")
            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                if st.button("▶ 开始 / 重置模拟", type="primary", key="uf_start"):
                    st.session_state.uf_sim = {'t': [], 'flux': [], 'tmp': [], 'aer': [], 'turb': []}
                    st.session_state.uf_t = 0
                    st.rerun(scope="fragment")
            with b2:
                if st.button("⏭ 采集下一帧", key="uf_next"):
                    uf_step()
                    st.rerun(scope="fragment")
            with b3:
                st.checkbox("自动滚动（每 1.5 s 一帧）", value=True, key="uf_auto")

            # ---------- ③ 当前实时指标与预警 ----------
            flux_now = sim['flux'][-1]
            tmp_now = sim['tmp'][-1]
            aer_now = sim['aer'][-1]
            turb_now = sim['turb'][-1]
            delta = tmp_now - sim['tmp'][-2] if len(sim['tmp']) > 1 else None
            # 把派生指标写进 session_state，供 fragment 外的「④ AI 预测」区块读取（避免 NameError）
            st.session_state['uf_aer'] = aer_now
            st.session_state['uf_turb'] = turb_now
            st.session_state['uf_tmp'] = tmp_now
            st.session_state['uf_flux'] = flux_now

            st.markdown("---")
            st.subheader("③ 当前实时指标与预警")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("运行通量", f"{flux_now:.1f} LMH", help="设计区间 22~26 LMH")
            with m2:
                st.metric("跨膜压差 TMP", f"{tmp_now:.2f} kPa",
                          delta=f"{delta:+.2f}" if delta is not None else None)
            with m3:
                st.metric("曝气强度", f"{aer_now:.1f} m³/(m²·h)", help="设计区间 60~70")
            with m4:
                st.metric("出水浊度", f"{turb_now:.3f} NTU", help="设计区间 0.02~0.06")

            if tmp_now >= tmp_alarm:
                st.error(f"⛔ TMP 已达报警限值（{tmp_alarm:.0f} kPa）：立即执行 CIP 化学清洗 / 完整性检测，必要时降负荷")
            elif tmp_now >= ceb_thr:
                st.warning(f"⚠️ TMP 超过 CEB 反洗阈值（{ceb_thr:.0f} kPa）：建议尽快安排 CEB 加强反洗")
            else:
                st.success(f"✅ 工况正常：TMP 距 CEB 阈值尚有 {ceb_thr - tmp_now:.1f} kPa 余量")


            # ---------- ④ AI 预测与运维指导 ----------
            # 从 session_state 读取 fragment 内实时计算的派生指标（避免 NameError）
            aer_now = st.session_state.get('uf_aer', 0.0)
            turb_now = st.session_state.get('uf_turb', 0.0)
            tmp_now = st.session_state.get('uf_tmp', 0.0)
            flux_now = st.session_state.get('uf_flux', 0.0)
            st.markdown("---")
            st.subheader("④ AI 预测与运维指导")
            n = len(sim['tmp'])
            # 稳定斜率：取近 30 帧帧间增量的中位数，抑制单帧噪声导致的预测跳动
            if n >= 10:
                win = min(n, 30)
                _inc = np.diff(sim['tmp'][-win:])
                slope = float(np.median(_inc)) if len(_inc) else 0.0
            else:
                slope = 0.0
            # 把 slope 写进 session_state，供 fragment 外的「④ 大模型诊断」按钮读取（避免 NameError / 作用域问题）
            st.session_state['uf_slope'] = slope
            pred_lines = []
            if n < 10:
                pred_lines.append("- 样本不足（需≥10帧），继续采集以稳定 CEB/报警预测。")
            elif slope > 1e-4:
                frames_to_ceb = (ceb_thr - tmp_now) / slope
                frames_to_alarm = (tmp_alarm - tmp_now) / slope
                pred_lines.append(
                f"- **CEB 反洗预测**：按当前上升趋势（≈{slope:.3f} kPa/帧，基于近{min(n,30)}帧中位增量），"
                f"约 **{max(frames_to_ceb, 0):.0f} 帧**后达到 CEB 阈值（{ceb_thr:.0f} kPa），"
                f"约合 **{max(frames_to_ceb, 0)/24:.1f} 天**（按 24 帧/天计）。")
                if frames_to_alarm > 0:
                    pred_lines.append(
                    f"- **报警预测**：约 **{frames_to_alarm:.0f} 帧**后触及报警限值（{tmp_alarm:.0f} kPa），"
                    f"约合 **{frames_to_alarm/24:.1f} 天**；请在报警前完成 CEB。")
                else:
                    pred_lines.append("- **报警预测**：当前 TMP 已超过报警限值，需立即处置。")
            else:
                pred_lines.append("- 当前 TMP 趋势平稳（斜率≈0），未见明显污染上升，维持现有维护性清洗周期即可。")
    
            anomalies = []
            if aer_now < 60:
                anomalies.append("曝气强度低于 60 m³/(m²·h)，膜面剪切不足，易加速污染——建议提高曝气。")
            elif aer_now > 70:
                anomalies.append("曝气强度高于 70，能耗偏高但利于控污——可酌情下调。")
            if turb_now > 0.06:
                anomalies.append("出水浊度 >0.06 NTU，疑似膜丝破损或断丝——建议做完整性检测。")
            if tmp_now >= ceb_thr and slope > 0:
                anomalies.append("TMP 已超 CEB 阈值且仍在上升，判断为可逆污染主导，优先 CEB（次氯酸钠/柠檬酸）而非 CIP。")
    
            st.info("**AI 趋势预测（机理+统计）**\n" + "\n".join(pred_lines))
            if anomalies:
                st.warning("**AI 异常诊断（规则引擎）**\n- " + "\n- ".join(anomalies))
            else:
                st.success("**AI 异常诊断**：各参数均在正常波动区间，未发现异常。")


        _uf_live()

        # ---------- ④ 高阶 AI：大模型运维诊断报告（置于 fragment 之外） ----------
        # 说明：若把"调用大模型"按钮写在上面的 _uf_live() fragment 内，会被 run_every=2 的
        # 定时重跑每 2 秒清空/重置，导致"点击后回答一闪而过、按钮状态丢失"。故将其移出 fragment，
        # 作为本页静态区块；实时数据从 session_state（由 fragment 每 2s 写入）读取，保证回答稳定留存。
        _llm_mode = st.session_state.get('_llm_status', ('offline',))[0]
        st.caption(f"AI 能力档位：当前为「{_llm_mode}」模式"
                   f"（offline=离线规则引擎；local/cloud=已接入大模型，可生成自然语言诊断报告）")
        if st.button("🤖 调用大模型生成运维诊断报告", key="uf_ai"):
            _flux = st.session_state.get('uf_flux', 0.0)
            _tmp = st.session_state.get('uf_tmp', 0.0)
            _aer = st.session_state.get('uf_aer', 0.0)
            _turb = st.session_state.get('uf_turb', 0.0)
            _slope = st.session_state.get('uf_slope', 0.0)
            ctx = (f"现场信息：UF设计水量{Q_uf:.0f} m³/d，已安装膜面积{area_installed:.0f} m²，"
                   f"回收率{recovery:.0f}%，已连续运行{run_days:.0f}天。\n"
                   f"当前实时：通量{_flux:.1f} LMH，TMP{_tmp:.2f} kPa（CEB阈值{ceb_thr:.0f}，报警{tmp_alarm:.0f}），"
                   f"曝气{_aer:.1f} m³/(m²·h)，出水浊度{_turb:.3f} NTU。\n"
                   f"TMP趋势斜率≈{_slope:.3f} kPa/帧。")
            q = ("请作为膜法水处理工程师，给出：1) TMP 上升的根因判断；2) 是否需要及何时安排 CEB 反洗；"
                 "3) 具体清洗配方与运行调整建议；4) 防止进一步污染的措施。语言简洁专业。")
            out = st.empty()
            text = ""
            for piece in ai_assistant_stream(q, ctx, kb_dir=None):
                text += piece
                out.markdown(text)

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
                aeration_kw = num_input("曝气风机总功率 (kW)", value=220, key="cost_aeration_kw")
                backflow_kw = num_input("污泥回流泵总功率 (kW)", value=30, key="cost_backflow_kw")
                internal_kw = num_input("内回流泵总功率 (kW)", value=45, key="cost_internal_kw")
                mix_kw = num_input("搅拌/推流器总功率 (kW)", value=25, key="cost_mix_kw")
            with col2:
                pump_kw = num_input("进水泵房总功率 (kW)", value=55, key="cost_pump_kw")
                dewater_kw = num_input("污泥脱水系统功率 (kW)", value=37, key="cost_dewater_kw")
                dewater_h = num_input("脱水系统日运行时长 (h)", value=8, key="cost_dewater_h")
                other_kw = num_input("辅助设备总功率 (kW)", value=20, key="cost_other_kw")

            if st.button("计算电耗成本", type="primary", key="cost_btn_power"):
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
                st.session_state.power_result = {
                    'schema': RESULT_SCHEMA,
                    'e_aeration': e_aeration, 'e_backflow': e_backflow, 'e_internal': e_internal,
                    'e_mix': e_mix, 'e_pump': e_pump, 'e_dewater': e_dewater, 'e_other': e_other,
                    'e_total_day': e_total_day, 'cost_day': cost_day, 'cost_month': cost_month,
                    'unit_power': unit_power, 'unit_cost': unit_cost
                }

            # ===== 持久展示：切换页面后仍保留计算结果 =====
            _stale = bool(st.session_state.get('power_result')) and st.session_state.power_result.get('schema') != RESULT_SCHEMA
            if _stale:
                st.info("计算结果格式已更新，请重新点击本页「计算」按钮以刷新结果。")
            if st.session_state.get('power_result') and st.session_state.get('power_result').get('schema') == RESULT_SCHEMA:
                res = st.session_state.power_result
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### 分项日电耗")
                    power_data = pd.DataFrame({
                        "设备类别": ["曝气系统", "污泥回流泵", "内回流泵", "搅拌推流器", "进水泵房", "污泥脱水", "辅助设备"],
                        "日耗电量 (kWh)": [res['e_aeration'], res['e_backflow'], res['e_internal'], res['e_mix'], res['e_pump'], res['e_dewater'], res['e_other']]
                    })
                    st.dataframe(power_data, use_container_width=True, hide_index=True)
                with col2:
                    st.metric("日总耗电量", f"{res['e_total_day']:.1f} kWh")
                    st.metric("日电费", f"{res['cost_day']:.2f} 元")
                    st.metric("月电费", f"{res['cost_month']:,.2f} 元")
                    st.metric("吨水电耗", f"{res['unit_power']:.3f} kWh/m³")
                    st.metric("吨水电费", f"{res['unit_cost']:.3f} 元/m³")

        # 药剂成本
        with tab2:
            st.subheader("二、药剂成本核算")
            col1, col2 = st.columns(2)
            with col1:
                naclo_daily = num_input("次氯酸钠日用量 (吨)", value=0.5, key="cost_naclo_daily")
                pam_daily = num_input("PAM日用量 (吨)", value=0.08, key="cost_pam_daily")
                hcl_daily = num_input("盐酸日用量 (吨，pH调节)", value=0.05, key="cost_hcl_daily")
            with col2:
                st.info("除磷药剂、碳源用量自动读取生化计算结果")
                if st.button("加载生化计算药剂用量", key="cost_btn_load_med"):
                    if st.session_state.bio_result:
                        bio = st.session_state.bio_result
                        st.success(f"已加载：{bio['phos_agent_name']} {bio['phos_daily']:.3f}吨/天，{bio['carbon_agent_name']} {bio['carbon_daily']:.3f}吨/天")
                    else:
                        st.warning("请先在「生化核心计算」页完成计算")

            if st.button("计算药剂总成本", type="primary", key="cost_btn_med"):
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
                st.session_state.med_result = {
                    'schema': RESULT_SCHEMA,
                    'phos_name': phos_name, 'carbon_name': carbon_name,
                    'phos_daily': phos_daily, 'carbon_daily': carbon_daily,
                    'naclo_daily': naclo_daily, 'pam_daily': pam_daily, 'hcl_daily': hcl_daily,
                    'cost_phos': cost_phos, 'cost_carbon': cost_carbon, 'cost_naclo': cost_naclo,
                    'cost_pam': cost_pam, 'cost_hcl': cost_hcl,
                    'total_day': total_day, 'total_month': total_month, 'unit_cost': unit_cost
                }

            # ===== 持久展示 =====
            _stale = bool(st.session_state.get('med_result')) and st.session_state.med_result.get('schema') != RESULT_SCHEMA
            if _stale:
                st.info("计算结果格式已更新，请重新点击本页「计算」按钮以刷新结果。")
            if st.session_state.get('med_result') and st.session_state.get('med_result').get('schema') == RESULT_SCHEMA:
                res = st.session_state.med_result
                col1, col2 = st.columns(2)
                with col1:
                    med_data = pd.DataFrame({
                        "药剂名称": [res['phos_name'], res['carbon_name'], "次氯酸钠(消毒)", "PAM(助凝)", "盐酸(pH调节)"],
                        "日用量 (吨)": [res['phos_daily'], res['carbon_daily'], res['naclo_daily'], res['pam_daily'], res['hcl_daily']],
                        "日成本 (元)": [res['cost_phos'], res['cost_carbon'], res['cost_naclo'], res['cost_pam'], res['cost_hcl']]
                    })
                    st.dataframe(med_data, use_container_width=True, hide_index=True)
                with col2:
                    st.metric("日药剂总成本", f"{res['total_day']:.2f} 元")
                    st.metric("月药剂总成本", f"{res['total_month']:,.2f} 元")
                    st.metric("吨水药剂成本", f"{res['unit_cost']:.3f} 元/m³")

            # ===== 药剂成本优化方案（智能选型，仅作对照说明，不覆盖上方实际计算）=====
            st.divider()
            st.subheader("💡 药剂成本优化方案（智能选型对照）")
            st.caption("以下为在满足相同碳源 / 除磷需求下，由线性规划在全部可选药剂中自动挑选成本最低组合的「理想方案」。"
                       "仅用于说明「若改用最优药剂组合，预计可较当前选型节省多少」，不改变上方按实际选型计算的成本。")
            bio_opt = get_compute_result("bio_result")
            if not bio_opt:
                st.warning("⚠️ 请先在「🧪 生化核心计算」页完成计算，以读取碳源缺口与除磷需求。")
            else:
                if st.button("🤖 查看 AI 智能优化方案可节省金额", type="primary", key="cost_opt_btn"):
                    # 结果持久化到 session_state，避免被电耗 tab 自动刷新触发的整页 rerun 冲掉
                    st.session_state.dosing_opt_result = optimize_dosing(bp, bio_opt)

                # 持久展示：依据 session_state 渲染，autorefresh 触发的整页 rerun 也不会丢失结果
                if st.session_state.get('dosing_opt_result'):
                    opt = st.session_state.dosing_opt_result

                    # 高亮对比卡片
                    st.markdown("#### 📊 成本对比（当前选型 vs AI 最优组合）")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("当前选型日药剂成本", f"{opt['cur_cost']:.2f} 元/天")
                    c2.metric("AI 最优组合日成本", f"{opt['opt_cost']:.2f} 元/天", delta=f"-{(opt['cur_cost']-opt['opt_cost'])/opt['cur_cost']*100:.1f}%")
                    if opt["saving_pct"] > 0:
                        c3.metric("💰 每日预计节省", f"{opt['saving']:.2f} 元/天", delta=f"{opt['saving_pct']:.1f}%")
                    else:
                        c3.metric("💰 每日预计节省", "0.00 元/天", delta="0.0%")

                    # 进度条式对比
                    st.markdown("#### 成本降幅可视化")
                    cur_w, opt_w = opt['cur_cost'], opt['opt_cost']
                    if cur_w > 0:
                        ratio = max(0.0, min(1.0, opt_w / cur_w))
                        st.progress(1.0 - ratio, text=f"AI 优化后较当前成本降低 {(1-ratio)*100:.1f}%")
                    else:
                        st.progress(0.0, text="当前成本为 0，无法计算降幅")

                    # 推荐组合
                    rows_opt = []
                    for name, dose in opt["rec_carbon"].items():
                        if dose > 0:
                            rows_opt.append([f"碳源·{name}", f"{dose:.3f} 吨/天"])
                    for name, dose in opt["rec_phos"].items():
                        if dose > 0:
                            rows_opt.append([f"除磷剂·{name}", f"{dose:.3f} 吨/天"])
                    if rows_opt:
                        st.markdown("#### 🧪 AI 推荐药剂组合")
                        st.table(pd.DataFrame(rows_opt, columns=["药剂", "推荐日投加量"]))

                    # 醒目总结
                    if opt["saving_pct"] > 0:
                        st.success(
                            f"✅ 若采用 AI 推荐的最优组合，预计 **每日节省 {opt['saving']:.2f} 元**，"
                            f"**每月节省约 {opt['saving']*30:.0f} 元**，**每年可降本约 {opt['saving']*365:.0f} 元**。"
                        )
                        st.info(f"提示：当前缺口为 {opt['carbon_deficit']:.1f} mg/L COD、{opt['tp_need_chem']:.2f} mg/L 化学除磷(P)。")
                    else:
                        st.info("当前选型已接近成本最优，AI 智能方案无可进一步节省空间。")
                else:
                    st.info("点击上方按钮，可查看在当前工艺需求下的最优药剂组合及预计节省金额。")

        # 污泥处置成本
        with tab3:
            st.subheader("三、剩余污泥处置成本")
            col1, col2 = st.columns(2)
            with col1:
                water_rate = num_input("脱水后污泥含水率 (%)", value=80, key="cost_water_rate") / 100
                pam_dosage = num_input("吨干泥PAM投加量 (kg/t)", value=4, key="cost_pam_dosage")
            with col2:
                st.info("污泥产量自动读取生化计算结果")
                if st.button("加载生化计算污泥量", key="cost_btn_load_sludge"):
                    if st.session_state.bio_result:
                        st.success(f"已加载：每日干污泥 {st.session_state.bio_result['sludge_dry_daily']:.2f} kg")
                    else:
                        st.warning("请先在「生化核心计算」页完成计算")

            if st.button("计算污泥处置成本", type="primary", key="cost_btn_sludge"):
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
                st.session_state.sludge_result = {
                    'schema': RESULT_SCHEMA,
                    'wet_daily': wet_daily, 'cost_pam_day': cost_pam_day, 'cost_dispose_day': cost_dispose_day,
                    'total_day': total_day, 'total_month': total_month, 'unit_cost': unit_cost
                }

            # ===== 持久展示 =====
            _stale = bool(st.session_state.get('sludge_result')) and st.session_state.sludge_result.get('schema') != RESULT_SCHEMA
            if _stale:
                st.info("计算结果格式已更新，请重新点击本页「计算」按钮以刷新结果。")
            if st.session_state.get('sludge_result') and st.session_state.get('sludge_result').get('schema') == RESULT_SCHEMA:
                res = st.session_state.sludge_result
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("每日脱水湿污泥量", f"{res['wet_daily']:.2f} 吨")
                    st.write(f"脱水PAM日费用：{res['cost_pam_day']:.2f} 元")
                    st.write(f"污泥外运处置费：{res['cost_dispose_day']:.2f} 元")
                with col2:
                    st.metric("日污泥处置总成本", f"{res['total_day']:.2f} 元")
                    st.metric("月污泥处置总成本", f"{res['total_month']:,.2f} 元")
                    st.metric("吨水污泥处置成本", f"{res['unit_cost']:.3f} 元/m³")

        # 全成本汇总
        with tab4:
            st.subheader("四、全厂全成本汇总分析")
            if st.button("生成全成本报表", type="primary", key="cost_btn_total"):
                power_cost = getattr(st.session_state, 'power_cost_month', 70000)
                med_cost = getattr(st.session_state, 'med_cost_month', 120000)
                sludge_cost = getattr(st.session_state, 'sludge_cost_month', 45000)
                staff_cost = bp['staff_num'] * bp['staff_salary']
                maintain_cost = bp['maintain_cost']
                other_cost = bp['other_cost']

                total_month = power_cost + med_cost + sludge_cost + staff_cost + maintain_cost + other_cost
                Q_month = bp['Q_actual'] * 30
                unit_cost = total_month / Q_month

                st.session_state.total_cost_result = {
                    'schema': RESULT_SCHEMA,
                    'power_cost': power_cost, 'med_cost': med_cost, 'sludge_cost': sludge_cost,
                    'staff_cost': staff_cost, 'maintain_cost': maintain_cost, 'other_cost': other_cost,
                    'total_month': total_month, 'unit_cost': unit_cost
                }

            # ===== 持久展示 =====
            _stale = bool(st.session_state.get('total_cost_result')) and st.session_state.total_cost_result.get('schema') != RESULT_SCHEMA
            if _stale:
                st.info("计算结果格式已更新，请重新点击本页「计算」按钮以刷新结果。")
            if st.session_state.get('total_cost_result') and st.session_state.get('total_cost_result').get('schema') == RESULT_SCHEMA:
                res = st.session_state.total_cost_result
                power_cost = res['power_cost']; med_cost = res['med_cost']; sludge_cost = res['sludge_cost']
                staff_cost = res['staff_cost']; maintain_cost = res['maintain_cost']; other_cost = res['other_cost']
                total_month = res['total_month']; unit_cost = res['unit_cost']

                labels = ["电费", "药剂费", "污泥处置", "人员工资", "维修耗材", "其他"]
                values = [power_cost, med_cost, sludge_cost, staff_cost, maintain_cost, other_cost]
                colors = ["#36a2eb", "#4bc0c0", "#ff9f40", "#ff6384", "#9966ff", "#c9cbcf"]
                total_cost = sum(values)
                slice_texts = []
                for i, v in enumerate(values):
                    pct = v / total_cost * 100
                    if pct >= 6:
                        slice_texts.append(f"{labels[i]}<br>{pct:.1f}%")
                    else:
                        slice_texts.append(f"{pct:.1f}%")
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    text=slice_texts,
                    textinfo="text",
                    textposition="inside",
                    marker=dict(colors=colors, line=dict(color="white", width=2)),
                    textfont=dict(size=13, family="Microsoft YaHei, PingFang SC, Hiragino Sans GB, sans-serif"),
                    hovertemplate="%{label}<br>%{value:,.0f} 元<br>占比 %{percent}<extra></extra>"
                )])
                fig.update_layout(
                    title=dict(
                        text="月度运行成本构成占比",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=16, family="Microsoft YaHei, PingFang SC, Hiragino Sans GB, sans-serif")
                    ),
                    font=dict(family="Microsoft YaHei, PingFang SC, Hiragino Sans GB, sans-serif"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
                    margin=dict(t=50, b=60, l=30, r=30),
                    width=600,
                    height=520,
                    autosize=False
                )

                col1, col2 = st.columns([1, 1.2])
                with col1:
                    st.write("#### 月度成本分项明细")
                    cost_data = pd.DataFrame({
                        "成本类别": ["电费", "药剂费", "污泥处置费", "人员工资", "设备维修费", "其他杂费"],
                        "月度费用 (元)": [power_cost, med_cost, sludge_cost, staff_cost, maintain_cost, other_cost],
                        "占比": [f"{v/total_month*100:.1f}%" for v in values]
                    })
                    st.dataframe(cost_data, use_container_width=True, hide_index=True)

                    st.markdown("---")
                    st.metric("📌 月度运行总成本", f"{total_month:,.2f} 元")
                    st.metric("📌 年度运行总成本", f"{total_month*12:,.2f} 元")
                    st.metric("📌 吨水处理综合成本", f"{unit_cost:.3f} 元/吨")

                with col2:
                    st.plotly_chart(fig, use_container_width=True)

    # ================= 页面13：报表导出 =================
    elif page == "📊 报表导出":
        st.header("📊 计算报表导出")
        st.caption("将当前所有计算结果汇总导出为排版精美的中文 PDF 报表（中文字体已嵌入，手机/电脑均可直接查看）")

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

        if not HAS_REPORTLAB:
            st.warning(
                "⚠️ 当前部署环境缺少 `reportlab` 库，PDF 导出暂时不可用。\n\n"
                "**修复方法（二选一）：**\n"
                "1. **Streamlit Cloud 部署**：确认 GitHub 仓库的 `requirements.txt` 包含 `reportlab==4.2.2`，"
                "然后在 Streamlit Cloud 管理后台点击 『Reboot app』或 『Manage app → Reboot』刷新依赖。\n"
                "2. **本地运行**：执行 `pip install reportlab==4.2.2` 后重启 Streamlit。\n\n"
                "下方可先下载 CSV 版报表作为临时替代。"
            )

        if st.button("生成并下载 PDF 报表", type="primary", use_container_width=True):
            bp = st.session_state.base_params
            bio = get_compute_result("bio_result")
            cost = get_compute_result("total_cost_result")
            try:
                pdf_buf = export_pdf_report(bp, bio, cost)
                st.success("✅ PDF 报表生成完成，可直接在手机/电脑上打开查看（中文字体已嵌入）")
                st.download_button(
                    label="📥 下载中文PDF报表",
                    data=pdf_buf.getvalue(),
                    file_name="五段Bardenpho污水厂运行报表.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except ModuleNotFoundError as e:
                st.error(f"⚠️ PDF 生成失败：{e}\n\n"
                         f"**请检查**：GitHub 上的 `requirements.txt` 是否已写入 `reportlab==4.2.2`，"
                         f"并在 Streamlit Cloud 后台点击 Reboot 刷新环境。")
            except Exception as e:
                st.error(f"⚠️ PDF 生成失败：{e}（请确认已安装依赖 reportlab：pip install reportlab）")

    # ================= 页面8：AI 预测预警 =================
    elif page == "🔮 AI 预测预警":
        st.header("🔮 出水水质AI预测预警")
        st.caption("多模型集成预测：Holt-Winters + 谐波回归 + 季节朴素，经历史回测逆误差自动加权选优；"
                   "数据源自内置「虚拟水厂」机理仿真引擎（五段 Bardenpho 动力学生成、物理自洽），"
                   "可预测出水水质、膜系统运行及进水负荷等关键参数。")

        # ---- 数据源：优先「虚拟水厂」机理仿真引擎，失败回退合成演示数据 ----
        PLANT_CSV = os.path.join(SCRIPT_DIR, "plant_sim_history.csv")
        df = None
        if os.path.exists(PLANT_CSV):
            try:
                df = pd.read_csv(PLANT_CSV)
                df["时间"] = pd.to_datetime(df["时间"])
                df = df.set_index("时间").sort_index()
            except Exception:
                df = None
        if df is None:
            try:
                st.info("⚙️ 正在生成「虚拟水厂」机理仿真数据（五段 Bardenpho 动力学，约数秒）…")
                df = simulate_plant(n_steps=24 * 60, seed=20260808)
                try:
                    df.to_csv(PLANT_CSV, index=False, encoding="utf-8-sig")
                except Exception:
                    pass
            except Exception:
                st.warning("机理仿真引擎不可用，已回退内置合成演示数据。")
                df = _build_sample_df()
            df["时间"] = pd.to_datetime(df["时间"])
            df = df.set_index("时间").sort_index()

        # 可预测指标：水质/膜运行/进水负荷（能耗成本类机理运行参数已精简移除）
        var_options = {
            "出水COD(mg/L)": 30, "出水NH3-N(mg/L)": 1.5, "出水TN(mg/L)": 15,
            "出水TP(mg/L)": 0.3, "UF出水浊度(NTU)": None,
            "跨膜压差(kPa)": None, "进水流量(m3/h)": None, "进水COD(mg/L)": None,
            "进水TN(mg/L)": None, "进水TP(mg/L)": None, "DO_好氧池(mg/L)": None,
        }
        # 动态过滤：兜底合成数据可能缺少机理字段，只保留实际存在的列（避免 KeyError）
        var_options = {k: v for k, v in var_options.items() if k in df.columns}

        season = 24
        # 顶部两个选择器分两列；按钮、指标卡、图表、回测表统一放到 col 外，铺满主区
        col1, col2 = st.columns([2, 1])
        with col1:
            var = st.selectbox("预测指标", list(var_options.keys()))
        with col2:
            horizon = st.selectbox("预测步长（小时）", [24, 48, 168], index=0)
        if st.button("🚀 运行预测", type="primary", key="predict_btn"):
            with st.spinner("正在多模型回测与集成预测…"):
                series = df[var].dropna().to_numpy(dtype=float)
                res = smart_forecast(series, season=int(season), h=int(horizon))
            st.session_state.predict_result = {
                "schema": RESULT_SCHEMA, "var": var, "horizon": int(horizon),
                "res": res, "std": var_options[var], "season": int(season),
            }
        if st.session_state.get("predict_result", {}).get("schema") == RESULT_SCHEMA:
            pr = st.session_state.predict_result
            res = pr["res"]; var = pr["var"]; std = pr["std"]
            s = res["series"]; fc = res["forecast"]; lo = res["lower"]; up = res["upper"]
            idx_hist = df.index
            last = idx_hist[-1]
            idx_fc = pd.date_range(last + pd.Timedelta(hours=1), periods=len(fc), freq="h")

            # ---- 指标卡（铺满主区宽度） ----
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("预测均值", f"{fc.mean():.2f}")
            c2.metric("预测峰值", f"{fc.max():.2f}")
            n_warn = int((up > std).sum()) if std is not None else 0
            c3.metric("超标风险时点", f"{n_warn}/{len(fc)}" if std is not None else "—")
            c4.metric("历史异常点", f"{len(res['anomalies'])}")

            # ---- 主预测图（铺满主区宽度） ----
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx_hist[-7 * 24:], y=s[-7 * 24:],
                                     name="历史(近7天)", line=dict(color="#0E7490")))
            fig.add_trace(go.Scatter(x=idx_fc, y=up, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=idx_fc, y=lo, line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(20,184,166,0.22)", name="95%置信区间"))
            fig.add_trace(go.Scatter(x=idx_fc, y=fc, name="AI集成预测",
                                     line=dict(color="#F59E0B", dash="dot", width=2)))
            if std is not None:
                fig.add_hline(y=std, line=dict(color="#DC2626", dash="dash"))
                fig.add_annotation(
                    x=0.99, y=std, xref="paper", yref="y",
                    text=f"标准限值 {std}",
                    showarrow=False,
                    font=dict(color="#DC2626", size=13),
                    xanchor="right", yanchor="bottom",
                    yshift=3,
                    bgcolor="white"
                )
            fig.update_layout(title=dict(text=f"{var} 未来 {pr['horizon']} 小时 AI 预测", x=0.02, xanchor="left"),
                              xaxis_title="时间", yaxis_title=var, template="plotly_white",
                              legend=dict(orientation="h", x=1.0, y=1.02,
                                           xanchor="right", yanchor="bottom",
                                           font=dict(color="black")),
                              margin=dict(t=60, b=80))
            fig.update_xaxes(tickformat="%m月%d日", dtick=86400000.0, tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # ---- 预警（概率化 + 机理联动，item 10）----
            if std is not None:
                # 基于残差σ的正态近似，估算整体达标风险概率（纯 numpy，无需 scipy）
                z = (std - fc) / max(res["sigma"], 1e-9)
                risk_pct = float(np.clip(1 - _norm_cdf(z), 0, 1).mean()) * 100
                if n_warn > 0 or risk_pct > 5:
                    bio = get_compute_result("bio_result")
                    st.error(f"⚠️ 预测区间上限有 {n_warn}/{len(fc)} 个时点超过标准限值 {std} mg/L；"
                             f"按历史波动估计，整体达标风险概率约 **{risk_pct:.0f}%**。建议提前调控。")
                    advice = mechanism_advice(var, bio)
                    if advice:
                        st.info("🔧 机理联动建议：" + advice)
                else:
                    st.success(f"✅ 预测期内出水 {var} 预计均低于标准限值 {std} mg/L"
                               f"（达标风险概率约 {risk_pct:.0f}%）")
            else:
                st.info("该指标无强制排放标准，仅作负荷趋势预测")
            # 进水负荷冲击的机理联动提示（跨变量，与预测指标无关）
            surge = influent_surge_note(var, s, res["forecast"], season=int(pr["season"]))
            if surge:
                st.warning("📈 " + surge)

            # ---- 模型回测对比 ----
            if res["metrics"]:
                st.subheader("📊 模型回测精度（留一法，尾部验证集）")
                mt = res["metrics"]
                rows = []
                for name in mt:
                    rows.append({
                        "模型": name,
                        "集成权重": f"{res['weights'].get(name, 0) * 100:.0f}%",
                        "RMSE": f"{mt[name]['RMSE']:.3f}",
                        "MAE": f"{mt[name]['MAE']:.3f}",
                        "MAPE(%)": f"{mt[name]['MAPE(%)']:.1f}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("系统按回测 RMSE 逆误差加权自动集成三个模型：权重越高代表该模型在历史验证集上越可靠。")


    # ================= 页面：工艺流程图看板（P&ID 暗色数字孪生） =================
    elif page == "🛰️ 数字孪生驾驶舱":
        # ================= 驾驶舱暗色主题 + 顶部标题栏 =================
        st.markdown(r"""
        <style>
        .stApp{ background: linear-gradient(135deg, #020617 0%, #0b1221 50%, #050a14 100%) !important; }
        .main .block-container{ padding-top:0.5rem; }
        .dash-header{ display:flex; align-items:center; justify-content:space-between;
          background:linear-gradient(90deg,rgba(4,16,31,0.98) 0%,rgba(10,33,56,0.98) 100%);
          border:1px solid rgba(56,189,236,0.35); border-radius:14px; padding:14px 22px; margin-bottom:14px;
          box-shadow:0 0 22px rgba(56,189,236,0.20); }
        .dash-title{ font-size:1.45rem; font-weight:800; color:#e6f6ff; letter-spacing:1px; }
        .dash-sub{ font-size:.78rem; color:#7dd3fc; margin-top:3px; letter-spacing:.5px; }
        .dash-live{ color:#22c55e; font-weight:700; font-size:.95rem; }
        .dash-live .dot{ width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px;
          background:#22c55e; box-shadow:0 0 8px #22c55e; animation:pulse 1.6s infinite; }
        @keyframes pulse{ 0%,100%{opacity:1;} 50%{opacity:.35;} }
        .dash-clock{ color:#7dd3fc; font-family:'Consolas','Courier New',monospace; font-size:1.1rem; margin-top:4px; }
        .cp-panel{ background:linear-gradient(180deg,rgba(9,21,40,0.96),rgba(6,15,29,0.96));
          border:1px solid rgba(56,189,236,0.28); border-radius:12px; padding:12px 14px; margin-bottom:12px;
          box-shadow:0 0 14px rgba(56,189,236,0.08) inset, 0 4px 18px rgba(0,0,0,0.25); transition:all .25s ease; }
        .cp-panel:hover{ border-color:rgba(56,189,236,0.48); box-shadow:0 0 20px rgba(56,189,236,0.15) inset, 0 6px 24px rgba(0,0,0,0.35); transform:translateY(-1px); }
        .cp-title{ font-size:.95rem; font-weight:700; color:#7dd3fc; letter-spacing:.5px;
          border-left:3px solid #38bdf8; padding-left:8px; margin:0 0 10px;
          display:flex; justify-content:space-between; align-items:center; }
        .cp-title small{ color:#5b7da3; font-weight:400; font-size:.68rem; }
        .cp-kv{ display:flex; justify-content:space-between; align-items:center; padding:5px 2px;
          border-bottom:1px dashed rgba(148,163,184,0.15); font-size:.84rem; }
        .cp-kv span{ color:#94a3b8; } .cp-kv b{ color:#e2e8f0; font-weight:600; font-family:'Consolas',monospace; }
        .cp-bar{ height:8px; border-radius:5px; background:rgba(148,163,184,0.18); margin:5px 0 9px; overflow:hidden; }
        .cp-bar>i{ display:block; height:100%; border-radius:5px;
          background:linear-gradient(90deg,#22d3ee,#38bdf8); box-shadow:0 0 8px rgba(56,189,236,0.6); }
        .cp-stat{ display:flex; justify-content:space-between; align-items:center; padding:6px 2px;
          font-size:.84rem; border-bottom:1px dashed rgba(148,163,184,0.12); }
        .cp-stat span{ color:#cbd5e1; } .cp-stat b{ color:#e2e8f0; font-weight:600; }
        .alarm{ display:flex; gap:8px; align-items:flex-start; padding:7px 8px; margin-bottom:7px; border-radius:8px;
          font-size:.8rem; line-height:1.35; }
        .alarm.ok{ background:rgba(34,197,94,0.10); border:1px solid rgba(34,197,94,0.30); color:#bbf7d0; }
        .alarm.warn{ background:rgba(234,179,8,0.10); border:1px solid rgba(234,179,8,0.35); color:#fde68a; }
        .alarm.err{ background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.40); color:#fecaca; }
        .kpi-strip{ display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; }
        .kpi-box{ flex:1 1 0; min-width:128px; background:linear-gradient(180deg,rgba(9,21,40,0.96),rgba(6,15,29,0.96));
          border:1px solid rgba(56,189,236,0.28); border-radius:12px; padding:14px 10px; text-align:center;
          box-shadow:0 0 14px rgba(56,189,236,0.08) inset; transition:all .25s ease; }
        .kpi-box:hover{ border-color:rgba(56,189,236,0.48); transform:translateY(-2px); }
        .kpi-box .v{ font-size:1.45rem; font-weight:800; color:#38bdf8; font-family:'Consolas',monospace; }
        .kpi-box .l{ font-size:.76rem; color:#94a3b8; margin-top:5px; }
        .kpi-box .s{ font-size:.7rem; color:#22c55e; margin-top:2px; }
        .unit-card{ background:linear-gradient(180deg,rgba(9,21,40,0.92),rgba(6,15,29,0.92));
          border:1px solid rgba(56,189,236,0.18); border-radius:12px; padding:13px 12px; margin-bottom:10px;
          box-shadow:0 2px 10px rgba(0,0,0,0.18); position:relative; overflow:hidden; min-height:118px;
          transition:all .25s ease; }
        .unit-card:hover{ border-color:rgba(56,189,236,0.45); box-shadow:0 0 18px rgba(56,189,236,0.12); transform:translateY(-2px); }
        .marquee-wrap{ background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.28); border-radius:10px;
          padding:9px 14px; overflow:hidden; white-space:nowrap; margin-bottom:12px; }
        .marquee-wrap.ok{ background:rgba(34,197,94,0.08); border-color:rgba(34,197,94,0.28); }
        .marquee{ display:inline-block; animation:marquee 18s linear infinite; color:#fecaca; font-size:.82rem; }
        .marquee-wrap.ok .marquee{ color:#bbf7d0; }
        @keyframes marquee{ 0%{transform:translateX(100%);} 100%{transform:translateX(-100%);} }
        </style>
        """, unsafe_allow_html=True)
        _now_str = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
        <div class="dash-header">
          <div>
            <div class="dash-title">🛰️ 五段 Bardenpho 污水厂 · <span style="color:#38bdf8;">数字孪生驾驶舱</span></div>
            <div class="dash-sub">DIGITAL TWIN OPERATION COCKPIT · 全流程智能监测与预警</div>
          </div>
          <div style="text-align:right;">
            <div class="dash-live"><span class="dot"></span>系统在线 LIVE</div>
            <div class="dash-clock">{_now_str}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 实时微波动：改为「客户端 JS 抖动」——服务端只渲染稳定基准值，不再整页自动重跑。
        # 浏览器端 setInterval 每 1.5s 对带 class="jit" 的数字加 ±3% 平滑噪声（慢漂移+随机），
        # 观感与原一致，但彻底消除整页重跑带来的残影/灰屏/切换卡顿（线上零网络往返）。
        import math, time, random as _rd
        def _jit(v, rel=0.03):
            # 服务端逻辑值取稳定基准，不再抖动（score/告警/表盘/趋势均基于真实值，更专业稳定）。
            return float(v)

        def _j(v, dec):
            # 生成带基准值的标记 span，供客户端 JS 抖动。v 为数值，dec 为小数位数。
            return f'<span class="jit" data-base="{v:.10f}" data-dec="{dec}">{v:.{dec}f}</span>'

        # 客户端 JS 抖动脚本：在不可见 iframe 内 setInterval 更新父页面 .jit 数字（±3%）与顶部时钟。
        if _st_html is not None:
            _st_html(
                """
<script>
(function(){
  var rel = 0.03;
  function p2(n){ return (n<10 ? '0':'') + n; }
  setInterval(function(){
    try {
      var d = window.parent.document;
      var t = Date.now()/1000;
      var slow = 0.5 * rel * Math.sin(t/30.0);
      // 同一基准值只算一次抖动，保证多个位置的同源数字一致
      // （顶部KPI↔左面板进水/出水/流量↔底部KPI；左侧去除率卡片↔plotly仪表盘；顶部跨膜压差↔UF仪表盘）
      var jcache = {};
      function jval(baseStr){
        if (jcache[baseStr] === undefined) {
          var b = parseFloat(baseStr);
          if (isNaN(b)) return null;
          var noise = rel * (Math.random()*2 - 1) * 0.5;
          var v = b * (1 + slow + noise);
          // 百分比类（去除率/达标率/运行指数等 base∈[50,100]）上限锁 100，避免越界
          if (b >= 50 && b <= 100) v = Math.min(100, v);
          jcache[baseStr] = v;
        }
        return jcache[baseStr];
      }
      var els = d.querySelectorAll('.jit');
      for (var i=0; i<els.length; i++){
        var el = els[i];
        var baseStr = el.getAttribute('data-base');
        var dec = parseInt(el.getAttribute('data-dec') || '0', 10);
        var v = jval(baseStr);
        if (v === null) continue;
        el.textContent = v.toFixed(dec);
      }
      var clk = d.querySelector('.dash-clock');
      if (clk){
        var now = new Date();
        clk.textContent = now.getFullYear()+'-'+p2(now.getMonth()+1)+'-'+p2(now.getDate())
          +' '+p2(now.getHours())+':'+p2(now.getMinutes())+':'+p2(now.getSeconds());
      }
      // plotly 仪表盘：复用 jcache 与对应 .jit 卡片保持一致；综合指数trace[0]不动
      var gb = d.getElementById('gauge-bases');
      if (gb && window.parent.Plotly) {
        var bC = jval(gb.getAttribute('data-cod-rem'));
        var bT = jval(gb.getAttribute('data-tn-rem'));
        var bM = jval(gb.getAttribute('data-tmp'));
        if (bC !== null && bT !== null && bM !== null) {
          var plots = d.querySelectorAll('.js-plotly-plot');
          for (var pi=0; pi<plots.length; pi++) {
            var gd = plots[pi];
            if (gd && gd._fullData) {
              var hasInd = false;
              for (var k=0; k<gd._fullData.length; k++){
                if (gd._fullData[k].type === 'indicator'){ hasInd = true; break; }
              }
              if (hasInd) {
                try {
                  window.parent.Plotly.restyle(gd, {value: [bC, bT, bM]}, [1,2,3]);
                } catch(e){}
                break;
              }
            }
          }
        }
      }
    } catch(e){}
  }, 1500);
})();
</script>
""",
                height=0, width=0, scrolling=False)

        # 优先使用真实/演示数据；若不存在则回退到基础参数。
        # 演示数据静态，整段缓存到 session_state，避免驾驶舱每 2s 自动刷新都重复读盘
        # （线上每次刷新都走网络往返，重复 I/O 会被明显放大）。
        if "dash_flow_df" not in st.session_state:
            if os.path.exists(SAMPLE_CSV):
                st.session_state["dash_flow_df"] = pd.read_csv(SAMPLE_CSV)
            else:
                st.session_state["dash_flow_df"] = _build_sample_df()
        df_flow = st.session_state["dash_flow_df"]
        # 两种来源统一把"时间"列规整为 datetime，避免下游 .dt 访问报错（趋势图）
        df_flow["时间"] = pd.to_datetime(df_flow["时间"], errors="coerce")
        latest = df_flow.iloc[-1]

        bp = st.session_state.base_params
        Q_avg = _jit(latest.get("进水流量(m3/h)", bp.get("Q_actual", 20000) / 24.0))
        if pd.isna(Q_avg):
            Q_avg = bp.get("Q_actual", 20000) / 24.0
        Q_day = Q_avg * 24.0

        # 内回流比例（用于单元卡片）
        r1_pct = 150.0
        bio = get_compute_result("bio_result")
        if bio:
            r1_pct = max(bio.get("min_R1", 150.0), 100.0)

        # ---- 单元状态判定 ----
        def _unit_status(name):
            if name == "进水":
                return "#38BDF8", "正常"
            if name == "UF 膜池":
                tmp = latest.get("跨膜压差(kPa)", 12.0)
                if tmp >= 35: return "#EF4444", "报警"
                if tmp >= 25: return "#EAB308", "偏高"
                return "#22C55E", "正常"
            if name == "出水":
                cod = latest.get("出水COD(mg/L)", 0); tn = latest.get("出水TN(mg/L)", 0)
                nh3 = latest.get("出水NH3-N(mg/L)", 0); tp = latest.get("出水TP(mg/L)", 0)
                ok = (cod <= 30) and (tn <= 15) and (nh3 <= 1.5) and (tp <= 0.3)
                return ("#22C55E", "达标") if ok else ("#EF4444", "超标")
            return "#22C55E", "正常"

        # 综合运行指数
        cod_in = _jit(latest.get("进水COD(mg/L)", 0))
        nh3_in = _jit(latest.get("进水NH3-N(mg/L)", 0))
        tn_in = _jit(latest.get("进水TN(mg/L)", 0))
        tp_in = _jit(latest.get("进水TP(mg/L)", 0))
        cod_out = _jit(latest.get("出水COD(mg/L)", 15.0))
        tn_out = _jit(latest.get("出水TN(mg/L)", 7.5))
        nh3_out = _jit(latest.get("出水NH3-N(mg/L)", 0.45))
        tp_out = _jit(latest.get("出水TP(mg/L)", 0.15))
        tmp = _jit(8.48)
        score = 100
        if cod_out > 30: score -= 15
        if tn_out > 15: score -= 15
        if nh3_out > 1.5: score -= 10
        if tp_out > 0.3: score -= 10
        if tmp >= 35: score -= 25
        elif tmp >= 25: score -= 10
        score = max(score, 0)
        score_text = "优" if score >= 90 else "良" if score >= 75 else "一般" if score >= 60 else "差"
        score_color = "#22C55E" if score >= 90 else "#38BDF8" if score >= 75 else "#EAB308" if score >= 60 else "#EF4444"

        # ================= 三列驾驶舱布局（图2式：中央主视图 + 左/右面板 + 底部KPI） =================
        def _rem(inv, outv):
            return max((1 - outv / max(inv, 1e-9)) * 100, 0.0)

        def _ok(v, lim):
            return v <= lim

        _run_days = 42

        _lcol, _ccol, _rcol = st.columns([1.0, 2.5, 1.0])

        # ---------- 中央：核心指标仪表盘 + 实时趋势 + AI 诊断 ----------
        with _ccol:
            # ---- 顶部 KPI 条（6 个关键数字） ----
            _达标率 = sum([cod_out <= 30, tn_out <= 15, nh3_out <= 1.5, tp_out <= 0.3]) / 4 * 100
            _kpi_items = [
                ("处理水量", _j(Q_avg, 1), "m³/h", "#38bdf8"),
                ("进水 COD", _j(cod_in, 0), "mg/L", "#a78bfa"),
                ("出水 COD", _j(cod_out, 2), "mg/L", "#22c55e" if cod_out <= 30 else "#ef4444"),
                ("跨膜压差", _j(tmp, 2), "kPa", "#f472b6"),
                ("TN 去除率", _j(_rem(tn_in, tn_out), 1), "%", "#22d3ee"),
                ("运行天数", f"{_run_days}", "d", "#f472b6"),
            ]
            _kpi_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">'
            for _kl, _kv, _ku, _kc in _kpi_items:
                _kpi_html += f'''<div style="flex:1 1 0;min-width:110px;background:linear-gradient(180deg,rgba(9,21,40,0.96),rgba(6,15,29,0.96));border:1px solid rgba(56,189,236,0.22);border-radius:12px;padding:12px 8px;text-align:center;box-shadow:0 0 12px rgba(56,189,236,0.06) inset;">
                  <div style="font-size:.72rem;color:#94a3b8;margin-bottom:5px;">{_kl}</div>
                  <div style="font-size:1.35rem;font-weight:800;color:{_kc};font-family:'Consolas',monospace;text-shadow:0 0 10px {_kc}44;">{_kv}</div>
                  <div style="font-size:.68rem;color:#64748b;margin-top:2px;">{_ku}</div>
                </div>'''
            _kpi_html += '</div>'
            st.markdown(_kpi_html, unsafe_allow_html=True)

            # 告警诊断消息
            _alarm_msgs = []
            _diag_actions = []
            if cod_out > 30:
                _alarm_msgs.append(f"出水 COD 超标：{cod_out:.2f} mg/L（限值 30）")
                _diag_actions.append("排查进水有机负荷冲击，提高曝气量或延长 SRT")
            if tn_out > 15:
                _alarm_msgs.append(f"出水 TN 超标：{tn_out:.2f} mg/L（限值 15）")
                _diag_actions.append("提高内回流比 R1，必要时投加外碳源")
            if nh3_out > 1.5:
                _alarm_msgs.append(f"出水 NH₃-N 超标：{nh3_out:.3f} mg/L（限值 1.5）")
                _diag_actions.append("提高好氧段 DO，检查硝化菌活性")
            if tp_out > 0.3:
                _alarm_msgs.append(f"出水 TP 超标：{tp_out:.3f} mg/L（限值 0.3）")
                _diag_actions.append("增加除磷药剂投加量，优化污泥龄")
            if tmp >= 35:
                _alarm_msgs.append(f"UF 跨膜压差报警：{tmp:.1f} kPa（限值 35）")
                _diag_actions.append("立即执行 CEB/CIP 清洗，检查膜完整性")
            elif tmp >= 25:
                _alarm_msgs.append(f"UF 跨膜压差偏高：{tmp:.1f} kPa，建议关注")
                _diag_actions.append("缩短维护性清洗周期，关注进水浊度")
            if not _alarm_msgs:
                _alarm_msgs.append("✅ 各单元运行正常，出水稳定达标")
                _diag_actions.append("维持当前运行工况，继续执行周期性巡检")
            _marquee_ok = "" if _alarm_msgs[0].startswith("✅") else "ok"
            _marquee_text = "  ★  ".join(_alarm_msgs) + "  ★  "

            st.markdown(f"""
            <div style="background:linear-gradient(90deg,rgba(9,21,40,0.98),rgba(6,15,29,0.96));border:1px solid rgba(56,189,236,0.30);border-radius:14px;padding:16px 20px;margin-bottom:12px;box-shadow:0 0 20px rgba(56,189,236,0.10) inset,0 4px 20px rgba(0,0,0,0.25);">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:20px;">
                <div style="min-width:160px;">
                  <div style="color:#94a3b8;font-size:.78rem;letter-spacing:1px;">综合运行指数</div>
                  <div style="font-size:2.6rem;font-weight:800;color:{score_color};font-family:'Consolas',monospace;text-shadow:0 0 18px {score_color}66;">{score}</div>
                  <div style="font-size:.72rem;color:{score_color};font-weight:600;">{score_text}</div>
                </div>
                <div style="flex:1;">
                  <div style="display:flex;justify-content:space-between;font-size:.72rem;color:#94a3b8;margin-bottom:5px;">
                    <span>运行健康度</span><span>{score}/100</span>
                  </div>
                  <div style="height:10px;border-radius:6px;background:rgba(148,163,184,0.15);overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,0.35);">
                    <div style="width:{score}%;height:100%;border-radius:6px;background:linear-gradient(90deg,{score_color},#7dd3fc);box-shadow:0 0 12px {score_color}88;"></div>
                  </div>
                  <div class="marquee-wrap {_marquee_ok}" style="margin-top:10px;">
                    <div class="marquee">{_marquee_text}</div>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ---- 核心指标仪表盘（2x2 圆形表盘） ----
            if go and make_subplots:
                _cod_rem = _rem(cod_in, cod_out)
                _tn_rem = _rem(tn_in, tn_out)
                _nh3_rem = _rem(nh3_in, nh3_out)

                # 仪表盘基准值（供客户端 JS 抖动 3 个仪表，综合运行指数保持 100 静态）
                st.markdown(
                    f'<div id="gauge-bases" style="display:none" '
                    f'data-cod-rem="{_cod_rem:.10f}" data-tn-rem="{_tn_rem:.10f}" data-tmp="{tmp:.10f}"></div>',
                    unsafe_allow_html=True)

                fig_gauges = make_subplots(
                    rows=2, cols=2,
                    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                           [{'type': 'indicator'}, {'type': 'indicator'}]],
                    vertical_spacing=0.30, horizontal_spacing=0.22
                )
                fig_gauges.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={"text": "综合运行指数", "font": {"size": 13, "color": "#e2e8f0"}},
                    number={"font": {"size": 28, "color": score_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickwidth": 1, "tickfont": {"size": 9, "color": "#94a3b8"}},
                        "bar": {"color": score_color, "thickness": 0.75},
                        "bgcolor": "rgba(15,23,42,0.8)",
                        "bordercolor": "rgba(56,189,236,0.3)",
                        "steps": [
                            {"range": [0, 60], "color": "rgba(239,68,68,0.2)"},
                            {"range": [60, 75], "color": "rgba(234,179,8,0.2)"},
                            {"range": [75, 90], "color": "rgba(56,189,236,0.2)"},
                            {"range": [90, 100], "color": "rgba(34,197,94,0.2)"},
                        ],
                        "threshold": {"line": {"color": "#f472b6", "width": 2}, "value": 75},
                    }
                ), row=1, col=1)
                fig_gauges.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=_cod_rem,
                    title={"text": "COD 去除率", "font": {"size": 13, "color": "#e2e8f0"}},
                    number={"font": {"size": 24, "color": "#22d3ee"}, "suffix": "%", "valueformat": ".1f"},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"size": 9, "color": "#94a3b8"}},
                        "bar": {"color": "#22d3ee", "thickness": 0.75},
                        "bgcolor": "rgba(15,23,42,0.8)",
                        "bordercolor": "rgba(56,189,236,0.3)",
                        "steps": [
                            {"range": [0, 60], "color": "rgba(239,68,68,0.15)"},
                            {"range": [60, 80], "color": "rgba(234,179,8,0.15)"},
                            {"range": [80, 100], "color": "rgba(34,197,94,0.15)"},
                        ],
                    }
                ), row=1, col=2)
                fig_gauges.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=_tn_rem,
                    title={"text": "TN 去除率", "font": {"size": 13, "color": "#e2e8f0"}},
                    number={"font": {"size": 24, "color": "#a78bfa"}, "suffix": "%", "valueformat": ".1f"},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"size": 9, "color": "#94a3b8"}},
                        "bar": {"color": "#a78bfa", "thickness": 0.75},
                        "bgcolor": "rgba(15,23,42,0.8)",
                        "bordercolor": "rgba(56,189,236,0.3)",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                            {"range": [50, 70], "color": "rgba(234,179,8,0.15)"},
                            {"range": [70, 100], "color": "rgba(34,197,94,0.15)"},
                        ],
                    }
                ), row=2, col=1)
                fig_gauges.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=tmp,
                    title={"text": "UF 跨膜压差", "font": {"size": 13, "color": "#e2e8f0"}},
                    number={"font": {"size": 24, "color": "#f472b6"}, "suffix": " kPa", "valueformat": ".2f"},
                    gauge={
                        "axis": {"range": [0, 50], "tickcolor": "#64748b", "tickfont": {"size": 9, "color": "#94a3b8"}},
                        "bar": {"color": "#f472b6", "thickness": 0.75},
                        "bgcolor": "rgba(15,23,42,0.8)",
                        "bordercolor": "rgba(56,189,236,0.3)",
                        "steps": [
                            {"range": [0, 15], "color": "rgba(34,197,94,0.15)"},
                            {"range": [15, 25], "color": "rgba(56,189,236,0.15)"},
                            {"range": [25, 35], "color": "rgba(234,179,8,0.2)"},
                            {"range": [35, 50], "color": "rgba(239,68,68,0.25)"},
                        ],
                        "threshold": {"line": {"color": "#ef4444", "width": 2}, "value": 35},
                    }
                ), row=2, col=2)

                fig_gauges.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e2e8f0", "family": "Microsoft YaHei"},
                    margin=dict(l=40, r=40, t=60, b=20),
                    height=380,
                    showlegend=False
                )
                st.plotly_chart(fig_gauges, use_container_width=True)
            else:
                st.warning("缺少 plotly/plotly.subplots，无法显示仪表盘。")

            # ---- AI 智能诊断结论 ----
            _diag_color = "#22c55e" if score >= 90 else "#38bdf8" if score >= 75 else "#eab308" if score >= 60 else "#ef4444"
            _diag_icon = "✅" if score >= 90 else "ℹ️" if score >= 75 else "⚠️" if score >= 60 else "⛔"
            _diag_title = f"{_diag_icon} AI 智能诊断结论"
            _diag_body = "；".join(_diag_actions)
            st.markdown(f"""
            <div style="background:linear-gradient(90deg,rgba(9,21,40,0.95),rgba(6,15,29,0.95));border:1px solid {_diag_color};border-radius:14px;padding:14px 18px;margin-top:12px;box-shadow:0 0 18px {_diag_color}33;">
              <div style="font-size:.95rem;font-weight:700;color:{_diag_color};margin-bottom:6px;">{_diag_title}</div>
              <div style="font-size:.82rem;color:#e2e8f0;line-height:1.5;">{_diag_body}</div>
            </div>
            """, unsafe_allow_html=True)

            # 底部工艺流向提示
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;gap:6px;margin-top:14px;color:#64748b;font-size:.74rem;">
              <span>进水</span><span>→</span><span>厌氧</span><span>→</span><span>缺氧1</span><span>→</span><span>好氧1</span><span>↔</span><span>缺氧2</span><span>→</span><span>好氧2</span><span>→</span><span>二沉</span><span>→</span><span>UF</span><span>→</span><span>出水</span>
            </div>
            """, unsafe_allow_html=True)

        # ---------- 左侧：进出水水质 + 去除率 ----------
        with _lcol:
            cod_in = _jit(latest.get("进水COD(mg/L)", 0))
            nh3_in = _jit(latest.get("进水NH3-N(mg/L)", 0))
            tn_in = _jit(latest.get("进水TN(mg/L)", 0))
            tp_in = _jit(latest.get("进水TP(mg/L)", 0))
            cod_out = _jit(latest.get("出水COD(mg/L)", 15.0))
            nh3_out = _jit(latest.get("出水NH3-N(mg/L)", 0.45))
            tn_out = _jit(latest.get("出水TN(mg/L)", 7.5))
            tp_out = _jit(latest.get("出水TP(mg/L)", 0.15))

            _out_rows = [
                ("COD (mg/L)", cod_out, 30.0, _ok(cod_out, 30)),
                ("NH₃-N (mg/L)", nh3_out, 1.5, _ok(nh3_out, 1.5)),
                ("TN (mg/L)", tn_out, 15.0, _ok(tn_out, 15.0)),
                ("TP (mg/L)", tp_out, 0.3, _ok(tp_out, 0.3)),
            ]
            _in_html = (
                '<div class="cp-panel"><div class="cp-title">🟦 进水实时水质 '
                '<small>LATEST</small></div>'
                f'<div class="cp-kv"><span>COD</span><b>{_j(cod_in, 0)}</b></div>'
                f'<div class="cp-kv"><span>NH₃-N</span><b>{_j(nh3_in, 1)}</b></div>'
                f'<div class="cp-kv"><span>TN</span><b>{_j(tn_in, 0)}</b></div>'
                f'<div class="cp-kv"><span>TP</span><b>{_j(tp_in, 2)}</b></div>'
                f'<div class="cp-kv"><span>流量</span><b>{_j(Q_avg, 1)} m³/h</b></div>'
                '</div>'
            )
            st.markdown(_in_html, unsafe_allow_html=True)

            _out_html = '<div class="cp-panel"><div class="cp-title">🟩 出水实时水质 <small>准四类</small></div>'
            for _nm, _v, _lim, _isok in _out_rows:
                _col = "#22c55e" if _isok else "#ef4444"
                _out_html += (
                    f'<div class="cp-kv"><span>{_nm}</span>'
                    f'<b style="color:{_col}">{_j(_v, 2)} / {_lim:.2f}</b></div>'
                )
            _out_html += '</div>'
            st.markdown(_out_html, unsafe_allow_html=True)

            _rem_rows = [
                ("COD", _rem(cod_in, cod_out)),
                ("NH₃-N", _rem(nh3_in, nh3_out)),
                ("TN", _rem(tn_in, tn_out)),
                ("TP", _rem(tp_in, tp_out)),
            ]
            _rem_html = '<div class="cp-panel"><div class="cp-title">📊 污染物去除率 <small>REMOVAL</small></div>'
            for _nm, _p in _rem_rows:
                _w = min(_p, 100)
                _rem_html += (
                    f'<div class="cp-kv"><span>{_nm}</span><b>{_j(_p, 1)}%</b></div>'
                    f'<div class="cp-bar"><i style="width:{_w:.0f}%"></i></div>'
                )
            _rem_html += '</div>'
            st.markdown(_rem_html, unsafe_allow_html=True)

        # ---------- 右侧：关键工艺参数 + 设备状态 + 告警 ----------
        with _rcol:
            # 关键工艺参数（2x3 卡片矩阵，与整体暗色霓虹风格统一）
            _srt = (bio or {}).get("srt", 15.0)
            _mlss = bp.get("MLSS", 3500.0)
            _fm = (cod_in * Q_avg * 24 / 1000.0) / (_mlss * sum([
                bp.get("V_ana", 0), bp.get("V_anox1", 0), bp.get("V_aero1", 0),
                bp.get("V_anox2", 0), bp.get("V_aero2", 0)
            ]) / 1000.0) if _mlss > 0 else 0.0
            _param_rows = [
                ("DO 设定", "2.0–2.5", "mg/L", "#22d3ee"),
                ("MLSS", f"{_mlss:.0f}", "mg/L", "#a78bfa"),
                ("内回流比 R1", f"{r1_pct:.0f}", "%", "#f472b6"),
                ("污泥回流比 R2", "50–100", "%", "#eab308"),
                ("SRT 污泥龄", f"{_srt:.1f}", "d", "#38bdf8"),
                ("F/M 负荷", _j(_fm, 2), "kgCOD/(kg·d)", "#22c55e"),
            ]
            _param_html = '<div class="cp-panel"><div class="cp-title">⚙️ 关键工艺参数 <small>KEY PARAMETERS</small></div><div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
            for _pn, _pv, _pu, _pc in _param_rows:
                _param_html += f'''<div style="background:linear-gradient(180deg,rgba(9,21,40,0.92),rgba(6,15,29,0.92));border:1px solid rgba(56,189,236,0.18);border-radius:10px;padding:12px 8px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.18);">
                  <div style="font-size:.72rem;color:#94a3b8;margin-bottom:4px;">{_pn}</div>
                  <div style="font-size:1.15rem;font-weight:800;color:{_pc};font-family:'Consolas',monospace;text-shadow:0 0 8px {_pc}44;">{_pv}</div>
                  <div style="font-size:.65rem;color:#64748b;margin-top:2px;">{_pu}</div>
                </div>'''
            _param_html += '</div></div>'
            st.markdown(_param_html, unsafe_allow_html=True)

            # 设备运行状态
            _uf_st = ("运行" if tmp < 25 else "预警" if tmp < 35 else "报警")
            _uf_col = ("#22c55e" if tmp < 25 else "#eab308" if tmp < 35 else "#ef4444")
            _dev_html = (
                '<div class="cp-panel"><div class="cp-title">⚙️ 设备运行状态 <small>RUNTIME</small></div>'
                '<div class="cp-stat"><span>🔵 进水泵组</span><b style="color:#22c55e">运行</b></div>'
                '<div class="cp-stat"><span>🟢 鼓风机</span><b style="color:#22c55e">运行</b></div>'
                '<div class="cp-stat"><span>🟣 污泥回流泵</span><b style="color:#22c55e">运行</b></div>'
                '<div class="cp-stat"><span>🟠 内回流泵</span><b style="color:#22c55e">运行</b></div>'
                '<div class="cp-stat"><span>🟡 加药泵</span><b style="color:#22c55e">运行</b></div>'
                f'<div class="cp-stat"><span>🟪 UF 膜组</span><b style="color:{_uf_col}">{_uf_st}</b></div>'
                '</div>'
            )
            st.markdown(_dev_html, unsafe_allow_html=True)

            # 实时告警
            _alarms = []
            if tmp >= 35:
                _alarms.append(("err", "⛔ 跨膜压差报警（≥35 kPa），建议立即 CIP 清洗"))
            elif tmp >= 25:
                _alarms.append(("warn", "⚠️ 跨膜压差偏高（≥25 kPa），建议缩短维护性清洗周期"))
            if tp_out > 0.3:
                _alarms.append(("err", "⛔ 出水 TP 超标（>0.3 mg/L），检查除磷加药"))
            if not (cod_out <= 30 and nh3_out <= 1.5 and tn_out <= 15):
                _alarms.append(("warn", "⚠️ 出水主要指标接近限值，关注工艺稳定性"))
            if not _alarms:
                _alarms.append(("ok", "✅ 各单元运行正常，出水稳定达标"))
            _al_html = '<div class="cp-panel"><div class="cp-title">🔔 实时告警 <small>ALARM</small></div>'
            for _lv, _msg in _alarms:
                _al_html += f'<div class="alarm {_lv}">{_msg}</div>'
            _al_html += '</div>'
            st.markdown(_al_html, unsafe_allow_html=True)


        # ================= 底部 KPI 指标条 =================
        _cod_rem = _rem(cod_in, cod_out)
        _tn_rem = _rem(tn_in, tn_out)
        _nh3_rem = _rem(nh3_in, nh3_out)
        _compliance = (
            df_flow.tail(24).apply(
                lambda r: (r["出水COD(mg/L)"] <= 30 and r["出水TN(mg/L)"] <= 15 and
                           r["出水NH3-N(mg/L)"] <= 1.5 and r["出水TP(mg/L)"] <= 0.3), axis=1
            ).mean() * 100
        ) if len(df_flow) >= 24 else 100.0
        _flow_latest = latest.get("进水流量(m3/h)", 833.0)
        _energy_latest = latest.get("电耗(kWh/h)", 0)
        _chem_latest = latest.get("药耗(kg/h)", 0)
        _kwh_m3 = (_energy_latest / _flow_latest) if _flow_latest > 0 else 0.0
        _kg_m3 = (_chem_latest / _flow_latest) if _flow_latest > 0 else 0.0
        _kpi_html = (
            '<div class="kpi-strip">'
            f'<div class="kpi-box"><div class="v">{_j(Q_avg, 1)}</div><div class="l">处理水量 (m³/h)</div><div class="s">DESIGN 833.3</div></div>'
            f'<div class="kpi-box"><div class="v">{_compliance:.0f}%</div><div class="l">综合达标率</div><div class="s">近24h</div></div>'
            f'<div class="kpi-box"><div class="v">{_j(_cod_rem, 1)}%</div><div class="l">COD 去除率</div></div>'
            f'<div class="kpi-box"><div class="v">{_j(_tn_rem, 1)}%</div><div class="l">TN 去除率</div></div>'
            f'<div class="kpi-box"><div class="v">{_j(_nh3_rem, 1)}%</div><div class="l">NH₃-N 去除率</div></div>'
            f'<div class="kpi-box"><div class="v">{_kwh_m3:.3f}</div><div class="l">吨水电耗 (kWh/m³)</div></div>'
            f'<div class="kpi-box"><div class="v">{_kg_m3:.3f}</div><div class="l">吨水药耗 (kg/m³)</div></div>'
            f'<div class="kpi-box"><div class="v">{_j(latest.get("UF出水浊度(NTU)", 0.04), 3)}</div><div class="l">UF 出水浊度 (NTU)</div></div>'
            f'<div class="kpi-box"><div class="v">{_run_days}</div><div class="l">连续运行 (天)</div></div>'
            '</div>'
        )
        st.markdown(_kpi_html, unsafe_allow_html=True)
        st.caption("💡 本驾驶舱使用演示数据绘制；接入真实 SCADA / 运行报表后，各指标、状态光环与告警将随实时数据自动刷新。")

    # ================= 页面12：AI 工艺助手 =================
    elif page == "💬 AI 工艺助手":
        st.header("💬 AI 工艺助手（自然语言交互）")
        st.caption("注入当前系统计算结果与知识库作为上下文；侧边栏显示「云端/本地模型已配置」时由大模型实时生成回答，"
                   "否则自动降级为内置规则引擎。当前模式见左侧「AI 模式」状态条。")

        ctx_parts = []
        bp = st.session_state.get("base_params", {})
        bio = get_compute_result("bio_result")
        if bp:
            ctx_parts.append("【基础参数】设计水量 {} m3/d，实际水量 {} m3/d，电价 {} 元/kWh".format(
                bp.get('Q_design'), bp.get('Q_actual'), bp.get('elec_price')))
        if bio:
            ctx_parts.append("【生化结果】理论出水TN {:.2f} mg/L，碳源缺口 {:.1f} mg/L，化学除磷需求 {:.2f} mg/L，SRT {:.1f} d，C/N {:.1f}".format(
                bio.get('tn_theory', 0), bio.get('carbon_deficit', 0), bio.get('tp_need_chem', 0),
                bio.get('srt', 0), bio.get('cn_ratio', 0)))
        ctx = "\n".join(ctx_parts) if ctx_parts else "（尚无可用的计算结果，请先在相关页面完成计算）"

        # ===== 页面吉祥物：小鹏（提前渲染，避免流式输出期间消失）=====
        mascot_path = os.path.join(SCRIPT_DIR, "mascot.png")
        if os.path.exists(mascot_path):
            import random
            _greetings = [
                "👋 你好！我是小鹏，<br>你的 AI 工艺助手～",
                "💡 试试问我：「当前总氮偏高怎么处理？」",
                "🔧 我可以帮你做工艺诊断、预测预警和运行优化哦～",
                "🌊 五段 Bardenpho 工艺相关问题，我都很熟悉！",
                "📉 出水 COD 异常？<br>我可以帮你分析原因和调控措施。",
                "⚡ 想优化运行成本？问问我关于电耗和药耗的建议。",
                "🔮 我可以基于历史数据预测未来出水水质趋势。"
            ]
            _greeting = random.choice(_greetings)
            # 小鹏图片只在会话首次读取并 base64 编码一次，之后复用；
            # 否则每次切到 AI 助手页都重读 51KB 图片并回传约 69KB 内联串（线上每次都走网络，明显变慢）。
            if "mascot_b64_cache" not in st.session_state:
                with open(mascot_path, "rb") as _f:
                    st.session_state["mascot_b64_cache"] = base64.b64encode(_f.read()).decode()
            _mascot_b64 = st.session_state["mascot_b64_cache"]
            st.markdown(f"""
            <style>
            .mascot-d {{
                position: fixed;
                right: 50px;
                bottom: 190px;
                z-index: 9999;
            }}
            .mascot-d summary {{
                list-style: none;
                cursor: pointer;
                display: block;
                width: 110px;
                height: 110px;
            }}
            .mascot-d summary::-webkit-details-marker {{ display: none; }}
            .mascot-xiao {{
                width: 100%;
                height: auto;
                display: block;
                filter: drop-shadow(0 6px 12px rgba(0,0,0,0.28));
                animation: mascot-bob 3s ease-in-out infinite;
                transition: transform 0.2s ease;
            }}
            .mascot-d summary:hover .mascot-xiao {{ transform: scale(1.08); }}
            @keyframes mascot-bob {{
                0%, 100% {{ transform: translateY(0) rotate(-1deg); }}
                50% {{ transform: translateY(-12px) rotate(1deg); }}
            }}
            .mascot-tip {{
                position: fixed;
                right: 50px;
                bottom: 315px;
                z-index: 9999;
                background: rgba(15,23,42,0.92);
                color: #e2e8f0;
                padding: 10px 16px;
                border-radius: 12px;
                font-size: 13px;
                max-width: 200px;
                line-height: 1.5;
                box-shadow: 0 4px 14px rgba(0,0,0,0.3);
                opacity: 0;
                transform: translateY(8px);
                transition: opacity 0.3s, transform 0.3s;
                pointer-events: none;
                text-align: center;
            }}
            .mascot-d[open] .mascot-tip {{
                opacity: 1;
                transform: translateY(0);
            }}
            </style>
            <details class="mascot-d">
                <summary>
                    <img src="data:image/png;base64,{_mascot_b64}" class="mascot-xiao" title="我是小鹏，你的 AI 工艺助手">
                </summary>
                <div class="mascot-tip">{_greeting}</div>
            </details>
            """, unsafe_allow_html=True)

        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = []
        for m in st.session_state.ai_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if q := st.chat_input("向工艺助手提问，如：当前总氮偏高怎么处理？"):
            st.session_state.ai_messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                reply = st.write_stream(ai_assistant_stream(q, ctx, kb_dir=find_kb_dir()))
            st.session_state.ai_messages.append({"role": "assistant", "content": reply})
        if st.button("清空对话", key="ai_clear"):
            st.session_state.ai_messages = []

