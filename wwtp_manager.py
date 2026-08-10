import numpy as np
import os
import re
try:
    import streamlit as st
except ImportError:
    st = None
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
        }, "【成本建议】可在『AI 工艺优化与诊断』页运行『一键优化投加方案』，在满足出水标准下"
           "最小化药剂成本；曝气能耗可通过优化 DO 设定与精确曝气进一步降低。"),
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
        key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("WWTP_LLM_KEY") or None
        base = st.secrets.get("OPENAI_BASE_URL") or None
        model = st.secrets.get("OPENAI_MODEL") or None
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
    except Exception:
        # 联网/鉴权失败：静默降级到规则引擎（保证演示永不中断）
        pass
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
    """标准正态 CDF 近似（Abramowitz-Stegun 公式），无需 scipy。"""
    return 0.5 * (1.0 + float(np.sign(x)) * np.sqrt(1.0 - np.exp(-2.0 * x * x / np.pi)))


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
    部署时若 sample_wwtp_history.csv 缺失，可即时在内存生成，无需额外文件。"""
    if pd is None:
        raise RuntimeError("需要 pandas 才能生成示例数据")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-06-01 00:00")
    t = pd.date_range(start, periods=hours, freq="h")
    hour = t.hour.to_numpy()
    dow = t.dayofweek.to_numpy()
    day_idx = np.arange(hours)

    diurnal = 0.5 * np.sin(2 * np.pi * (hour - 4) / 24) + 0.5 * np.sin(2 * np.pi * (hour - 16) / 24)
    diurnal = (diurnal - diurnal.min()) / (diurnal.max() - diurnal.min())
    week_factor = np.where(dow >= 5, 0.88, 1.0)
    trend = 1.0 + 0.08 * np.sin(2 * np.pi * day_idx / (24 * 30))

    base_flow = 420.0
    flow = base_flow * (0.55 + 0.55 * diurnal) * week_factor * trend
    flow *= (1 + rng.normal(0, 0.05, hours))
    flow = np.clip(flow, 200, 900)

    def polluant(base, cv, spike_prob, spike_mul):
        val = base * (1 + rng.normal(0, cv, hours))
        spike = rng.random(hours) < spike_prob
        val[spike] *= rng.uniform(spike_mul, spike_mul + 0.6, spike.sum())
        return np.clip(val, base * 0.3, None)

    cod_in = polluant(350, 0.10, 0.04, 1.4)
    nh3_in = polluant(28, 0.10, 0.03, 1.3)
    tn_in = nh3_in + polluant(13, 0.10, 0.02, 1.2)
    tp_in = polluant(5.0, 0.12, 0.03, 1.4)

    def effluent(inf, removal, std, occasional):
        out = inf * (1 - removal) + rng.normal(0, std, hours)
        hit = (inf > np.percentile(inf, 92)) & (rng.random(hours) < occasional)
        out[hit] = out[hit] * rng.uniform(1.3, 1.8, hit.sum())
        return np.clip(out, 0.05, None)

    cod_out = effluent(cod_in, 0.90, 2.5, 0.5)
    nh3_out = effluent(nh3_in, 0.965, 0.3, 0.4)
    tn_out = effluent(tn_in, 0.70, 1.0, 0.5)
    tp_out = effluent(tp_in, 0.93, 0.05, 0.4)

    energy = 760 + 0.45 * flow + rng.normal(0, 35, hours)
    energy = np.clip(energy, 400, None)
    chem = 18 + 0.02 * flow + 6 * (tp_out > 0.45) + rng.normal(0, 2.5, hours)
    chem = np.clip(chem, 5, None)

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
        "电耗(kWh/h)": np.round(energy, 1),
        "药耗(kg/h)": np.round(chem, 1),
    })


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
        page_title="五段Bardenpho污水厂运维管理系统",
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
    header[data-testid="stHeader"]{ display:none !important; }
    .main .block-container{ padding-top:2.2rem; padding-bottom:3rem; }

    /* 标题体系 */
    h1{ font-size:1.7rem; font-weight:700; color:var(--text);
        border-left:5px solid var(--primary); padding-left:14px; margin-bottom:.6rem; }
    h2{ font-size:1.28rem; font-weight:650; color:var(--text); }
    h3{ font-size:1.05rem; font-weight:600; color:var(--text); }
    .stCaption{ color:var(--text2) !important; }
    .stMarkdown p{ color:var(--text2); }

    /* 侧边栏：深蓝渐变 + 白字 */
    [data-testid="stSidebar"]{
      background:linear-gradient(180deg,#0f3043 0%,#0a2233 100%);
      border-right:1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] *{ color:#cbd5e1; }
    [data-testid="stSidebar"] .css-1oe5cao, [data-testid="stSidebar"] h1{ color:#fff !important; }
    [data-testid="stSidebar"] hr{ border-color:rgba(255,255,255,0.12) !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
      gap:6px;
    }
    [data-testid="stSidebar"] [role="radio"]{
      padding:10px 14px; border-radius:9px; transition:.15s;
    }
    [data-testid="stSidebar"] [aria-checked="true"]{
      background:rgba(20,184,166,0.18) !important;
      color:#ffffff !important; font-weight:700;
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
        # —— 登录页专属样式：仅登录页隐藏侧边栏/顶栏，并美化表单 ——
        st.markdown(r"""
        <style>
        [data-testid="stSidebar"]{ display:none !important; }
        header[data-testid="stHeader"]{ display:none !important; }
        #login-scope ~ .element-container [data-testid="stTextInput"]{ max-width:320px; margin:0 auto; }
        #login-scope ~ .element-container [data-testid="stButton"]{ max-width:320px; margin:14px auto 0; }
        #login-scope ~ .element-container [data-testid="stTextInput"] input{
            border-radius:10px; border:1.5px solid #cbd5e1; padding:11px 14px; font-size:1rem;
        }
        #login-scope ~ .element-container [data-testid="stTextInput"] input:focus{
            border-color:#0e7490; box-shadow:0 0 0 3px rgba(14,116,144,0.15);
        }
        .login-card{ max-width:480px; margin:6vh auto 2vh; text-align:center;
            background:rgba(255,255,255,0.92); backdrop-filter:blur(8px);
            border:1px solid rgba(255,255,255,0.6); border-radius:20px;
            padding:40px 34px; box-shadow:0 24px 60px rgba(15,23,42,0.20); }
        .login-logo{ width:74px;height:74px;border-radius:50%;margin:0 auto 18px;
            display:flex;align-items:center;justify-content:center;font-size:34px;
            background:linear-gradient(135deg,#0e7490,#14b8a6);
            box-shadow:0 10px 26px rgba(14,116,144,0.35); }
        .login-title{ font-size:1.55rem;font-weight:800;color:#0f3043;line-height:1.45; }
        .login-en{ margin-top:10px;font-size:.72rem;letter-spacing:2px;color:#64748b;
            text-transform:uppercase; }
        .login-divider{ width:64px;height:3px;border-radius:2px;margin:20px auto;
            background:linear-gradient(90deg,#0e7490,#14b8a6); }
        .login-tip{ color:#475569;font-size:.95rem; }
        </style>
        <div id="login-scope"></div>
        <div class="login-card">
            <div class="login-logo">💧</div>
            <div class="login-title">五段Bardenpho污水厂<br>智能运维管理系统</div>
            <div class="login-en">Five-Stage Bardenpho WWTP O&amp;M Platform</div>
            <div class="login-divider"></div>
            <div class="login-tip">请输入访问密码以进入系统</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1.6, 1])
        with col2:
            input_pwd = st.text_input("访问密码", type="password",
                                      placeholder="请输入访问密码", help="默认密码：123456")
            if st.button("登 录 系 统", type="primary", use_container_width=True):
                # 从平台后台读取正确密码（本地无 secrets.toml 时自动回退默认密码）
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
                "🔮 AI 预测预警",
                "🛠️ AI 工艺优化与诊断",
                "💬 AI 工艺助手",
                "📊 报表导出"
            ]
        )
        st.markdown("---")
        st.caption("工艺路线：厌氧→缺氧1→好氧1→缺氧2→好氧2→二沉池")
        st.caption("内回流：好氧1 → 缺氧1；好氧1自流至缺氧2深度反硝化")
        st.caption("好氧2功能：吹脱氮气 + 防止二沉池反硝化")
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

        with col2:
            st.subheader("二、生化动力学系数")
            Y = num_input("污泥产率系数 Y", value=st.session_state.base_params['Y'], min_value=0.0)
            Kd = num_input("内源衰减系数 Kd (d⁻¹)", value=st.session_state.base_params['Kd'], min_value=0.0)
            nitr_rate = num_input("硝化速率 kgNH3/(kgMLSS·d)", value=st.session_state.base_params['nitr_rate'], min_value=0.0)
            denitr_rate = num_input("反硝化速率 kgNO3/(kgMLSS·d)", value=st.session_state.base_params['denitr_rate'], min_value=0.0)
            mlvss_mlss = num_input("MLVSS / MLSS 比值", value=st.session_state.base_params['mlvss_mlss'], min_value=0.0)
            carbon_cod_eq = num_input("碳源COD当量基准值 (gCOD/g药剂)",
                                            value=st.session_state.base_params['carbon_cod_eq'], min_value=0.0)

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
            cod_in = num_input("进水COD (mg/L)", value=350)
            bod_in = num_input("进水BOD5 (mg/L)", value=180)
            nh3_in = num_input("进水氨氮 (mg/L)", value=28)
            tn_in = num_input("进水总氮 TN (mg/L)", value=40)
            tp_in = num_input("进水总磷 TP (mg/L)", value=5)
        with col2:
            tn_out_target = num_input("出水TN目标 (mg/L)", value=15)
            tp_out_target = num_input("出水TP目标 (mg/L)", value=0.5)
            bod_eff = num_input("实际出水BOD5 (mg/L)", value=10)
            cod_eff = num_input("实际出水COD (mg/L)", value=50)
            nh3_eff = num_input("实际出水氨氮 (mg/L)", value=1.5)
            mlss = num_input("MLSS 混合液浓度 (mg/L)", value=3500)
            R = num_input("污泥回流比 R (%)", value=100) / 100
            R1 = num_input("内回流比 R1 (好氧1→缺氧1, %)", value=200) / 100
            waste_sludge_volume = num_input("每日外排剩余污泥量(m³/d)", value=20.0)

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

                st.info(f"💡 节能提示：曝气系统占总电耗 {res['e_aeration']/res['e_total_day']*100:.1f}%，采用DO变频曝气可节电15%~25%")

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

    # ================= 页面7：报表导出 =================
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
        st.header("🔮 进水负荷与出水水质 AI 预测预警")
        st.caption("多模型集成预测：Holt-Winters + 谐波回归 + 季节朴素，经历史回测逆误差自动加权选优；"
                   "纯算法无需联网。默认载入内置合成测试数据。")

        if not os.path.exists(SAMPLE_CSV):
            st.info("⚙️ 未找到本地示例数据，正在即时生成合成演示数据…")
            df = _build_sample_df()
            try:
                df.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
            except Exception:
                pass
        else:
            df = pd.read_csv(SAMPLE_CSV)
            df["时间"] = pd.to_datetime(df["时间"])
            df = df.set_index("时间").sort_index()
            var_options = {
                "出水COD(mg/L)": 50, "出水NH3-N(mg/L)": 5, "出水TN(mg/L)": 15,
                "出水TP(mg/L)": 0.5, "进水流量(m3/h)": None, "进水COD(mg/L)": None,
                "进水TN(mg/L)": None, "进水TP(mg/L)": None,
            }
            with st.expander("⚙️ 高级设置", expanded=False):
                season = num_input("季节周期（小时，默认日周期=24）", min_value=1, max_value=168,
                                         value=24, step=1,
                                         help="数据呈现的周期性长度；小时级数据通常取 24（日周期）")
            col1, col2 = st.columns(2)
            with col1:
                var = st.selectbox("预测指标", list(var_options.keys()))
            with col2:
                horizon = st.selectbox("预测步长（小时）", [24, 48, 168], index=0)
            if st.button("运行预测", type="primary", key="predict_btn"):
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

                # ---- 指标卡 ----
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("预测均值", f"{fc.mean():.2f}")
                c2.metric("预测峰值", f"{fc.max():.2f}")
                n_warn = int((up > std).sum()) if std is not None else 0
                c3.metric("超标风险时点", f"{n_warn}/{len(fc)}" if std is not None else "—")
                c4.metric("历史异常点", f"{len(res['anomalies'])}")

                # ---- 主预测图 ----
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=idx_hist[-7 * 24:], y=s[-7 * 24:],
                                         name="历史(近7天)", line=dict(color="#0E7490")))
                fig.add_trace(go.Scatter(x=idx_fc, y=up, line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=idx_fc, y=lo, line=dict(width=0), fill="tonexty",
                                         fillcolor="rgba(20,184,166,0.22)", name="95%置信区间"))
                fig.add_trace(go.Scatter(x=idx_fc, y=fc, name="AI集成预测",
                                         line=dict(color="#F59E0B", dash="dot", width=2)))
                if std is not None:
                    fig.add_hline(y=std, line=dict(color="#DC2626", dash="dash"),
                                  annotation_text=f"标准限值 {std}")
                fig.update_layout(title=dict(text=f"{var} 未来 {pr['horizon']} 小时 AI 预测", x=0.5, xanchor="center"),
                                  xaxis_title="时间", yaxis_title=var, template="plotly_white",
                                  legend=dict(orientation="h"))
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

                # ---- 时序分解 ----
                st.subheader("🔍 时序分解（趋势 / 季节 / 残差）")
                fig2 = go.Figure()
                trend = res["trend"]
                fig2.add_trace(go.Scatter(x=idx_hist[-14 * 24:], y=trend[-14 * 24:], name="趋势",
                                          line=dict(color="#7C3AED")))
                fig2.add_trace(go.Scatter(x=idx_hist, y=res["seasonal"], name="季节分量",
                                          line=dict(color="#0891B2", width=1)))
                fig2.update_layout(title=dict(text="趋势与季节分量（近14天）", x=0.5, xanchor="center"),
                                   xaxis_title="时间", yaxis_title=var, template="plotly_white",
                                   legend=dict(orientation="h"))
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = go.Figure()
                resid = res["resid"]
                fig3.add_trace(go.Scatter(x=idx_hist, y=resid, name="残差",
                                          line=dict(color="#64748B", width=1)))
                if len(res["anomalies"]):
                    an = res["anomalies"]
                    fig3.add_trace(go.Scatter(x=idx_hist[an], y=resid[an], mode="markers",
                                              name="异常点(|残差|>3σ)",
                                              marker=dict(color="#DC2626", size=6, symbol="x")))
                fig3.add_hline(y=0, line=dict(color="#94A3B8", width=1))
                fig3.update_layout(title=dict(text="残差与异常检测", x=0.5, xanchor="center"),
                                   xaxis_title="时间", yaxis_title="残差", template="plotly_white",
                                   legend=dict(orientation="h"))
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("异常点表示历史运行中显著偏离模型预期的时刻（如进水冲击、设备异常），可作为运行复盘重点。")

    # ================= 页面9：AI 工艺优化与诊断 =================
    elif page == "🛠️ AI 工艺优化与诊断":
        st.header("🛠️ 智能加药优化 与 异常诊断")
        st.caption("基于线性规划在满足出水标准下最小化药剂成本；超标时给出可解释根因排序")

        bio = get_compute_result("bio_result")
        if not bio:
            st.warning("⚠️ 请先在「🧪 生化核心计算」页面完成计算，本页将读取其结果进行优化与诊断。")
        else:
            bp = st.session_state.base_params
            st.subheader("一、智能加药优化（最小成本方案）")
            if st.button("一键优化投加方案", type="primary", key="opt_btn"):
                st.session_state.opt_result = optimize_dosing(bp, bio)
            if "opt_result" in st.session_state:
                opt = st.session_state.opt_result
                st.write(f"当前碳源缺口 {opt['carbon_deficit']:.1f} mg/L（COD），化学除磷需求 {opt['tp_need_chem']:.2f} mg/L（P）")
                rows = []
                for name, dose in opt["rec_carbon"].items():
                    if dose > 0:
                        rows.append([f"碳源·{name}", f"{dose:.3f} 吨/天"])
                for name, dose in opt["rec_phos"].items():
                    if dose > 0:
                        rows.append([f"除磷剂·{name}", f"{dose:.3f} 吨/天"])
                if not rows:
                    rows = [["无需外加药剂", "0 吨/天（当前已达标）"]]
                st.table(pd.DataFrame(rows, columns=["药剂", "推荐日投加量"]))
                c1, c2, c3 = st.columns(3)
                c1.metric("当前药剂日成本(元)", f"{opt['cur_cost']:.0f}")
                c2.metric("优化后药剂日成本(元)", f"{opt['opt_cost']:.0f}")
                c3.metric("预计可节约", f"{opt['saving']:.0f} 元/天 ({opt['saving_pct']:.1f}%)")
                if opt["saving_pct"] > 1:
                    st.success(f"✅ 在满足出水标准前提下，优化方案预计每日节省约 {opt['saving']:.0f} 元药剂费。")
                else:
                    st.info("当前投加方案已接近成本最优。")

            st.subheader("二、异常诊断（可解释根因排序）")
            issues = diagnose_process(bp, bio)
            if not issues:
                st.success("✅ 未检出明显异常，当前工艺参数处于合理区间。")
            else:
                for title, causes in issues:
                    with st.expander(f"⚠️ {title}", expanded=True):
                        for cause, advice, w in sorted(causes, key=lambda x: -x[2]):
                            st.markdown(f"- **可能原因（置信度 {w * 100:.0f}%）**：{cause}\n\n  → 处置建议：{advice}")

    # ================= 页面10：AI 工艺助手 =================
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
