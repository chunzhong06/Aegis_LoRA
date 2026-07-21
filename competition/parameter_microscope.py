# Aegis-LoRA: 参数手术显微镜
# 读取清洗审计数据，展示后门定位结果与 LoRA 参数修改范围。
import html
import json
import webbrowser
from pathlib import Path


def show_parameter_microscope(report_path):
    """根据清洗审计 JSON 生成并打开参数手术显微镜。"""
    # -----------------------------------------------------------------
    # 1. 读取并整理参数手术数据
    # -----------------------------------------------------------------
    # report_path 通常是 pipeline 返回的 HTML；json_path 指向同名结构化审计数据。
    report_path = Path(report_path).expanduser().resolve()
    json_path = (
        report_path
        if report_path.suffix == ".json"
        else report_path.with_suffix(".json")
    )

    # report 保存清零数量、目标 Head 与逐层手术记录，是页面的唯一数据源。
    report = json.loads(json_path.read_text(encoding="utf-8"))

    # surgery_rows 只保留实际发生修改的结构，并按层号、结构类型稳定排序。
    surgery_rows = sorted(
        (row for row in report.get("surgery_rows", []) if int(row.get("count", 0)) > 0),
        key=lambda row: (
            (
                int(str(row.get("layer", "")).lstrip("L"))
                if str(row.get("layer", "")).lstrip("L").isdigit()
                else 9999
            ),
            str(row.get("structure", "")),
        ),
    )

    # target_heads 保存最终选中的 Attention Head，按后门签名分数降序展示。
    target_heads = sorted(
        report.get("target_attention_heads", []),
        key=lambda target: float(target.get("score", 0.0)),
        reverse=True,
    )

    # 四个摘要变量依次描述清零规模、影响层数、分数刻度和目标 LoRA 名称。
    suppressed_count = int(report.get("suppressed_count", 0))
    affected_layers = len({str(row.get("layer", "")) for row in surgery_rows})
    max_head_score = max(
        (max(0.0, float(target.get("score", 0.0))) for target in target_heads),
        default=0.0,
    )
    lora_name = Path(str(report.get("lora_path", "LoRA"))).name

    # -----------------------------------------------------------------
    # 2. 生成“找到哪里”与“修改哪里”两组核心内容
    # -----------------------------------------------------------------
    # head_cards 将每个目标 Head 转换为位置、签名分数和相对强度条。
    head_cards = (
        "".join(f"""
        <div class="head-item">
            <div>
                <div class="location">L{html.escape(str(target.get('layer', '-')).lstrip('L'))} · H{int(target.get('head', 0))}</div>
                <div class="module">Q / K / V / O Attention Head</div>
            </div>
            <div class="score">{float(target.get('score', 0.0)):.4f}</div>
            <div class="bar"><span style="width:{max(4.0, float(target.get('score', 0.0)) / max_head_score * 100) if max_head_score else 0:.1f}%"></span></div>
        </div>
        """ for target in target_heads)
        or '<div class="empty">本次手术未选择 Attention Head</div>'
    )

    # surgery_table 展示实际修改结构、清零参数量和手术前后有效范数。
    surgery_table = (
        "".join(f"""
        <tr>
            <td><strong>{html.escape(str(row.get('layer', '-')))}</strong></td>
            <td><span class="badge {html.escape(str(row.get('structure', '')))}">{'Attention' if row.get('structure') == 'attention' else 'MLP'}</span></td>
            <td>{html.escape(', '.join(f'H{head}' for head in row.get('heads', [])) or '神经元通道')}</td>
            <td class="number">{int(row.get('count', 0)):,}</td>
            <td class="norm">{float(row.get('before_norm', 0.0)):.4f}<span>→</span>{float(row.get('after_norm', 0.0)):.4f}</td>
            <td class="drop">-{float(row.get('drop', 0.0)):.2f}%</td>
        </tr>
        """ for row in surgery_rows)
        or '<tr><td colspan="6" class="empty">没有可展示的参数修改记录</td></tr>'
    )

    # -----------------------------------------------------------------
    # 3. 写出单文件页面并交给浏览器展示
    # -----------------------------------------------------------------
    # page 内嵌样式和全部数据片段，保证离线演示无需额外静态资源。
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Aegis 参数手术显微镜</title>
<style>
:root{{--navy:#102A43;--blue:#1677FF;--cyan:#13C2C2;--green:#20A37A;--red:#E5484D;--ink:#243B53;--muted:#6B7C93;--line:#DCE6EF;--panel:#FFFFFF;--bg:#F3F7FA}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Microsoft YaHei",system-ui,sans-serif}}.page{{max-width:1180px;margin:0 auto;padding:28px}}
.hero{{position:relative;overflow:hidden;padding:32px 36px;border-radius:20px;background:linear-gradient(125deg,#0B1F33,#123E5A 62%,#116466);color:white;box-shadow:0 18px 40px rgba(16,42,67,.16)}}
.hero:after{{content:"";position:absolute;width:260px;height:260px;right:-80px;top:-130px;border:55px solid rgba(255,255,255,.07);border-radius:50%}}.eyebrow{{font-size:12px;font-weight:800;letter-spacing:2px;color:#75E6DA}}h1{{margin:8px 0 6px;font-size:32px}}.subtitle{{margin:0;color:#D7E7F2}}.source{{display:inline-flex;margin-top:20px;padding:7px 11px;border:1px solid rgba(255,255,255,.2);border-radius:8px;background:rgba(255,255,255,.08);font-size:13px}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}}.stat{{padding:18px 20px;border:1px solid var(--line);border-radius:14px;background:var(--panel);box-shadow:0 6px 18px rgba(16,42,67,.05)}}.stat-label{{font-size:12px;color:var(--muted)}}.stat-value{{margin-top:6px;font-size:27px;font-weight:800;color:var(--navy)}}.stat-note{{margin-top:3px;font-size:11px;color:#8FA1B3}}
.grid{{display:grid;grid-template-columns:.85fr 1.45fr;gap:18px}}.card{{padding:24px;border:1px solid var(--line);border-radius:16px;background:var(--panel);box-shadow:0 6px 18px rgba(16,42,67,.04)}}.section-tag{{font-size:11px;font-weight:800;letter-spacing:1.5px;color:var(--blue)}}h2{{margin:5px 0 4px;font-size:20px;color:var(--navy)}}.description{{margin:0 0 18px;font-size:13px;color:var(--muted)}}
.heads{{display:grid;gap:10px}}.head-item{{display:grid;grid-template-columns:1fr auto;gap:7px 14px;padding:13px 14px;border:1px solid #E4EDF4;border-radius:11px;background:#FAFCFE}}.location{{font-weight:800;color:var(--navy)}}.module{{margin-top:2px;font-size:11px;color:var(--muted)}}.score{{font-weight:800;color:var(--red)}}.bar{{grid-column:1/-1;height:5px;overflow:hidden;border-radius:6px;background:#E7EEF5}}.bar span{{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,var(--cyan),var(--blue),var(--red))}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{padding:10px;border-bottom:1px solid var(--line);color:var(--muted);font-size:11px;text-align:left;white-space:nowrap}}td{{padding:13px 10px;border-bottom:1px solid #EDF2F6;white-space:nowrap}}.number{{font-weight:800;text-align:right}}.norm span{{padding:0 6px;color:#9AAABC}}.drop{{font-weight:800;color:var(--green)}}.badge{{display:inline-block;padding:4px 8px;border-radius:6px;font-size:11px;font-weight:800}}.badge.attention{{background:#E8F3FF;color:#1264C4}}.badge.mlp{{background:#E7F8F3;color:#13795B}}.empty{{padding:30px;text-align:center;color:var(--muted)}}
.rules{{margin-top:18px}}.rule-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}}.rule{{padding:14px;border:1px solid var(--line);border-radius:11px;background:#FAFCFE}}.rule strong{{display:block;color:var(--navy)}}.rule span{{display:block;margin-top:5px;font-size:12px;color:var(--muted)}}.conclusion{{display:flex;align-items:center;gap:10px;margin-top:16px;padding:13px 15px;border-radius:10px;background:#EAF8F4;color:#116149;font-weight:700;font-size:13px}}.dot{{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(32,163,122,.13)}}
.footer{{padding:18px;text-align:center;color:#91A1B2;font-size:11px}}@media(max-width:900px){{.stats,.rule-grid{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}}}@media(max-width:560px){{.page{{padding:14px}}.hero{{padding:25px 22px}}.stats{{grid-template-columns:1fr 1fr}}h1{{font-size:25px}}}}
</style>
</head>
<body>
<main class="page">
    <header class="hero">
        <div class="eyebrow">AEGIS · PARAMETER SURGERY</div>
        <h1>参数手术显微镜</h1>
        <p class="subtitle">Aegis 具体找到了哪里，又具体修改了哪里？</p>
        <div class="source">目标 LoRA：{html.escape(lora_name)}</div>
    </header>

    <section class="stats">
        <div class="stat"><div class="stat-label">清零 LoRA 参数</div><div class="stat-value">{suppressed_count:,}</div><div class="stat-note">实际置零的 A/B 因子参数</div></div>
        <div class="stat"><div class="stat-label">影响网络层</div><div class="stat-value">{affected_layers}</div><div class="stat-note">发生参数修改的 Transformer Layer</div></div>
        <div class="stat"><div class="stat-label">目标 Attention Head</div><div class="stat-value">{len(target_heads)}</div><div class="stat-note">签名分数最高的定位目标</div></div>
        <div class="stat"><div class="stat-label">修改结构</div><div class="stat-value">{len(surgery_rows)}</div><div class="stat-note">按 Layer 与 MLP/Attention 聚合</div></div>
    </section>

    <section class="grid">
        <article class="card">
            <div class="section-tag">01 · LOCATE</div><h2>Aegis 找到了哪里</h2>
            <p class="description">按后门签名分数排序的 Attention Head 清洗目标</p>
            <div class="heads">{head_cards}</div>
        </article>
        <article class="card">
            <div class="section-tag">02 · OPERATE</div><h2>Aegis 修改了哪里</h2>
            <p class="description">实际发生清零的结构、参数数量与有效更新范数变化</p>
            <div class="table-wrap"><table>
                <thead><tr><th>层</th><th>结构</th><th>目标</th><th style="text-align:right">清零参数</th><th>有效范数</th><th>下降</th></tr></thead>
                <tbody>{surgery_table}</tbody>
            </table></div>
        </article>
    </section>

    <section class="card rules">
        <div class="section-tag">03 · RULE</div><h2>参数手术规则</h2>
        <div class="rule-grid">
            <div class="rule"><strong>Q / K / V Head</strong><span>清零 LoRA B 的对应行</span></div>
            <div class="rule"><strong>O Head</strong><span>清零 LoRA A 的对应列</span></div>
            <div class="rule"><strong>MLP gate / up</strong><span>清零 LoRA B 的目标行</span></div>
            <div class="rule"><strong>MLP down</strong><span>清零 LoRA A 的目标列</span></div>
        </div>
        <div class="conclusion"><span class="dot"></span>手术范围限制在 LoRA A/B 因子，基座模型参数不参与修改。</div>
    </section>
    <footer class="footer">Aegis-LoRA · Competition Showcase</footer>
</main>
</body>
</html>"""

    # output_path 固定指向竞赛结果目录，每次演示覆盖上一份显微镜页面。
    output_path = (
        Path(__file__).resolve().parent / "results" / "parameter_microscope.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")

    # 浏览器打开本地页面后立即返回路径，供 demo 继续进入清洗后复测。
    print(f"\n      [-] [参数手术显微镜] output : {output_path}")
    webbrowser.open(output_path.as_uri())
    return str(output_path)
