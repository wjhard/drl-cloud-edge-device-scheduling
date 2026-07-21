# 作品介绍 PPT 自主研究记录

研究日期：2026-07-21

## 1. 学术答辩与研究成果汇报的设计原则

- MIT AeroAstro Communication Lab, “Slide Design”  
  https://mitcommlab.mit.edu/aeroastro/commkit/slide-design/
- MIT NSE Communication Lab, “Doctoral Qualifying Exam Presentation”  
  https://mitcommlab.mit.edu/nse/commkit/doctoral-qualifying-exam-presentation/

采用的判断：每页只承担一个主要结论，并让标题直接表达该结论；建立稳定的视觉层级和左到右阅读顺序；图表按演示场景重新绘制而不是直接截取论文/报告；正文最小字号不低于 16 pt；用统一的字体、配色、网格和页脚；幻灯片是讲解辅助，不是报告全文搬运。

## 2. 强化学习/深度学习技术成果的常见呈现方式

- Henderson et al., “Deep Reinforcement Learning that Matters”, AAAI 2018  
  https://ojs.aaai.org/index.php/AAAI/article/download/11694/11553
- Agarwal et al., “Deep Reinforcement Learning at the Edge of the Statistical Precipice”, NeurIPS 2021  
  https://proceedings.neurips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf
- Google Research, `rliable` 可靠强化学习评测工具与可视化示例  
  https://github.com/google-research/rliable

采用的判断：技术路线按“失败诊断—结构性改动—最终组合”展开；消融实验保持共同基线并同时展示未采用方向；随机算法不只报最好一次，而展示配对重复、均值、样本标准差、差值和显著性检验；对计算预算进行对齐，防止把更多采样误判为算法改进；明确区分算法不变量与经验统计结论。

## 3. GitHub 开源学术模板与配色/排版参考

- Metropolis Beamer theme（低视觉噪声、强调内容空间、可选进度条）  
  https://github.com/matze/mtheme
- Ultimate Beamer Theme List（学术演示主题索引）  
  https://github.com/martinbjeldbak/ultimate-beamer-theme-list

采用的判断：不直接复制模板，而借鉴 Metropolis 的“低噪声、大留白、细进度线、有限强调色”设计语言；本 PPT 使用暖白底、深蓝主色、青绿表示有效改进、橙色表示方法转折、红色表示风险与失败；统一使用微软雅黑/Segoe UI，所有图表使用同一调色板。

## 4. 本作品的具体落地

整体叙事为：“把难以验证的联合策略，逐步改造成有物理规则兜底、可统计验证的混合调度器”。结构依次回答：问题是什么、为什么难、原方案为什么失败、动作空间重构与残差学习如何解决、最终方案如何串联、结果是否稳定且排除算力混淆、哪些方向无效、泛化边界在哪里、如何复现与审计。

所有具体数据来自 `docs/技术报告.docx`，并通过 `evaluation/results/` 下原始 JSON 复核。PPT 不出现姓名、学校或指导教师信息。
