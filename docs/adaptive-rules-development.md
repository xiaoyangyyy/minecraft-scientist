# 自适应规则与有界随机 Wp：隔离开发阶段

日期：2026-10-08。实现位置：`research/adaptive_automata`，不在主包或默认
测试收集范围内。基础 L# 主包 130 个源码、131 个测试文件的冻结未改变。
本阶段是强基线适配契约的实现进展，不是完整研究、基线优势或创新证明。

## 规则与独立检查

已实现一次性 R/PP 重建阶段，以及 Ex → APS → S → P → AMR → AMS → Eq
优先级。匹配度按公开观测边计算，即使较早输出不符，后续边仍参与比较；
分数用有理数保留并列。旧模型仅指导新实验，不充当当前 epoch 的观测。
参考计划在 dispatch 前绑定时代、树摘要、参考摘要、basis、指导和输入序列。
双查询预检正确计入第一个长序列给第二个短序列带来的前缀缓存。

独立规则检查器使用原始公开表，重算匹配、basis apartness、规则前提、查询
及成本。测试从先前已支付计划与 evidence 重建树快照，不调用学习器的树。
注入“全部参考状态都匹配”的共享缺陷后，学习器仍能 dispatch，但独立检查拒绝。
bool 不能冒充整数模型索引或成本。

## 随机 Wp 及不可隐藏的适配差异

一手核对：[Adaptive L#](https://arxiv.org/pdf/2406.19714) §4–7/Table 3，及
[LearnLib 0.18 源码](https://raw.githubusercontent.com/LearnLib/learnlib/learnlib-0.18.0/oracles/equivalence-oracles/src/main/java/de/learnlib/oracle/equivalence/RandomWpMethodEQOracle.java)。
采样包含均匀状态 cover、最少 3 步加期望额外 3 步的几何随机段，以及各半
概率的全局/局部状态后缀。公开原文并不是只允许免费真值等价 oracle。

本实现使用 Python 随机数和规范 pair-BFS 后缀集，不是 Java/AutomataLib
位级复现。有限 H 下完整记录并拒绝超长样本；不截短，拒绝采样明确改变
原无界分布。达到有限样本/计算上限时，未凑齐合法实验必须 unresolved。
重复或已见输入仍允许采样，但每次验证都真实 reset/执行/支付，不能用缓存
冒充独立试验。未来从头/自适应比较必须共用此协议；当前主包旧基准使用
另一测试器，不能把两批成绩差异直接解释成算法优势。

独立原始采样重放验证 seeded 序列、组成、后缀、完整预测、拒绝与成本；
共享 sampler 预测缺陷也被拒绝。有限通过不代表唯一因果结构或全部 H 认证。

## 本阶段实际证据

Windows `adaptive-rules-windows-v9.xml` 与 Linux `adaptive-rules-wsl-v1.xml`
各 **107 通过、0 失败/错误/跳过**，用例集合一致，两个命令退出 0。
注册绑定 9 个 Python 文件和 README；验证前后哈希一致。完整本地证据：
`runs/adaptive-rules-source-registration-v1.json`、
`runs/adaptive-rules-cross-platform-verification-v1.json`。

显式计数器/初始状态偏移的 8 个单元夹具另导出两次 Windows、一次 Linux。
每次 26 个 JSON 产物及 manifest 完全相同；8/8 夹具的评测行为程序正确。
这些夹具直接提供参考程序先验，不是支付旧世界学习后得到的参考；只做计划
检查，不是完整运行审计、结构留出或方法比较。报告 SHA256：
`c71f4e4b8f4fe9fb2750c824f83bfd4bf79d15c2b3b0d93fd547f2560dd891bc`。

```powershell
./scripts/verify_adaptive_prototype.ps1
uv run pytest research/adaptive_automata -q
uv run python scripts/export_adaptive_fixture_evidence.py --output runs/<new-directory>
```

失败记录保留：v3 是 75 用例中 1 项失败（错误要求单状态夹具必须触发参考
规则）；v6 是 84 用例中 28 项失败（暖缓存 MappingProxy 执行表不能深拷贝）。
已分别修正断言和通过声明式数据重建模型，并新增暖缓存回归。旧失败 XML
不作为当前通过证据，也不删除。

## 尚未关闭的验收门槛

下一步是完整自适应策略/公共账本重放、私有读取前置隔离、控制失败与部分
查询、多类反例程序、支付旧模型学习及在线 epoch 包装，再做同先验/预算/
测试器的强基线比较和预先冻结结构留出。现有两个原始检查器只是计划级
检查，不能冒充完整 auditor。未知控制器、噪声、H4/负迁移、公平 LLM、
受控 Minecraft、自然 RGB/导航/恢复与完整发布继续保留。
