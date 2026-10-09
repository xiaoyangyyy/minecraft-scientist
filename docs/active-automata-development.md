# 主动结构学习基线：基础 L# 适配开发

这是从头行为结构学习的实际实现，不是完整 Adaptive L#、强基线矩阵、
结构留出或原创方法验收。完整目标继续见 `research-framework-v2.md`。

## 实现与已有方法边界

`discovery/active_automata.py` 使用公开输入/输出前缀构建 observation tree，
以真实输出不同的后缀证明状态 apartness。缺失边不算区分证据。
实现基础提升、扩展、区分、充分树假设构建、乘积一致性检查和二分反例
处理；输出直接编译到现有 total-mealy-v1 / TheoryIR，没有另建理论体系。
这些策略来自 [L# §3](https://arxiv.org/html/2107.05419v4)，不算本项目创新。

查询通过 AgentWorldView.begin_trial()/step() 真实执行。每次复位成本 1，
每个输入按动作成本收费；发现、反例处理、conformance 和已知控制失败进入
同一预算。整条查询与整批验证均先检查可支付性。公开预测及目的在 dispatch
前冻结；失败不插入观测树，同一时代的缓存读取不伪装成新实验。
观测树插入先核对完整输出、一致性和容量，再原子更新；旧时代不能回填。

本实现还没有原文 Hybrid-ADS 测试器，而使用预先声明的有限新查询批次：
最长序列优先、动作覆盖、固定策略 seed 排序；每个候选的整批查询/预测在
执行前冻结。先前已知查询不入批次，后续候选必须取得新验证。
反例可以进入下一轮发现，但不把失败候选重新标为通过。
因此只能返回 `provisional_conformance_passed`，没有 accepted/唯一结构/
全 H 查询保证。预算、时域、容量、非平稳/噪声冲突及运算预算耗尽均明确
报告未解决；没有通过验证时不返回可冒充最终结论的 candidate_ir。

没有参考模型状态匹配、近似匹配、多参考和 Adaptive L# 的完整规则顺序。
不能把它命名为完整 [Adaptive L#](https://arxiv.org/html/2406.19714v1)，也
不能用较弱的测试器在 H3 比较中制造完整方法优势。适配约束仍见
`adaptive-baseline-contract.md`。

## 公开先审计、再私有评分

`evaluation/active_automata.py` 先用公开回执重放策略、规则、批次与预算；
另行检查 episode/step 时钟、回执交叉一致性、唯一 transition ID、动作与
输出字母表、原始前缀树及证据 lineage。独立原始表折叠验证 basis apartness、
frontier 唯一对应、转移及输出；直接用原始行检查冻结预测与受限 IR 契约。
JSON 规范比较区分布尔/整数，不使用 Python 的 True == 1 作为日志一致性。

最初隔离审计批次 46 通过/1 失败：篡改第一个 state_after.step=999 后，
下一次 reset 切断连续性，共享重放仍会沿用假回执而错误打开私有文件。
真实失败报告 `runs/apartness-audit-windows-v1.xml` 保留。增加独立公开时钟与
回执检查后 55 项通过；随后接入主包并扩展类型、共享序列化错误、IR 作用域
和真实错误提案测试。没有降低断言或删除失败记录。

只有公开阶段全部通过才打开 evaluator 私有表，独立重放实际动作结果，并
以有界原始表乘积遍历评分 H 内行为。审计有效不等于模型正确；真实但错误
的 provisional 提案保留为错误指标，而不是排除该运行。专门负例构造未测
序列，在有限验证通过后仍被私有 H 评分判错。Evaluator 不回传反例给 Agent。

## 当前开发协议和结果

`world/active_automata.py` 是四个合成族（dense/sparse/cycle/branch）、2–5 个
隐藏状态、两个种子的开发网格，共 32 个案例。骨架保证状态可达，随机行
和输出提供结构变化；这不是正式结构留出，也不把同 seed 回放当独立样本。
公开标签只依赖 seed/role，与族和状态数无关。Agent 仅获动作/输出、成本、
协议及 capability view，不获候选、真实表、状态数、族或世界 seed。
原始世界逐行执行，不使用学习器执行函数产生真值。

协议 B=240、H=5、conformance_count=8、策略 seed=20261008。开发设计见
`configs/automata-development-v1.json`；不是统计预注册。

| 冻结结果（32 案例） | 数量 |
|---|---:|
| 有限 conformance 通过 | 21 |
| 提案在全部允许 H 内查询上正确 | 18 |
| 有限验证通过、但 H 内预测错误 | 3 |
| 未解决（均 query_horizon_exhausted） | 11 |
| 独立审计有效的完整运行 | 32 |

错误包括两个五状态 cycle 和 seed 1 的四状态 dense。这个负结果说明有限
测试不足以证明 H 完整行为，不能把 21/32 全部报告为机制恢复。没有为了
提高分数加大预算、提供私有反例或隐藏十一项未解决。

Windows v1/v2 与 Linux v1 各运行同一 32 案例；每轮 162 个确定性产物完全
一致，另有 timing.json 明确排除出确定性比较。该文件分别记录完整 learner
调用（含发现、真实验证、构建）的 wall/CPU 时间，以及独立审计时间；
observation_visits 仅是部分搜索计数，不冒充完整计算成本。

新阶段 Windows/Linux 各 342 个聚焦兼容用例通过，零失败/错误/跳过，用例
集合相同；其中新增 81 项，其余是共享 H1/H2/H3/IR/注册表/CLI 兼容测试。
进程 58361/43751 都退出 0。错误测试入口的零执行 XML 也保留，不作为通过。
全部 130 个包源码与注册表及 wheel 按字节匹配，完整证据为
`runs/apartness-cross-platform-verification-v1.json`。可复核：

```powershell
./scripts/verify_automata_development.ps1
uv run minecraft-scientist benchmark-automata --output runs/<new-directory> --seeds 2 --budget 240
```

每个实例保存 config/run/events/audit/evaluator 私有表；根目录保存源码注册、
所有正负结果、实际成本、独立 timing 和 manifest。已有输出拒绝覆盖。
后续包源码变化应使此冻结阶段 verifier 拒绝复用证据。

## 未关闭的完整目标及下一动作

这一步让从头结构学习基线可执行、可独立审计，尚不能算强基线比较完成。
下一步需要忠实 conformance 协议及其预算/反例边界、Adaptive L# 的参考模型
匹配与完整规则、统一在线变化包装、同先验/语法/成本比较和真实结构留出。
三项错误提案要求保留有限验证的失败风险；不能因为 API 返回 provisional
就变更 H3 的验收口径。未知/随机控制器、噪声、H4 迁移和负迁移、公平 LLM、
受控 Minecraft、自然 RGB/导航/恢复及完整发布继续保留，整体目标未完成。

H3 的 1,665 通过/2 跳过默认回归绑定前一阶段 127 文件包，不借给新模块。
本阶段 Windows 默认全量已实际退出 0：1,746 通过、2 项 Torch 跳过；全部
130 个源码和 131 个测试哈希不变，见 `runs/apartness-full-windows-verification-v1.json`。
同冻结 Linux 默认全量 v1 的句柄 8597 在约 49% 返回终态退出码
1073807364，未生成 JUnit；确认内部 pytest 也不再存活，不算通过。失败
记录保留于 `runs/apartness-full-linux-terminal-v1.json`。同 130/131 文件
冻结的 v2（句柄 87685）现已实际退出 0：1748 通过、2 项 Node.js 依赖跳过，
XML 共 1750 项、无失败/错误，内部 pytest 已确认不再存活。核验记录为
`runs/apartness-full-linux-verification-v2.json`。Linux 的三个训练测试和严格
消融均通过；Windows 缺少 Torch，在收集时跳过训练模块（一个占位条目），
并跳过严格消融，所以两平台收集数不同，不能简单宣称全量 case ID 相同。
130 个源/131 个测试未变，另保存 261 文件字节归档供后续修改保留历史证据。
聚焦 342 项不替代默认全量或全部训练能力。隔离自适应原型另见 `adaptive-rules-development.md`，
不属于本阶段主包默认全量，也不算完整强基线矩阵验收。
