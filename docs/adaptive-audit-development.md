# 自适应学习完整静态运行审计：开发阶段 v2

历史阶段：本页 182 项测试与提供免费参考模型的开发实验，绑定到
`runs/adaptive-audit-source-v1` 的字节快照，不证明后续修改后的当前源码。
新的付费旧模型学习与配对上下文迁移阶段见 `paid-transfer-development.md`。

日期：2026-10-09。源代码在 `research/adaptive_automata`，主包/默认测试
仍冻结；没有借主包默认全量结果替隔离原型验收。本阶段推进强基线实现，
不意味着研究创新、H3/H4、Minecraft/RGB 或完整项目发布已经完成。

## 从计划检查到完整静态运行审计

`audit_run.py` 在私有读取之前先完成两层检查：

1. 独立原始公开账本：查询/复位/动作/事件顺序、时钟、精确成本类型、全批
   预检、控制失败与不完整查询、缓存、时代、证据 lineage、状态晋升的实际
   apartness、basis 和 fold 表、冻结预测、验证批次顺序与最终提案状态。
   调用此前独立原始参考规则与随机 Wp 检查器，不调用其匹配/采样方法。
2. 从提供的公开控制记录完整重放学习策略，比较事件、规则、搜索计数、
   验证和结果。此层使用同一学习器，不能单独排除共享缺陷；第一层因此
   单独存在，保护已测试的共享匹配、缓存、序列化、类型与预测缺陷。

学习日志新增 `basis_promotion`、`rebuilding_closed` 和 `hypothesis_plan`，
把晋升/关闭/折叠的 basis、公开树摘要与理论绑定到证据出现的具体时刻。
这不是事后只审查最终模型。通过后才能读取 Evaluator 私有表，检查成功
控制轨迹并用独立原始配对状态 BFS 评分 H 内行为。

已知终止控制失败真实计费、保留收据，但不写为机制证据、不晋升理论；
失败步本身不强行按完美送达重放私有机制。未知送达/混杂、噪声与非平稳
因子归因不属于当前静态 trusted-reset 契约，不由这里宣称解决。

## 被测试发现并修复的缺陷

序列化器与策略重放共同把 `certified_reset_only=True` 写为整数 `1` 时，
原始条件的 Python 相等检查曾漏过并尝试私有读取。现在用精确类型与
声明程序的规范序列化检查拒绝。最早新审计测试 XML 保留两项失败：
其中目标改写负例最初没有真正改变 target，已改成不同目标；另一项是真实
bool/int 漏检。它们不当作当前通过，也不从实验记录中删除。

共享缓存缺陷把已取得的参考前缀再次当成支付实验时，完整策略重放也会
跟着错误实现一致；独立账本现按先前实际证据拒绝。失败控制、部分验证、
多次反例修订和匿名非计数器程序也有覆盖。一次真实错误提案仍可成为有效
实验记录，其私有行为分数为 false；不能把它隐藏为审计失败。

## 冻结结果与复现

当前冻结绑定 12 个 Python 原型文件、README 和开发 runner，共 14 项；
另绑定未改变的主包 130 源码、131 测试文件。Windows/Linux 各 **182 项
通过，零失败/错误/跳过，用例集合相同，命令退出 0**。证据：
`runs/adaptive-audit-source-registration-v1.json`、
`runs/adaptive-audit-cross-platform-verification-v1.json`。

开发 runner 执行 2 seeds × 4 匿名族 × 2–5 状态，共 32 实例，B=240、H=5，
每次有限验证要求 8 次真实输出查询。固定公开先验是同 opaque 字母表的
二状态 cycle 模型，不按当前真值族/状态数选参考。该先验直接给定，**没有
支付旧模型学习成本**，不能称为在线迁移或与主包的公平方法比较。

- 32 个运行全部通过完整静态审计；24 个有限验证提案，其中 19 个 H 内
  正确、5 个错误；8 个未解决均为查询时域耗尽。
- 错误实例保留：seed 0 dense/4、cycle/4、cycle/5；seed 1 dense/4、cycle/5。
  每个实例都有独立原始私有行为评分，不从有限通过直接推断全部 H 成功。
- 两次 Windows 和一次 Linux 各 162 个确定性 JSON/JSONL 产物逐字节一致，
  manifest 也一致；`timing.json` 独立排除。每次实际 164 文件，144 个源码
  哈希不变。运行/审计 wall 和 CPU 分开保存，部分 visit 计数不冒充总计算。

报告 SHA256：`fdaeccf1d26ff45bd23e97e0a6bebd61acbe65a6fd6fa517f928b7f3ce9d9bb9`。

```powershell
./scripts/verify_adaptive_audit_development.ps1
uv run pytest research/adaptive_automata -q
uv run python scripts/run_adaptive_audit_development.py --output runs/<new-directory>
```

原 107 用例/8 夹具阶段源码已原样保存到 `runs/adaptive-rules-source-v1`。
旧 `verify_adaptive_prototype.ps1` 只核验其历史快照，不把旧证据覆盖到当前
更改后的学习器/审计器。第一次开发导出发生在 runner 边界测试补充前，
也保留，但当前冻结证据仅绑定后续 Windows v2/v3 与 Linux v1。

## 未完成的整体目标与下一动作

新基线下一步要加入支付旧理论学习、共用有界随机 Wp 的从头/自适应模式、
在线变化与 shadow validation/注册表包装，再冻结同先验/成本比较以及
真实结构留出。本开发集不是留出，固定参考不算知识迁移收益。
当前方法尚不能从 19/32 或与旧测试器的成绩差异宣称胜出、创新或完成 H3。

未知/随机控制器、噪声、H4 复用与负迁移、公平 LLM、受控 Minecraft、
自然 RGB/导航/长程恢复和完整发布仍是要求，不由静态基线阶段替代。
基础 L# 主包 Linux 默认全量 v1 在约 49% 非零终止且无 JUnit；失败留存。
确认进程不存活后，同冻结 v2 重跑句柄 87685 仍需终态核验。182 聚焦测试
不是该默认全量或可选 Torch 训练结果。
