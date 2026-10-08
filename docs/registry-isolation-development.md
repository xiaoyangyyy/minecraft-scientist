# 理论版本内容隔离：H3 前置修复

旧 e05811e 中外层 frozen 没有保护内部字典，输入 IR 和 get 返回值都能改变
已保存版本；原失败探针保留在 `runs/h2-source-v1/registry-content-mutation-probe.json`。

注册表现在持有自己的深快照，propose、get、latest、history、all_versions 和
状态转换返回值全部隔离。父版本必须是已存在的正整数；零、负数、布尔值、
浮点和字符串不能通过 Python 索引别名引用有效版本。转换元数据经过模式
校验再发布，不因失败转换改变原状态。

35 项新增测试覆盖七个输入/输出端口的 parameters、effects、scope 嵌套别名，
重复输入的跨版本隔离、非法父/版本号及转换元数据。加 H1/H2、公共类型、
旧解释器和 CLI 兼容回归，Windows/Linux 各 169 项通过，零失败/错误/跳过，
用例集合一致。进程 62196/69353 分别退出 0。新内存探针两次修改后注册表
内容均保持 0，旧失败不是被删除或重写。

全部 122 个包源码与冻结 wheel 按字节匹配，证据为
`runs/revision-foundation-verification-v1.json`，快照为
`runs/revision-foundation-source-v1/minecraft_scientist-0.1.0-py3-none-any.whl`。
后续 H3 源码修改不能自动继承本冻结阶段的覆盖。

这是注册表拥有内容的保证，不是所有 TheoryIR 对象都变成深不可变容器，
也不是防恶意 Python 反射的安全沙箱。独立影子验证、结构生成及版本晋升
仍需实现；此前 1,538 项 Windows 默认通过属于修复前的 H2 源码，不是本阶段
全量回归证据。
