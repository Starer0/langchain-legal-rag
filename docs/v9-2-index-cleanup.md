# V9.2：旧索引可见化与受限清理

## 背景

每次资料变化后，系统会建立一个新的版本化 `legal_...` Chroma collection，再让 Manifest 指向新索引。旧 collection 不参与问答，但会继续留在同一个 Chroma 数据目录中。

## 查看状态

```powershell
python main.py --corpus-status
```

除了资料是否变化，该命令现在还会显示：

```text
当前索引：Manifest 正在使用的 collection
可清理的旧索引：未被 Manifest 指向的 legal_... collection
保留的未知索引：不符合项目 legal_ 命名规则的 collection
```

## 清理命令

```powershell
python main.py --prune-stale-indexes
```

该命令只删除同时满足以下条件的 collection：

```text
名称以 legal_ 开头
不等于当前 Manifest 指向的活动 collection
```

因此当前索引不会被删，`langchain` 等未知或早期遗留 collection 也不会被自动删除。清理必须显式运行，不会在导入或问答启动时自动发生。

## 当前本机检查结果（2026-09-29）

```text
活动索引：legal_b82f3cacef8fb93e3729c35d
可安全清理：legal_1be24c10527f23319fc84504、legal_1cf83843546a958c8d608d1c
保留：langchain
```

是否执行清理由用户决定。清理不会影响当前问答索引；它的作用只是回收旧版本索引占用的磁盘空间。
