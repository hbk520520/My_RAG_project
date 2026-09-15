"""
dataset —— 知识图谱层

阶段 4：加上 __init__.py 使其成为可导入包，这样 project 内其它模块可以统一写
    from dataset.graph import LegalDenseGraphBuilder

而不再需要各自 sys.path 拼路径、或各自复制一份同名实现。

各文件也可以单独作为脚本运行（python dataset/graph.py），因为脚本所在目录
会被自动加入 sys.path，同层的 `from graph import ...` 仍然可用。
"""
