# 简介

本项目基于论文“A step-by-step tutorial on active inference and its application to empirical data”中的原始代码。原始代码使用Matlab语言编写，文件类型为.m。为了推广Friston自由能的思想，我们将代码转换为Python语言。

This project is based on the original code from the paper "A step-by-step tutorial on active inference and its application to empirical data". The original code was written in Matlab with .m file types. To promote Friston's ideas, we have translated the code into Python.

## 论文信息

Smith, R., Friston, K. J., & Whyte, C. J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of mathematical psychology, 107, 102632.

链接：https://www.sciencedirect.com/science/article/pii/S0022249621000973?via%3Dihub

DOI: https://doi.org/10.1016/j.jmp.2021.102632

原matlab脚本仓库：https://github.com/rssmith33/Active-Inference-Tutorial-Scripts

## 目录结构

`original_matlab_code/`: 包含论文中所附原始matlab代码

其他文件说明如下（翻译自原matlab脚本仓库README）：

># 主动推理教程脚本
>
>分步主动推理建模教程的补充脚本
>
>作者：Ryan Smith 和 Christopher Whyte
>
>Step_by_Step_AI_Guide.m：
>
>这是主要的教程脚本。它以一个简单的 explore-exploit 任务为例，说明了如何在主动推理框架中构建部分可观察的马尔可夫决策过程 （POMDP） 模型。它展示了如何运行单试和多试模拟，包括感知、决策和学习。它还展示了如何生成模拟的神>经元反应。它进一步说明了如何将任务模型拟合到行为研究的经验数据中，并进行后续的贝叶斯组分析。注意：此代码已于 24 年 8 月 28 日更新，以改进遗忘率的实施方式。与最初发布的教程不同，此更新版本指定 omega 值越大，遗忘越严重。浓度参数的初始值现在也充当下限，防止这些参数随着时间的推移演变为难以置信的低值。
>
>Step_by_Step_Hierarchical_Model：
>
>单独的脚本说明如何构建分层（深度时间）模型，使用常用的古怪任务范例作为示例。这也显示了如何模拟在实证研究中使用此任务观察到的预测神经元反应（事件相关电位）。
>
>EFE_Precision_Updating：
>
>单独的脚本，允许读取器通过其先前 （beta） 中的更新来模拟预期自由能精度 （gamma） 的更新。在脚本顶部，您可以选择 prior over 策略、预期自由能 over 策略、新观测后 policies 的变分自由能 over 策略的值，以及初始先验 on expected precision 的值。然后，该脚本将模拟 16 次迭代更新，并在 Gamma 中绘制结果变化。通过改变先验和自由能的初始值，你可以对这些更新的动态以及它们如何依赖于所选初始值之间的关系有更多的直觉。
>
>VFE_calculation_example：
>
>单独的脚本，允许读者在给定新观察的情况下计算近似后验信念的变分自由能。读者可以指定一个生成模型（先验和似然矩阵）和一个观察值，然后实验当近似后验信念接近真正的后验信念时如何减少变分自由能。
>
>Prediction_error_example：
>
>允许读者计算状态和结果预测误差的单独脚本。它们分别最小化变分能和预期自由能。最小化状态预测误差可以保持准确的信念（同时也尽可能少地改变信念）。最大限度地减少结果预测误差可以最大限度地提高奖励和信息增益。
>
>Message_passing_example：
>
>允许读者执行 （边际） 消息传递的单独脚本。在第一个示例中，代码逐个遵循正文（第 2 节）中描述的消息传递步骤。在第二个示例中，这被扩展到计算与主动推理相关的神经过程理论中与消息传递相关的发射速率和 ERP。
>
>EFE_learning_novelty_term：
>
>单独的脚本，允许读者在学习似然矩阵 （A） 的狄利克雷浓度参数 （a） 时计算添加到预期自由能中的新颖项。较小的浓度参数会导致新颖性术语的值较大，该值是从策略的总 EFE 值中减去的。因此，对 A 矩阵中状态-结果映射的信念的信心较低，会导致代理选择能够增加对这些信念的信心的策略（“参数探索”）。
>
>Pencil_and_paper_exercise_solutions：
>
>教程论文中提供的 “铅笔和纸 ”练习的解决方案。提供这些是为了帮助读者对主动推理中使用的方程式形成直觉。
>
>spm_MDP_VB_X_tutorial：
>
>运行主动推理 （POMDP） 模型的标准例程的教程版本。注意：此代码已于 24 年 8 月 28 日更新，以改进遗忘率的实施方式。与最初发布的教程不同，此更新版本指定 omega 值越大，遗忘越严重。浓度参数的初始值现在也充当下限，防止这些参数随着时间的推移演变为难以置信的低值。
>
>Simplified_simulation_script：
>
>spm_MDB_VB_X_tutorial 脚本的简化和大量注释版本。提供此功能是为了使读者更容易理解标准仿真例程的工作原理。注意：此代码已于 24 年 8 月 28 日更新，以改进遗忘率的实施方式。与最初发布的教程不同，此更新版本指定 omega 值越大，遗忘越严重。浓度参数的初始值现在也充当下限，防止这些参数随着时间的推移演变为难以置信的低值。
>
>Estimate_parameters：
>
>由主教程脚本调用的脚本，用于估计 （模拟的） 行为数据的参数。
>
>注意： 附加脚本是主脚本调用的辅助函数，用于绘制仿真输出。

## 安装

克隆此仓库：

`git clone git@github.com:ouyangzhiping/feppy.git`

## 使用方法

为了方便学习和可视化演示，本项目使用Jupyter Notebook编写。推荐使用python3.10及以上版本。

## 免责声明

本项目中的Python代码为非官方翻译版本，虽然已经过人工校验，但仍可能与原始Matlab代码存在差异。使用者需自行承担使用过程中可能产生的风险。

The Python code in this project is an unofficial translation and may differ from the original Matlab code. Users should assume responsibility for any risks that may arise during use.

## 贡献

欢迎提交问题（issues）和请求（pull requests）以改进本项目。

Contributions are welcome. Please submit issues and pull requests to improve this project.