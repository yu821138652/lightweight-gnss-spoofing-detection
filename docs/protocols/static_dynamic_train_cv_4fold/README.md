# 多环境静态+动态训练、静态 Session-CV v1

该协议以 `../static_session_cv_4fold/` 的 4 折静态 Session 划分为基础：每折的静态 train、val、test 均同时包含新主楼与操场的完整录制 Session。

相对静态联合训练对照，唯一改动是将 `dynamic_train_sessions.csv` 中 10 个动态 Session 追加到每折的 `train`。静态 val/test 完全保持不变，因此同一折的静态 test 指标可直接衡量动态训练数据的影响。

动态来源仅使用既有 mixed 清单中 `split=train` 的动态 Session；原清单中的动态 val/test Session 未加入此协议，保留供后续独立动态测试。
