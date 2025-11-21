
使用 scripts/seqrec 下的脚本来测试 seqrec / fusionseqrec / item2index 性能

1. 基本只需使用 case_seqrec.sh 和 metric_ddp.sh
2. 使用 lora 需要加上 --lora, 同时提供 base_model 和 ckpt_model。否则只需要提供 ckpt_model 如果也提供了 base_model 则不会影响加载过程。
3. 不同的 task 使用 --test_task 来控制。默认为 seqrec。
4. metric_ddp.ssh 相比 metric_*.sh 使用了多卡数据并行。同时保存逻辑更完善。


使用 scripts/text_generate 下的脚本来测试 text_enrich 性能

1. lora 使用 evaluate_lora.sh 不是 lora 的使用其他的 evaluate 就行