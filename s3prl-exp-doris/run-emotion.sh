python -m torch.distributed.launch --nproc_per_node 2 \
    run_downstream.py -n ExpMad -m train -u hubert_large_ll60k \
    -d emotion -c downstream/emotion/emotion_config.yaml \
    -o "config.downstream_expert.datarc.test_fold='fold1'"